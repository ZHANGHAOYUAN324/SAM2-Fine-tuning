import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast

# 导入SAM2相关模块
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def initialize_sam2_predictor(model_cfg_path, checkpoint_path, device="cuda"):
    """
    初始化SAM2预测器
    
    Args:
        model_cfg_path: SAM2模型配置文件路径
        checkpoint_path: 预训练模型检查点路径
        device: 使用的设备 ('cuda' 或 'cpu')
        
    Returns:
        SAM2ImagePredictor: 初始化好的预测器实例
    """
    # 构建SAM2模型
    sam_model = build_sam2(
        model_cfg_path,
        checkpoint_path,
        device="cuda"
    )
    
    # 创建预测器
    predictor = SAM2ImagePredictor(sam_model)
    
    # 冻结图像编码器参数以防止过拟合
    for param in predictor.model.image_encoder.parameters():
        param.requires_grad = False
    
    print("图像编码器已冻结，仅掩码解码器和提示编码器将进行微调")
    
    return predictor

def read_batch(df, images_dir, masks_dir):
    """
    读取和预处理一批数据
    
    Args:
        df: 包含图像和掩码信息的DataFrame
        images_dir: 图像目录路径
        masks_dir: 掩码目录路径
    
    Returns:
        tuple: (图像, 掩码, 点坐标, 点标签)
    """
    # 随机选择一个图像
    image_ids = df['ImageId'].unique()
    selected_image = np.random.choice(image_ids)
    
    # 获取该图像的所有掩码
    image_masks = df[df['ImageId'] == selected_image]
    
    # 读取图像
    image_path = os.path.join(images_dir, selected_image)
    img = cv2.imread(image_path)[...,::-1]  # BGR to RGB
    
    # 调整图像大小
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    
    masks = []
    points = []
    
    # 处理每个掩码
    for _, row in image_masks.iterrows():
        mask_path = os.path.join(masks_dir, row['MaskId'])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 调整掩码大小，与图像一致
        mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), 
                          interpolation=cv2.INTER_NEAREST)
        
        # 二值化掩码（如果不是二值的）
        binary_mask = (mask > 0).astype(np.uint8)
        masks.append(binary_mask)
        
        # 使用CSV中提供的点坐标，并根据缩放调整
        point_x = int(row['PointX'] * r)
        point_y = int(row['PointY'] * r)
        points.append([[point_x, point_y]])
    
    return img, np.array(masks), np.array(points), np.ones([len(masks), 1])

def train_sam2_model(predictor, df, images_dir, masks_dir, max_iterations=50000, learning_rate=1e-5, weight_decay=4e-5):
    """
    训练SAM2模型
    
    Args:
        predictor: SAM2图像预测器实例
        df: 包含图像和掩码信息的DataFrame
        images_dir: 图像目录路径
        masks_dir: 掩码目录路径
        max_iterations: 最大迭代次数
        learning_rate: 学习率
        weight_decay: 权重衰减率
    """
    # 启用训练模式
    predictor.model.sam_mask_decoder.train(True)  # 启用掩码解码器的训练
    predictor.model.sam_prompt_encoder.train(True)  # 启用提示编码器的训练
    
    # 设置优化器
    optimizer = torch.optim.AdamW(
        params=[p for p in predictor.model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # 设置混合精度训练
    scaler = GradScaler()
    mean_iou = 0
    
    # 创建保存模型的目录
    os.makedirs("models", exist_ok=True)
    
    print(f"开始训练，共{max_iterations}次迭代...")
    
    for itr in range(max_iterations):
        # 使用混合精度训练
        with torch.amp.autocast('cuda'):
            # 加载数据批次
            image, masks, points, labels = read_batch(df, images_dir, masks_dir)
            
            # 忽略空批次
            if len(masks) == 0:
                continue
            
            # 确保数据格式正确
            gt_masks = torch.tensor(masks, dtype=torch.float32, device=predictor.device)
            
            # 对图像应用SAM图像编码器
            predictor.set_image(image)
            
            # 对每个点/掩码对进行处理
            batch_loss = 0
            batch_iou = 0
            
            for i in range(len(points)):
                point = points[i:i+1]
                gt_mask = gt_masks[i:i+1]
                label = labels[i:i+1]
                
                # 准备提示
                mask_input, unnorm_coords, point_labels, unnorm_box = predictor._prep_prompts(
                    point, 
                    label, 
                    box=None, 
                    mask_logits=None, 
                    normalize_coords=True
                )
                
                # 生成嵌入
                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                    points=(unnorm_coords, point_labels),
                    boxes=None,
                    masks=None,
                )
                batched_mode = unnorm_coords.shape[0] > 1
                # 准备高分辨率特征
                high_res_features = [
                    feat_level[-1].unsqueeze(0) 
                    for feat_level in predictor._features["high_res_feats"]
                ]
                
                # 生成掩码
                low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=batched_mode,  # 添加这个参数
                    high_res_features=high_res_features,
                )
                
                # 后处理掩码到原始图像分辨率
                prd_masks = predictor._transforms.postprocess_masks(
                    low_res_masks, 
                    predictor._orig_hw[-1]
                )
                
                # 将logit图转换为概率图
                prd_mask = torch.sigmoid(prd_masks[:, 0])
                
                # 计算交叉熵损失
                seg_loss = (-gt_mask * torch.log(prd_mask + 1e-5) - 
                          (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-5)).mean()
                
                # 计算IoU
                inter = (gt_mask * (prd_mask > 0.5)).sum()
                union = gt_mask.sum() + (prd_mask > 0.5).sum() - inter
                iou = inter / (union + 1e-8)  # 添加小值防止除零
                
                # 计算得分损失
                score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                
                # 混合损失
                mask_loss = seg_loss + score_loss * 0.05
                batch_loss += mask_loss
                batch_iou += iou.item()
            
            # 计算平均损失和IoU
            if len(masks) > 0:
                avg_loss = batch_loss / len(masks)
                avg_iou = batch_iou / len(masks)
            else:
                continue
        
        # 清空梯度
        optimizer.zero_grad()
        
        # 反向传播（使用混合精度）
        scaler.scale(avg_loss).backward()
        
        # 更新权重
        scaler.step(optimizer)
        scaler.update()
        
        # 定期保存模型
        if itr % 1000 == 0 and itr > 0:
            torch.save(predictor.model.state_dict(), f"models/model_{itr}.torch")
        
        # 更新平均IoU（使用指数移动平均）
        if itr == 0:
            mean_iou = avg_iou
        else:
            mean_iou = mean_iou * 0.99 + 0.01 * avg_iou
        
        # 打印训练进度
        if itr % 100 == 0:
            print(f"步骤 {itr}, 准确率 (IoU) = {mean_iou:.4f}, 损失 = {avg_loss.item():.4f}")
    
    # 训练结束，保存最终模型
    torch.save(predictor.model.state_dict(), "models/model_final.torch")
    print(f"训练完成。最终准确率 (IoU) = {mean_iou:.4f}")


# 主程序
if __name__ == "__main__":
    # 数据路径
    csv_path = "/media/ps/data/zhy/Sam2_new/sam2/data_train/train.csv"
    images_dir = "/media/ps/data/zhy/Sam2_new/sam2/data_train/JPEGImages"
    masks_dir = "/media/ps/data/zhy/Sam2_new/sam2/data_train/Annotations"
    
    # SAM2模型路径
    model_cfg_path = "configs/sam2.1/sam2.1_hiera_l.yaml"  # 修改为实际的配置文件路径
    checkpoint_path = "checkpoints/sam2.1_hiera_large.pt"  # 修改为实际的检查点路径
    
    # 加载数据
    df = pd.read_csv(csv_path)
    print(f"加载了{len(df)}条训练数据")
    
    # 初始化SAM2预测器
    predictor = initialize_sam2_predictor(model_cfg_path, checkpoint_path)
    print("SAM2预测器初始化完成")
    
    # 开始训练
    train_sam2_model(predictor, df, images_dir, masks_dir, max_iterations=50000)