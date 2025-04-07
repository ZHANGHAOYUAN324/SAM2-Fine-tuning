import os
import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast

import re
import glob

# 导入SAM2相关模块
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def inference_sam2_compare_models(
    image_path,
    mask_paths,  
    model_cfg,
    sam2_checkpoint,
    fine_tuned_path=None,
    num_samples_per_mask=10,
    output_dir=None,  
    show_results=False,
    device="cuda"
):



    def read_image(image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
            
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        img = img[..., ::-1]  # BGR to RGB
        
        # 将图像调整到最大尺寸1024
        resize_factor = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
        img = cv2.resize(img, (int(img.shape[1] * resize_factor), int(img.shape[0] * resize_factor)))
        return img, resize_factor
    
    # 在输入掩码内采样点
    def get_points(mask, num_points):
        points = []
        coords = np.argwhere(mask > 0)
        
        if len(coords) == 0:
            print("警告：掩码为空，无法采样点")
            return np.array([])
            
        for _ in range(num_points):
            if len(coords) > 0:
                idx = np.random.randint(len(coords))
                yx = coords[idx]
                points.append([[yx[1], yx[0]]])
        
        return np.array(points) if points else np.array([])
    
    # 读取图像
    image, resize_factor = read_image(image_path)
    
    # 读取所有掩码并采样点
    all_points = []
    all_point_labels = []
    all_masks = []
    
    for i, mask_path in enumerate(mask_paths):
        if not os.path.exists(mask_path):
            print(f"警告：掩码文件不存在: {mask_path}，将跳过")
            continue
            
        mask = cv2.imread(mask_path, 0)  # 读取掩码
        if mask is None:
            print(f"警告：无法读取掩码: {mask_path}，将跳过")
            continue
            
        # 调整掩码大小以匹配图像 - 使用resize_factor而不是r
        mask = cv2.resize(mask, (int(mask.shape[1] * resize_factor), int(mask.shape[0] * resize_factor)), 
                         interpolation=cv2.INTER_NEAREST)
        
        all_masks.append(mask)
        
        # 获取该掩码的点
        points = get_points(mask, num_samples_per_mask)
        if len(points) > 0:
            all_points.append(points)
            all_point_labels.append(np.ones((len(points), 1)))
    
    if not all_points:
        raise ValueError("所有掩码都无法采样点，无法进行分割")
    
    
    # 合并所有掩码的点和标签
    input_points = np.concatenate(all_points, axis=0)
    input_labels = np.concatenate(all_point_labels, axis=0)
    
    # 创建真实掩码的可视化
    ground_truth_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    ground_truth_rgb = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    
    # 为每个掩码分配唯一的颜色和ID
    colors = {}
    
    for i, mask in enumerate(all_masks):
        # 生成随机颜色，但确保所有可视化中使用相同颜色
        if i+1 not in colors:
            colors[i+1] = [
                np.random.randint(255),
                np.random.randint(255),
                np.random.randint(255)
            ]
        
        # 更新掩码图和RGB图
        ground_truth_map[mask > 0] = i + 1
        ground_truth_rgb[mask > 0] = colors[i+1]
    
    # 创建真实掩码的混合图像
    gt_blended_image = (ground_truth_rgb / 2 + image / 2).astype(np.uint8)
    
    # 初始化结果字典
    results = {
        "ground_truth": {
            "map": ground_truth_map,
            "rgb": ground_truth_rgb,
            "blended": gt_blended_image
        }
    }
    
    # 使用两个模型进行推理
    for model_key, model_path in [
        ("original", sam2_checkpoint), 
        ("fine_tuned", fine_tuned_path)
    ]:
        # 如果是微调模型但路径为None，则跳过
        if model_key == "fine_tuned" and model_path is None:
            continue
            
        print(f"\n使用{model_key}模型进行推理...")
        
        # 初始化SAM2模型
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        
        # 如果是微调模型，加载微调权重
        if model_key == "fine_tuned":
            try:
                predictor.model.load_state_dict(torch.load(model_path, weights_only=True))
            except:
                predictor.model.load_state_dict(torch.load(model_path))
            print(f"已加载微调模型: {model_path}")
        
        # 设置模型为评估模式
        predictor.model.eval()
        
        # 使用混合精度进行推理
        with torch.no_grad():
            # 图像编码器
            predictor.set_image(image)
            
            print(f"{model_key}模型正在进行预测...")
            print("输入点形状:", input_points.shape)
            
            # prompt编码器 + mask解码器
            masks, scores, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True
            )
            
            print(f"{model_key}模型预测完成")
            print(f"预测掩码数量: {masks.shape[0] if hasattr(masks, 'shape') else '未知'}")
        
        # 处理masks和scores
        if isinstance(masks, torch.Tensor):
            np_masks = masks.cpu().numpy()
            if np_masks.ndim > 3:
                np_masks = np_masks[:, 0]
        else:
            np_masks = masks
            if np_masks.ndim > 3:
                np_masks = np_masks[:, 0]
        
        if isinstance(scores, torch.Tensor):
            np_scores = scores.cpu().numpy()
            if np_scores.ndim > 1:
                np_scores = np_scores[:, 0]
        else:
            np_scores = scores
            if np_scores.ndim > 1:
                np_scores = np_scores[:, 0]
        
        # 根据分数排序掩码（降序）
        sorted_indices = np.argsort(np_scores)[::-1]
        sorted_masks = np_masks[sorted_indices]
        
        # 创建分割图
        seg_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        occupancy_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
        
        for i, mask in enumerate(sorted_masks):
            # 确保掩码是布尔类型
            mask_bool = mask.astype(bool)
            
            # 计算重叠比例
            # 使用逻辑与运算而不是位与运算
            overlap_ratio = np.logical_and(mask_bool, occupancy_mask).sum() / (mask_bool.sum() + 1e-6)
            
            if mask_bool.sum() > 0 and overlap_ratio > 0.15:
                continue
                
            # 更新掩码和分割图
            non_overlap = np.logical_and(mask_bool, np.logical_not(occupancy_mask))
            seg_map[non_overlap] = i + 1
            occupancy_mask[mask_bool] = True
        
        # 创建可视化图像（使用与真实掩码相同的颜色）
        rgb_image = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
        for id_class in range(1, seg_map.max() + 1):
            # 尝试使用与真实掩码相同的颜色
            color = colors.get(id_class, [
                np.random.randint(255),
                np.random.randint(255),
                np.random.randint(255)
            ])
            rgb_image[seg_map == id_class] = color
        
        # 创建混合图像
        blended_image = (rgb_image / 2 + image / 2).astype(np.uint8)
        
        # 创建差异图 (显示预测与真实值的不同)
        difference = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        
        # 绿色表示正确预测的区域
        correct = np.logical_and(ground_truth_map > 0, seg_map > 0)
        difference[correct] = [0, 255, 0]
        
        # 红色表示未预测到的区域 (漏检)
        missed = np.logical_and(ground_truth_map > 0, seg_map == 0)
        difference[missed] = [255, 0, 0]
        
        # 蓝色表示错误预测的区域 (误检)
        false_positive = np.logical_and(ground_truth_map == 0, seg_map > 0)
        difference[false_positive] = [0, 0, 255]
        
        # 保存到结果字典
        results[model_key] = {
            "map": seg_map,
            "rgb": rgb_image,
            "blended": blended_image,
            "difference": difference
        }
    
    # 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存真实掩码
        cv2.imwrite(os.path.join(output_dir, "gt_segmentation.png"), 
                    cv2.cvtColor(results["ground_truth"]["rgb"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, "gt_blended.png"), 
                    cv2.cvtColor(results["ground_truth"]["blended"], cv2.COLOR_RGB2BGR))
        
        # 保存原始模型结果
        if "original" in results:
            cv2.imwrite(os.path.join(output_dir, "original_segmentation.png"), 
                        cv2.cvtColor(results["original"]["rgb"], cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(output_dir, "original_blended.png"), 
                        cv2.cvtColor(results["original"]["blended"], cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(output_dir, "original_difference.png"), 
                        cv2.cvtColor(results["original"]["difference"], cv2.COLOR_RGB2BGR))
        
        # 保存微调模型结果
        if "fine_tuned" in results:
            cv2.imwrite(os.path.join(output_dir, "fine_tuned_segmentation.png"), 
                        cv2.cvtColor(results["fine_tuned"]["rgb"], cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(output_dir, "fine_tuned_blended.png"), 
                        cv2.cvtColor(results["fine_tuned"]["blended"], cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(output_dir, "fine_tuned_difference.png"), 
                        cv2.cvtColor(results["fine_tuned"]["difference"], cv2.COLOR_RGB2BGR))
        
        # 创建三模型对比图 (并排显示)
        # 计算每个模型的列宽
        col_width = image.shape[1]
        models_count = 1 + ("original" in results) + ("fine_tuned" in results)
        
        comparison = np.zeros((image.shape[0], col_width * models_count, 3), dtype=np.uint8)
        
        # 添加真实掩码
        col_idx = 0
        comparison[:, col_idx*col_width:(col_idx+1)*col_width] = results["ground_truth"]["blended"]
        col_idx += 1
        
        # 添加原始模型结果
        if "original" in results:
            comparison[:, col_idx*col_width:(col_idx+1)*col_width] = results["original"]["blended"]
            col_idx += 1
        
        # 添加微调模型结果
        if "fine_tuned" in results:
            comparison[:, col_idx*col_width:(col_idx+1)*col_width] = results["fine_tuned"]["blended"]
        
        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        col_idx = 0
        cv2.putText(comparison, "Ground Truth", (col_idx*col_width + 10, 30), font, 1, (255, 255, 255), 2)
        col_idx += 1
        
        if "original" in results:
            cv2.putText(comparison, "Original Model", (col_idx*col_width + 10, 30), font, 1, (255, 255, 255), 2)
            col_idx += 1
        
        if "fine_tuned" in results:
            cv2.putText(comparison, "Fine-tuned Model", (col_idx*col_width + 10, 30), font, 1, (255, 255, 255), 2)
        
        cv2.imwrite(os.path.join(output_dir, "model_comparison.png"), 
                    cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        
        print(f"所有结果已保存到 {output_dir} 目录")
    
    # 显示结果（在有GUI的环境中）
    if show_results:
        try:
            if "original" in results and "fine_tuned" in results:
                cv2.imshow("模型对比", cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
            print("按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"无法显示图像，可能是在无GUI环境中运行: {e}")
            print("结果已保存到指定目录")
    
    return results


def find_related_masks_advanced(image_path, masks_dir):
        """
        使用更高级的方法查找与图像相关的掩码
        """
        
        # 获取图像文件名（不含扩展名）
        image_filename = os.path.basename(image_path)
        image_name, _ = os.path.splitext(image_filename)
        
        # 基本情况：掩码名以图像名开头
        pattern1 = os.path.join(masks_dir, f"{image_name}*.png")
        masks1 = glob.glob(pattern1)
        
        # 从图像名中提取患者ID和序列号，用于更复杂的匹配
        # 例如: a4c_PatientD0062_a4c_93.jsonD_116.jpg
        # 提取: PatientD0062, 93, 116
        matches = re.search(r'(Patient[A-Za-z0-9]+).*?(\d+)\.json[A-Z]_(\d+)', image_name)
        
        if matches:
            patient_id = matches.group(1)
            seq_num1 = matches.group(2)
            seq_num2 = matches.group(3)
            
            # 使用提取的信息尝试其他可能的匹配模式
            pattern2 = os.path.join(masks_dir, f"*{patient_id}*{seq_num1}*{seq_num2}*.png")
            masks2 = glob.glob(pattern2)
            
            # 合并去重
            all_masks = list(set(masks1 + masks2))
        else:
            all_masks = masks1
        
        print(f"为图像 {image_filename} 找到 {len(all_masks)} 个相关掩码:")
        for mask in all_masks:
            print(f"  - {os.path.basename(mask)}")
        
        return all_masks
 
    
    # 读取图像
def process_image_by_name(data_dir, image_filename, model_cfg, sam2_checkpoint, fine_tuned_path=None, output_dir=None):
        """
        通过图像文件名处理单个图像，自动查找相关掩码
        
        参数:
            data_dir: 数据根目录
            image_filename: 图像文件名（不含路径）
            model_cfg: 模型配置文件路径
            sam2_checkpoint: 原始模型检查点路径
            fine_tuned_path: 微调模型路径
            output_dir: 输出目录
        """
        images_dir = os.path.join(data_dir, "JPEGImages")
        masks_dir = os.path.join(data_dir, "Annotations")
        
        # 构建完整图像路径
        image_path = os.path.join(images_dir, image_filename)
        
        if not os.path.exists(image_path):
            print(f"图像文件不存在: {image_path}")
            return
        
        # 自动查找相关掩码
        mask_paths = find_related_masks_advanced(image_path, masks_dir)
        
        if not mask_paths:
            print("未找到相关掩码文件")
            return
        
        # 设置输出目录（使用图像名作为子目录）
        if output_dir is None:
            image_name, _ = os.path.splitext(image_filename)
            output_dir = os.path.join("results", image_name)
        
        # 运行模型比较
        results = inference_sam2_compare_models(
            image_path=image_path,
            mask_paths=mask_paths,
            model_cfg=model_cfg,
            sam2_checkpoint=sam2_checkpoint,
            fine_tuned_path=fine_tuned_path,
            output_dir=output_dir,
            show_results=False
        )
        
        return results


if __name__ == "__main__":
    # 设置基本路径
    data_dir = "/media/ps/data/zhy/Sam2_new/sam2/data_train"
    images_dir = os.path.join(data_dir, "JPEGImages")
    masks_dir = os.path.join(data_dir, "Annotations")
    
    # 模型配置
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    fine_tuned_path = "models/model_final.torch"
    
    # 指定图像文件
    image_filename = "a4c_PatientD0062_a4c_93.jsonD_116.jpg"
    image_path = os.path.join(images_dir, image_filename)
    
    print(f"检查图像文件...")
    if not os.path.exists(image_path):
        print(f"图像文件不存在: {image_path}")
        exit(1)
    
    # 自动查找相关掩码
    mask_paths = find_related_masks_advanced(image_path, masks_dir)
    
    # 检查掩码文件
    valid_masks = []
    for mask_path in mask_paths:
        if os.path.exists(mask_path):
            valid_masks.append(mask_path)
            print(f"掩码文件存在: {mask_path}")
        else:
            print(f"掩码文件不存在: {mask_path}")
    
    if not valid_masks:
        print("没有有效的掩码文件")
        exit(1)
    
    # 检查其他文件
    for path, desc in [
        (model_cfg, "模型配置文件"),
        (sam2_checkpoint, "预训练检查点"),
        (fine_tuned_path, "微调模型")
    ]:
        exists = os.path.exists(path)
        print(f"{desc}: {path} {'存在' if exists else '不存在'}")
    
    # 设置输出目录
    output_dir = os.path.join("results_comparison", os.path.splitext(image_filename)[0])
    
    try:
        # 运行推理
        print("\n开始模型对比推理...")
        results = inference_sam2_compare_models(
            image_path=image_path,
            mask_paths=valid_masks,
            model_cfg=model_cfg,
            sam2_checkpoint=sam2_checkpoint,
            fine_tuned_path=fine_tuned_path,
            num_samples_per_mask=10,
            output_dir=output_dir,
            show_results=False
        )
        print("对比推理完成!")
        
        # 输出简单的性能指标
        if "original" in results and "fine_tuned" in results:
            gt_map = results["ground_truth"]["map"]
            
            for model_key in ["original", "fine_tuned"]:
                model_map = results[model_key]["map"]
                
                # 计算基本指标
                true_positive = np.sum(np.logical_and(gt_map > 0, model_map > 0))
                false_negative = np.sum(np.logical_and(gt_map > 0, model_map == 0))
                false_positive = np.sum(np.logical_and(gt_map == 0, model_map > 0))
                
                # 计算性能指标
                precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
                recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"\n{model_key.capitalize()}模型性能:")
                print(f"精确率(Precision): {precision:.4f}")
                print(f"召回率(Recall): {recall:.4f}")  
                print(f"F1分数: {f1:.4f}")
    
    except Exception as e:
        print(f"推理过程中出错: {e}")
        import traceback
        traceback.print_exc()