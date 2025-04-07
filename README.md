# 基于SAM2的心脏腔室CT分割项目 🫀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![SAM2](https://img.shields.io/badge/SAM-2.0-green.svg)](https://github.com/facebookresearch/segment-anything-2)

微调Segment Anything Model 2 (SAM2)用于心脏CT扫描腔室分割任务。本项目提供了用于训练SAM2模型进行心脏腔室分割的工具和指南。

## 📋 目录

- [项目概述](#项目概述)
- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [数据集准备](#数据集准备)
- [使用方法](#使用方法)
- [项目结构](#项目结构)

## 📊 项目概述

本项目专注于心脏CT扫描图像中以下腔室和组织的精确分割:
- LA (左心房)
- RA (右心房)
- LV (左心室)
- RV (右心室)
- 心肌 (Myocardium)

支持不同视角的心脏CT扫描:
- a2c (两腔室视图)
- a3c (三腔室视图)
- a4c (四腔室视图)

## 🔧 环境要求

- Python 3.11.11
- CUDA支持的GPU (推荐)
- Git
- 所需Python包

## 🚀 安装步骤

1. 克隆此仓库:
```bash
git clone https://github.com/yourusername/heart_chambers_sam2.git
cd heart_chambers_sam2
```

2. 运行安装脚本:
```bash
chmod +x installation.sh
./installation.sh
```

此脚本会:
- 克隆并安装SAM2
- 下载必要的模型检查点
- 设置训练环境
- 安装所需依赖
- 创建数据和模型目录

## 📊 数据集准备

1. 准备按视角分类的心脏CT扫描图像和对应的JSON标注文件:
   ```
   data/heart_chambers_dataset/
   ├── a2c/  # 两腔室视图数据
   │   ├── PatientA0001_a2c_27.png
   │   ├── PatientA0001_a2c_27.json
   │   └── ...
   ├── a3c/  # 三腔室视图数据
   │   ├── PatientA0001_a3c_15.png
   │   ├── PatientA0001_a3c_15.json
   │   └── ...
   └── a4c/  # 四腔室视图数据
       ├── PatientA0001_a4c_42.png
       ├── PatientA0001_a4c_42.json
       └── ...
   ```

2. JSON标注格式 (Labelme格式):
   ```json
   {
     "version": "3.16.7",
     "flags": {},
     "shapes": [
       {
         "label": "LV",
         "points": [[x1, y1], [x2, y2], ...],
         "shape_type": "polygon"
       },
       {
         "label": "LA",
         "points": [[x1, y1], [x2, y2], ...],
         "shape_type": "polygon"
       },
       {
         "label": "M",
         "points": [[x1, y1], [x2, y2], ...],
         "shape_type": "polygon"
       }
     ],
     "imagePath": "PatientA0001_a2c_27.png"
   }
   ```

3. 文件命名规范:
   - 文件名中应包含病人ID (例如 PatientA0001)
   - 文件名中应标明视角类型 (a2c, a3c, a4c)
   - 每个图像文件必须有对应的同名JSON标注文件

## 💻 使用方法

1. 进入微调目录:
```bash
cd segment-anything-2/src-finetuning
```

2. 启动并运行微调Jupyter笔记本:
```bash
jupyter notebook heart_chambers_fine_tune.ipynb
```

3. 按照笔记本中的说明:
   - 加载并预处理数据
   - 配置训练参数
   - 训练模型
   - 评估结果
   - 可视化不同视角的分割结果

## 🔧 微调流程
![微调流程图](./heart_chambers_finetuning_schema.jpg)

## 📁 项目结构

```
.
├── installation.sh                          # 安装脚本
├── HeartChambersSAMTrainer.py               # 心脏腔室SAM训练器
├── heart_chambers_fine_tune.ipynb           # 训练和评估Jupyter笔记本
├── data/                                    # 数据目录
│   └── heart_chambers_dataset/
│       ├── a2c/                             # 两腔室视图文件
│       ├── a3c/                             # 三腔室视图文件
│       └── a4c/                             # 四腔室视图文件
└── segment-anything-2/                      # 安装后生成的SAM2代码目录
    ├── checkpoints/                         # 模型检查点目录
    └── src-finetuning/                      # 微调代码目录
        ├── HeartChambersSAMTrainer.py       # 训练器代码副本
        ├── heart_chambers_fine_tune.ipynb   # 笔记本副本
        └── models/                          # 保存训练后模型的目录
```

## ⚠️ 注意事项

1. 标签映射:
   - "LA" - 左心房
   - "RA" - 右心房
   - "LV" - 左心室
   - "RV" - 右心室
   - "M" 或 "myocardium" - 心肌

2. 数据分割策略:
   - 系统会自动按病人ID分割训练集和测试集
   - 同一病人的所有视角数据都会被分到相同的集合中
   - 确保每个病人都有足够数量的不同视角图像

3. 模型选择:
   - 默认使用SAM2 Hiera Small模型
   - 可以在notebooks中修改配置使用其他模型

---
基于心脏医学图像分析创建
