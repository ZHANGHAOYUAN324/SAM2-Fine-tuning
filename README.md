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
1. 克隆sam2仓库以及安装配置文件
```bash
git clone https://github.com/facebookresearch/segment-anything-2.git

cd sam2; pip install -e .
```
2. 下载所有模型点：
```bash
cd checkpoints
./download_ckpts.sh
```

3. 克隆此仓库:
```bash
cd sam2
(https://github.com/ZHANGHAOYUAN324/SAM2-Fine-tuning.git)
```

## 📊 数据集准备

1. 准备心脏CT扫描图像以及对应掩码的的JSON标注文件:
   ```
   data_train/
   ├── a2c/  # 两腔室视图数据
   │   ├── PatientA0001_a2c_27.json
   │   └── ...
   ├── a3c/  # 三腔室视图数据
   │   ├── PatientA0001_a3c_15.json
   │   └── ...
   └── a4c/  # 四腔室视图数据
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
cd sam2
```

2. 启动并运行微调Jupyter笔记本:
```bash
jupyter notebook data_process.ipynb
```

3. 按照笔记本中的说明:
   - 加载并预处理数据
   - 生成可供sam2训练的原始图像以及掩码图像

4.开始使用数据训练sam2模型
```bash
python training.py
```
5.使用训练好的模型进行预测生成原始掩码图，原始模型图以及微调模型图
```bash
python testing.py
```


## 📁 项目结构

```
├── assets
│   ├── model_diagram.png
│   └── sa_v_dataset.jpg
├── backend.Dockerfile
├── checkpoints
│   ├── download_ckpts.sh
│   ├── sam2.1_hiera_base_plus.pt
│   ├── sam2.1_hiera_large.pt
│   ├── sam2.1_hiera_small.pt
│   └── sam2.1_hiera_tiny.pt
├── check.py
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── data_check.ipynb
├── data_train
│   ├── Annotations
│   ├── JPEGImages
│   ├── train.csv
│   └── Visualization
├── demo
│   ├── backend
│   ├── data
│   ├── frontend
│   └── README.md
├── docker-compose.yaml
├── example.ipynb
├── INSTALL.md
├── LICENSE
├── LICENSE_cctorch
├── MANIFEST.in
├── Miniconda3-latest-MacOSX-arm64.sh
├── models
│   ├── model_10000.torch
│   ├── model_1000.torch
│   ├── model_11000.torch
│   ├── model_12000.torch
│   ├── model_13000.torch
│   ├── model_14000.torch
│   ├── model_15000.torch
│   ├── model_16000.torch
│   ├── model_17000.torch
│   ├── model_18000.torch
│   ├── model_19000.torch
│   ├── model_20000.torch
│   ├── model_2000.torch
│   ├── model_21000.torch
│   ├── model_22000.torch
│   ├── model_23000.torch
│   ├── model_24000.torch
│   ├── model_25000.torch
│   ├── model_26000.torch
│   ├── model_27000.torch
│   ├── model_28000.torch
│   ├── model_29000.torch
│   ├── model_30000.torch
│   ├── model_3000.torch
│   ├── model_31000.torch
│   ├── model_32000.torch
│   ├── model_33000.torch
│   ├── model_34000.torch
│   ├── model_35000.torch
│   ├── model_36000.torch
│   ├── model_37000.torch
│   ├── model_38000.torch
│   ├── model_39000.torch
│   ├── model_40000.torch
│   ├── model_4000.torch
│   ├── model_41000.torch
│   ├── model_42000.torch
│   ├── model_43000.torch
│   ├── model_44000.torch
│   ├── model_45000.torch
│   ├── model_46000.torch
│   ├── model_47000.torch
│   ├── model_48000.torch
│   ├── model_49000.torch
│   ├── model_5000.torch
│   ├── model_6000.torch
│   ├── model_7000.torch
│   ├── model_8000.torch
│   ├── model_9000.torch
│   └── model_final.torch
├── notebooks
│   ├── automatic_mask_generator_example.ipynb
│   ├── image_predictor_example.ipynb
│   ├── images
│   ├── video_predictor_example.ipynb
│   └── videos
├── pyproject.toml
├── README.md
├── RELEASE_NOTES.md
├── results_comparison
│   └── a4c_PatientD0062_a4c_93.jsonD_116
├── sam2
│   ├── automatic_mask_generator.py
│   ├── benchmark.py
│   ├── build_sam.py
│   ├── configs
│   ├── csrc
│   ├── __init__.py
│   ├── modeling
│   ├── __pycache__
│   ├── sam2_hiera_b+.yaml -> configs/sam2/sam2_hiera_b+.yaml
│   ├── sam2_hiera_l.yaml -> configs/sam2/sam2_hiera_l.yaml
│   ├── sam2_hiera_s.yaml -> configs/sam2/sam2_hiera_s.yaml
│   ├── sam2_hiera_t.yaml -> configs/sam2/sam2_hiera_t.yaml
│   ├── sam2_image_predictor.py
│   ├── sam2_video_predictor_legacy.py
│   ├── sam2_video_predictor.py
│   └── utils
├── SAM_2.egg-info
│   ├── dependency_links.txt
│   ├── PKG-INFO
│   ├── requires.txt
│   ├── SOURCES.txt
│   └── top_level.txt
├── sam2_training_visualization.png
├── sav_dataset
│   ├── example
│   ├── LICENSE
│   ├── LICENSE_DAVIS
│   ├── LICENSE_VOS_BENCHMARK
│   ├── README.md
│   ├── requirements.txt
│   ├── sav_evaluator.py
│   ├── sav_visualization_example.ipynb
│   └── utils
├── setup.py
├── testing.py
├── tools
│   ├── README.md
│   └── vos_inference.py
├── training
│   ├── assets
│   ├── dataset
│   ├── __init__.py
│   ├── loss_fns.py
│   ├── model
│   ├── optimizer.py
│   ├── __pycache__
│   ├── README.md
│   ├── scripts
│   ├── trainer.py
│   ├── train.py
│   └── utils
├── training.ipynb
├── training.py
└── training_vis.py
```

## ⚠️ 注意事项

1. 标签映射:
   - "LA" - 左心房
   - "RA" - 右心房
   - "LV" - 左心室
   - "RV" - 右心室
   - "M" 或 "myocardium" - 心肌


2. 模型选择:
   - 默认使用SAM2 Hiera large模型
   - 可以在notebooks中修改配置使用其他模型

---
基于心脏医学图像分析创建
