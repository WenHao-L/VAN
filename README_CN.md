# 目录

<!-- TOC -->

- [目录](#目录)
- [Visual Attention Network描述](#Visual_Attention_Network描述)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
- [训练和测试](#训练和测试)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [ImageNet-1k上的Visual Attention Network](#imagenet-1k上的Visual_Attention_Network)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# [Visual_Attention_Network描述](#目录)

图像的二维特性给计算机视觉中的自我注意应用带来了三个挑战。(1)将图像视为一维序列，忽略了其二维结构；(2)二次复杂度对于高分辨率图像来说过于昂贵；(3)仅捕获空间适应性，忽略了通道适应性。

基于上述问题，作者提出了一种新的大核注意（LKA）模型，在避免上述问题的同时，实现自我注意中的自适应和远程相关性。并进一步介绍了一种基于LKA的新型神经网络，即视觉注意网络（VAN）。

# [数据集](#目录)

使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：共1000个类、224*224彩色图像
    - 训练集：共1,281,167张图像
    - 测试集：共50,000张图像
- 数据格式：JPEG
    - 注：数据在dataset.py中处理。
- 下载数据集，目录结构如下：

 ```text
└─dataset
    ├─train                 # 训练数据集
    └─val                   # 评估数据集
```

# [特性](#目录)

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/experts/zh-CN/master/others/mixed_precision.html)
的训练方法，使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。

# [环境要求](#目录)

- 硬件（Ascend）
    - 使用Ascend来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/r1.3/index.html)

# [脚本说明](#目录)

## 脚本及样例代码

```bash
├── Visual_Attention_Network
  ├── README_CN.md                        // Visual Attention Network相关说明
  ├── src
      ├──configs                          // Visual Attention Network的配置文件
      ├──data                             // 数据集配置文件
          ├──imagenet.py                  // imagenet配置文件
          ├──augment                      // 数据增强函数文件
          ┕──data_utils                   // modelarts运行时数据集复制函数文件
  │   ├──models                           // 模型定义文件夹
          ┕──van                          // Visual Attention Network定义文件
  │   ├──trainers                         // 自定义TrainOneStep文件
  │   ├──tools                            // 工具文件夹
          ├──callback.py                  // 自定义回调函数，训练结束测试
          ├──cell.py                      // 一些关于cell的通用工具函数
          ├──criterion.py                 // 关于损失函数的工具函数
          ├──get_misc.py                  // 一些其他的工具函数
          ├──optimizer.py                 // 关于优化器和参数的函数
          ┕──schedulers.py                // 学习率衰减的工具函数
  ├── train.py                            // 训练文件
  ├── eval.py                             // 评估文件
  ├── export.py                           // 导出模型文件
  ├── postprocess.py                      // 推理计算精度文件
  ├── preprocess.py                       // 推理预处理图片文件

```

## 脚本参数

- 配置Visual Attention Network和ImageNet-1k数据集。

  ```python
    # Architecture
    arch: van_base_224                  # Visual Attention Network结构选择
    # ===== Dataset ===== #
    data_url: ./data                    # 数据集地址
    set: ImageNet                       # 数据集名字
    num_classes: 1000                   # 数据集分类数目
    mix_up: 0.8                         # MixUp数据增强参数
    cutmix: 1.0                         # CutMix数据增强参数
    auto_augment: rand-m9-mstd0.5-inc1  # AutoAugment参数
    interpolation: bicubic              # 图像缩放插值方法
    re_prob: 0.25                       # 数据增强参数
    re_mode: pixel                      # 数据增强参数
    re_count: 1                         # 数据增强参数
    mixup_prob: 1.                      # 数据增强参数
    switch_prob: 0.5                    # 数据增强参数
    mixup_mode: batch                   # 数据增强参数
    # ===== Learning Rate Policy ======== #
    optimizer: adamw                    # 优化器类别
    base_lr: 0.001                      # 基础学习率
    warmup_lr: 0.000006                 # 学习率热身初始学习率
    min_lr: 0.000006                    # 最小学习率
    lr_scheduler: cosine_lr             # 学习率衰减策略
    warmup_length: 5                    # 学习率热身轮数
    nonlinearity: GELU                  # 激活函数类别
    # ===== Network training config ===== #
    amp_level: O2                       # 混合精度策略
    beta: [ 0.9, 0.999 ]                # adamw参数
    clip_global_norm_value: 5.          # 全局梯度范数裁剪阈值
    is_dynamic_loss_scale: True         # 是否使用动态缩放
    epochs: 400                         # 训练轮数
    label_smoothing: 0.1                # 标签平滑参数
    weight_decay: 0.05                  # 权重衰减参数
    momentum: 0.9                       # 优化器动量
    batch_size: 128                     # 批次
    # ===== Hardware setup ===== #
    num_parallel_workers: 32            # 数据预处理线程数
    device_target: Ascend               # Ascend npu
    # ===== Model config ===== #        # 模型结构参数
    drop_path_rate: 0.1
    embed_dims: [64, 128, 320, 512]
    mlp_ratios: [8, 8, 4, 4]
    depths: [3, 3, 12, 3]
    num_stages: 4
    image_size: 224                     # 图像大小
  ```



# [训练和测试](#目录)

- Ascend处理器环境运行

  ```bash
  # 在openi平台使用多卡训练
  python train.py --device_num 8 --dataset_sink_mode True --run_openi True --device_target Ascend

  # 在openi平台使用单卡评估
  python eval.py --device_num 1 --device_target Ascend --run_openi True --pretrained True
  ```

# [模型描述](#目录)

## 性能

### 评估性能

#### ImageNet-1k上的Visual_Attention_Network

| 参数 | Ascend |
| ---- | ------ |
| 模型 | Visual Attention Network |
| 模型版本 | van_base_224 |
| 资源 | Ascend 910 |
| 上传日期 | 2022-7-11 |
| MindSpore版本 | 1.5.1 |
| 数据集 | ImageNet-1k Train，共1,281,167张图像 |
| 训练参数| epoch=400, batch_size=128 |
| 优化器 | AdamWeightDecay |
| 损失函数 | SoftTargetCrossEntropy |
| 损失| 0.835328337779412 |
| 输出 | 概率 |
| 分类准确率 | 八卡：top1: 82.87660256410256% , top5: 96.28205128205128% |
| 速度 | 八卡：559.653毫秒/步 |
| 训练耗时 | 87h11min57s（run on OpenI）|

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)