# 2D Motion Lifting

## 依赖
执行`pip install -r requirement.txt`安装所需要依赖

## 模型训练
模型配置文件见`configs/base.yaml`; 对denoiser, diffusion, dataset等进行配置

首先需要对数据集进行预处理:
```
python preprocess.py --smpl_dir=/path/to/smpl --source_dir=/path/to/dataset --target_dir=/path/to/save --dataset=beat2
```
> 注: 可能需要对数据预处理代码进行修改, 当前版本没有对translation进行归一化, 且表示格式比较简单粗暴

然后执行以下命令训练模型

```
python train.py --config=/path/to/config.yaml
```

## 采样
模型训练中执行得到随机采样结果, 保存在`sample.npz`中(可以直接加载进blender)
```
python inference.py --config=/path/to/config.yaml
```

Checkpoint位置由`save_dir: /path/to/checkpoint`指定

## Lifting
首先需要运行SMPLer-X得到初始人体姿态估计:
1. `cd`到SMPLer-X目录, 再`cd main`
2. 执行`sh detech.sh ${path_to_video} ${fps} ${checkpoint_name}`, 执行结果保存到对应video名称下`smplx.npz'文件中
3. `cd`到本项目目录下, 执行`python lift.py --config=/path/to/config --input=/path/to/smplx.npz --inference_step=1000`


由于SMPLer-X存在比较严重的depth ambiguity, 所以使用diffusion进行refine的过程需要先对depth进行inpaint, 然后再使用DDIM inversion结合smooth score进行优化, 这个流程参考了以下两篇文章:
1. Score-Guided Diffusion for 3D Human Recovery (CVPR 2024)
2. COIN: Control-Inpainting Diffusion Prior for Human and Camera Motion Estimation (ECCV 2024)