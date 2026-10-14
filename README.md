# ACTD

本项目实现论文中的自适应张量补全系统矩阵建模与快速重建流程。论文链接：

<https://ieeexplore.ieee.org/abstract/document/11576603/>

## 项目结构

```text
ACTD/
├── ACTD.py                         # ACTD 模型和优化过程
├── test.py                         # 系统矩阵补全和重建示例
├── result_save.py                  # 结果保存和可视化入口
├── environment.yml                 # Conda 环境配置
├── requirements.txt                # Python 依赖
├── Dataset/
│   └── SystemMatrixLoad.py         # MDF 数据读取、采样掩码生成
├── Recon/
│   └── KaczmarzReg.py              # 原域与变换域中的 Kaczmarz 重建算法
├── utils/
│   ├── DataProcess.py              # 归一化和变换域处理
│   ├── TestResultLogger.py         # 指标计算、结果保存和绘图
│   ├── metric_calculator.py        # NRMSE、PSNR、SSIM 等指标
│   └── loggerx.py                  # 日志工具
├── OpenMPIData/                    # 原始 OpenMPIData 的 MDF 数据
```

## 环境安装

推荐使用 Python 3.12，并根据本机 CUDA 版本安装匹配的 PyTorch。例如，CUDA 12.6 可执行：

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

然后安装其余依赖：

```bash
pip3 install tqdm tensorly h5py scikit-image matplotlib
```


## 原始数据

由于 MDF 原始数据体积较大，`OpenMPIData/` 不包含在本仓库中。运行 `test.py` 前，请从 TUHH 的 OpenMPIData 数据页下载原始数据：

<https://media.tuhh.de/ibi/openMPIData/data/>

需要下载并放置到本地项目中的文件如下。

```text
OpenMPIData/
├── Calibrations/
│   ├── 5.mdf
│   └── 6.mdf
└── Measurements/
	├── concentration/
	│   ├── 2.mdf
	│   └── 3.mdf
	├── resolution/
	│   ├── 2.mdf
	│   └── 3.mdf
	└── shape/
		├── 2.mdf
		└── 3.mdf
```



⚠️注意：`Calibrations/5.mdf` 和 `Calibrations/6.mdf` 是校准系统矩阵数据，其分别对应Measurements中`2.mdf` 和 `3.mdf`的观测数据，所以只能取`Calibration_num = 5, Meas_num = 2`或者`Calibration_num = 6, Meas_num = 3`。


## 运行示例

准备好原始数据后，在项目根目录执行：

```bash
# 使用 Calibration-5 和 Measurement-2（默认值）
python test.py --calibration-num 5 --meas-num 2 --sparsity 0.0625 --snr 7

# 或使用 Calibration-6 和 Measurement-3
python test.py --calibration-num 6 --meas-num 3 --sparsity 0.0625 --snr 7
```

其中，`--calibration-num`、`--meas-num`、`--sparsity` 和 `--snr` 分别设置校准文件编号、测量文件编号、采样稀疏度和信噪比。省略参数时使用默认值：`5`、`2`、`0.0625` 和 `7`。


## 输出结果

示例代码将结果保存到 `save_path` 指定的目录。系统矩阵结果和重建结果由
`result_plot_save` 的参数分别控制：

- 传入 `S_completed` 和 `mask` 时，始终保存 `CompletedSystemMatrix.npy` 和
	`SMMetric.json`；
- `plot_plane` 不为 `None` 时，额外保存各线圈在指定系统矩阵平面上的二维对比图；
- `plot_plane=None` 时，不生成 `coil-rx/`、`coil-ry/` 或 `coil-rz/` 文件夹；
- `recon=True` 时，保存三类 phantom 的重建结果、指标和二维切片图；
- `recon=False` 时，不生成 `StandardView/` 及其下的切片文件夹。

例如，当前 `test.py` 使用 `plot_plane=19`、`recon=True`，输出结构如下：

```text
Test Results/
└── snr-7/
    └── Calibration-5/
        └── result_0.0625/
        ├── CompletedSystemMatrix.npy
        ├── SMMetric.json
        ├── coil-rx/
        │   └── K=*.png                 # rx 线圈，系统矩阵第 19 层
        ├── coil-ry/
        │   └── K=*.png                 # ry 线圈，系统矩阵第 19 层
        ├── coil-rz/                    # 使用 rz 线圈时生成
        │   └── K=*.png
        └── StandardView/
            ├── Shape/
            │   ├── ReconResult.npy
            │   ├── ReconMetric.json
            │   ├── xy-slice/
            │   │   ├── xy=8.png
            │   │   └── ... xy=25.png
            │   ├── xz-slice/
            │   │   ├── xz=14.png
            │   │   └── ... xz=21.png
            │   └── yz-slice/
            │       ├── yz=12.png
            │       └── ... yz=28.png
            ├── Resolution/              # 文件结构同 Shape，范围为 xy=16..19、xz=10..29、yz=13..27
            └── Concentration/           # 文件结构同 Shape，范围为 xy=22..25、xz=9..14、yz=13..26
```

其中每张 `xy=*.png`、`xz=*.png` 或 `yz=*.png` 都是一个二维对比图，包含原始系统矩阵重建结果、补全系统矩阵重建结果以及残差图。对于 `Calibration-5`，各 phantom 的切片范围为：

```text
Shape:         yz x=12..28, xz y=14..21, xy z=8..25
Resolution:    yz x=13..27, xz y=10..29, xy z=16..19
Concentration: yz x=13..26, xz y=9..14,  xy z=22..25
```

例如，`Shape/xy-slice/xy=8.png` 表示 Shape phantom 在 `z=8` 处的二维 `xy` 平面对比图。

当设置 `plot_plane=None`、仍保持 `recon=True` 时，输出中不会有任何 `coil-*` 文件夹，但会保留 `StandardView/`；当设置 `recon=False`、仍传入 `S_completed` 和 `mask` 时，只保存系统矩阵结果：

```text
snr-7/Calibration-5/result_0.0625/
├── CompletedSystemMatrix.npy
└── SMMetric.json
```



## 备注

> 切比雪夫基替换为其他基（例如离散余弦基）可能得到类似效果。
```python
def dct_transform_matrix(N, M=None):
    """正交归一化 DCT-II 基矩阵（输出形状与 cheb_transform_matrix 相同： (M, N) ）"""
    if M is None:
        M = N
    n = np.arange(N)
    k = np.arange(M)
    # DCT-II 基函数：phi_k(n) = cos(pi*(n+0.5)*k/N)
    V = np.cos(np.pi * (n[:, None] + 0.5) * k[None, :] / N)
    # 归一化
    D = np.ones(M)
    D[0] = 1 / np.sqrt(N)
    D[1:] = np.sqrt(2 / N)
    C = (V * D).T
    return C
```
