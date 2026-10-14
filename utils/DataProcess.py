import numpy as np
import tensorly as tl
import torch
from numpy.polynomial import chebyshev as cheb
from tensorly.tenalg.core_tenalg import multi_mode_dot

tl.set_backend("pytorch")


def normalize(tensor, normalize_type="zscore"):
    """
    对单个复数张量进行归一化，并返回归一化参数。  归一化参数仅在观测部分计算
    可选的归一化类型：'zscore'（标准化归一化），'minmax'（0-1归一化）和 'robust'（Robust Scaler）。

    参数：
    - tensor: 输入的复数张量。
    - normalize_type: 归一化的类型，'zscore'、'minmax' 或 'robust'。默认为 'zscore'。

    返回：
    - normalized_tensor: 归一化后的张量。
    - normalization_params: 归一化的参数，包括实部和虚部的均值、标准差、最小值、最大值、四分位数等。
    """
    # 计算实部和虚部的均值、标准差
    mu_real = np.mean(tensor.real, axis=1)[:, None, None, None]
    std_real = np.std(tensor.real, axis=1)[:, None, None, None]
    mu_imag = np.mean(tensor.imag, axis=1)[:, None, None, None]
    std_imag = np.std(tensor.imag, axis=1)[:, None, None, None]

    # 0-1归一化的最小值和最大值
    min_real = np.min(tensor.real, axis=1)[:, None, None, None]
    max_real = np.max(tensor.real, axis=1)[:, None, None, None]
    min_imag = np.min(tensor.imag, axis=1)[:, None, None, None]
    max_imag = np.max(tensor.imag, axis=1)[:, None, None, None]

    # 计算四分位数 (IQR)
    q1_real = np.percentile(tensor.real, 25, axis=1)[:, None, None, None]
    q3_real = np.percentile(tensor.real, 75, axis=1)[:, None, None, None]
    q1_imag = np.percentile(tensor.imag, 25, axis=1)[:, None, None, None]
    q3_imag = np.percentile(tensor.imag, 75, axis=1)[:, None, None, None]

    iqr_real = q3_real - q1_real
    iqr_imag = q3_imag - q1_imag

    if normalize_type == "zscore":
        # 进行 Z-Score 归一化（标准化归一化）
        normalized_tensor = np.copy(tensor)
        normalized_tensor.real = (tensor.real - mu_real[:, :, 0, 0]) / std_real[
            :, :, 0, 0
        ]
        normalized_tensor.imag = (tensor.imag - mu_imag[:, :, 0, 0]) / std_imag[
            :, :, 0, 0
        ]
    elif normalize_type == "minmax":
        # 进行 0-1 归一化
        normalized_tensor = np.copy(tensor)
        normalized_tensor.real = (tensor.real - min_real[:, :, 0, 0]) / (
            max_real[:, :, 0, 0] - min_real[:, :, 0, 0]
        )
        normalized_tensor.imag = (tensor.imag - min_imag[:, :, 0, 0]) / (
            max_imag[:, :, 0, 0] - min_imag[:, :, 0, 0]
        )
    elif normalize_type == "robust":
        # 进行 Robust Scaler 归一化
        normalized_tensor = np.copy(tensor)
        normalized_tensor.real = (tensor.real - q1_real[:, :, 0, 0]) / iqr_real[
            :, :, 0, 0
        ]
        normalized_tensor.imag = (tensor.imag - q1_imag[:, :, 0, 0]) / iqr_imag[
            :, :, 0, 0
        ]
    else:
        raise ValueError("normalize_type should be 'zscore', 'minmax', or 'robust'")

    # 返回归一化后的张量及其参数
    normalization_params = {
        "mu_real": mu_real,
        "std_real": std_real,
        "mu_imag": mu_imag,
        "std_imag": std_imag,
        "min_real": min_real,
        "max_real": max_real,
        "min_imag": min_imag,
        "max_imag": max_imag,
        "q1_real": q1_real,
        "q3_real": q3_real,
        "iqr_real": iqr_real,
        "q1_imag": q1_imag,
        "q3_imag": q3_imag,
        "iqr_imag": iqr_imag,
    }

    return normalized_tensor, normalization_params


def denormalize(tensor, normalization_params, normalize_type="zscore"):
    """
    对单个归一化张量进行解归一化，恢复原始值。

    参数：
    - tensor: 归一化后的复数张量。
    - normalization_params: 归一化时保存的参数，包括实部和虚部的均值、标准差、最小值、最大值、四分位数等。
    - normalize_type: 选择归一化的类型 ('zscore'、'minmax' 或 'robust')。

    返回：
    - denormalized_tensor: 解归一化后的张量。
    """
    # 获取归一化参数
    mu_real = normalization_params["mu_real"]
    std_real = normalization_params["std_real"]
    mu_imag = normalization_params["mu_imag"]
    std_imag = normalization_params["std_imag"]
    min_real = normalization_params["min_real"]
    max_real = normalization_params["max_real"]
    min_imag = normalization_params["min_imag"]
    max_imag = normalization_params["max_imag"]
    q1_real = normalization_params["q1_real"]
    # q3_real = normalization_params["q3_real"]
    iqr_real = normalization_params["iqr_real"]
    q1_imag = normalization_params["q1_imag"]
    # q3_imag = normalization_params["q3_imag"]
    iqr_imag = normalization_params["iqr_imag"]

    denormalized_tensor = np.copy(tensor)  # 创建副本避免修改输入张量

    if normalize_type == "zscore":
        # 解 Z-Score 归一化
        denormalized_tensor.real = (tensor.real * std_real) + mu_real
        denormalized_tensor.imag = (tensor.imag * std_imag) + mu_imag
    elif normalize_type == "minmax":
        # 解 0-1 归一化
        denormalized_tensor.real = (tensor.real * (max_real - min_real)) + min_real
        denormalized_tensor.imag = (tensor.imag * (max_imag - min_imag)) + min_imag
    elif normalize_type == "robust":
        # 解 Robust Scaler 归一化
        denormalized_tensor.real = (tensor.real * iqr_real) + q1_real
        denormalized_tensor.imag = (tensor.imag * iqr_imag) + q1_imag
    else:
        raise ValueError("normalize_type should be 'zscore', 'minmax', or 'robust'")

    return denormalized_tensor


def denormalize_core(tensor_core, S_shape, normalization_params):
    """
    对归一化的核心张量进行Z-score解归一化，以恢复其原始值。

    参数：
    - tensor_core: 归一化后的复数核心张量。
    - S_shape: 原始完整张量（在切比雪夫变换之前）的形状。
    - normalization_params: 包含实部和虚部均值与标准差的字典。

    返回：
    - denormalized_tensor: 解归一化后的切比雪夫张量核心。
    """
    # 获取归一化参数
    mu_real = normalization_params["mu_real"]
    std_real = normalization_params["std_real"]
    mu_imag = normalization_params["mu_imag"]
    std_imag = normalization_params["std_imag"]

    # Z-Score 解归一化
    # 合并实部和虚部参数
    mu = mu_real + 1j * mu_imag

    # 构造 M_full：一个形状为 S_shape 的张量，其中每个第 f 层都填充了 mu[f]
    M_full = mu * np.ones(S_shape)
    M_core = tensor_cheb_transform(M_full, tensor_core.shape, S_shape)

    # 分别缩放核心张量的实部和虚部
    G_scaled_real = tensor_core.real * std_real
    G_scaled_imag = tensor_core.imag * std_imag
    G_scaled = G_scaled_real + 1j * G_scaled_imag

    # 将各部分相加，得到解归一化后的切比雪夫张量核心
    denormalized_tensor = G_scaled + M_core

    return denormalized_tensor


def cheb_transform_matrix(N, M=None):
    # 生成切比雪夫变换矩阵并归一化
    if M is None:
        M = N
    k = np.arange(N)
    x = np.cos((2 * k + 1) * np.pi / (2 * N))
    V = cheb.chebvander(x, M - 1)
    D = np.ones(M)
    D[0] = 1 / np.sqrt(N)
    D[1:] = np.sqrt(2 / N)
    C = (V * D).T
    return C  # C^plus = C.T，即C的伪逆等于C的转置


def tensor_cheb_transform(C, Core_shape, S0_shape, device="cpu", scale=1):
    """
    对实数或复数 numpy 数组形式的张量 C 做切比雪夫模式乘积。
    支持 3D 或 4D（多频率）的输入。

    参数：
      C            : shape=(..., Nx, Ny, Nz) 或 (..., r1, r2, r3)
      Core_shape   : 核空间维度大小(..., r1, r2, r3)
      S0_shape     : 原空间维度大小shape=(..., Nx, Ny, Nz)
      device       : torch 设备
      scale        : 超分辨缩放因子

    返回：
      return       : 变换后的张量，shape=(..., r1, r2, r3) 或 (..., Nx, Ny, Nz)
    """
    # 判断是否多个频率
    is_batch = C.ndim == 4
    # 准备空间维度和秩
    ranks = Core_shape[-3:]
    S_shape = tuple(map(int, np.array(S0_shape[-3:]) * scale))

    # 根据 shape 决定正向或逆向
    if not is_batch:
        if C.shape == ranks:
            inv = True
        elif C.shape == S_shape:
            inv = False
        else:
            raise ValueError(
                f"输入 C 的形状 {C.shape} 与 rank {ranks} 或 S_shape {S_shape} 都不匹配"
            )
    else:
        # 取第一个切片确定 inv
        sample = C[0]
        inv = sample.shape == ranks

    # 构造模式积因子
    C_factors = []
    for dim, r in zip(S_shape, ranks):
        M = cheb_transform_matrix(dim, r)
        dtype = torch.cfloat if np.iscomplexobj(M) else torch.float32
        M_t = torch.tensor(M, dtype=dtype, device=device)
        C_factors.append(M_t.T if inv else M_t)

    # 单张量处理函数
    def _process_single(c3d: np.ndarray):
        # 转为 torch 张量
        dtype = torch.cfloat if np.iscomplexobj(c3d) else torch.float32
        C_t = torch.tensor(c3d, dtype=dtype, device=device)
        # 实虚部分开或直接处理
        if C_t.is_complex():
            real = multi_mode_dot(C_t.real, C_factors)
            imag = multi_mode_dot(C_t.imag, C_factors)
            out = real + 1j * imag
        else:
            out = multi_mode_dot(C_t, C_factors)
        # 转回 numpy 并恢复尺度
        return out.cpu().numpy()

    # 批量模式或单张量模式
    if is_batch:
        # 对每一帧复用 C_factors
        return np.stack([_process_single(C[f]) for f in range(C.shape[0])], axis=0)
    else:
        return _process_single(C)
