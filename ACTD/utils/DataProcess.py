import numpy as np
from ACTD import tensor_cheb_transform

def normalize(tensor):
    """
    对复数张量（观测部分）进行z-score归一化（标准化），并返回归一化参数。

    参数：
    - tensor: 输入的复数张量。
    返回：
    - normalized_tensor: 归一化后的张量。
    - normalization_params: 包含实部和虚部均值与标准差的字典。
    """
    # 计算实部和虚部的均值和标准差
    mu_real = np.mean(tensor.real, axis=1)[:, None, None, None]
    std_real = np.std(tensor.real, axis=1)[:, None, None, None]
    mu_imag = np.mean(tensor.imag, axis=1)[:, None, None, None]
    std_imag = np.std(tensor.imag, axis=1)[:, None, None, None]

    # 进行 Z-Score 归一化
    normalized_tensor = np.copy(tensor)
    # 确保参数的维度与张量维度匹配以便进行广播
    normalized_tensor.real = (tensor.real - mu_real[:, :, 0, 0]) / std_real[:, :, 0, 0]
    normalized_tensor.imag = (tensor.imag - mu_imag[:, :, 0, 0]) / std_imag[:, :, 0, 0]

    # 返回归一化后的张量及其参数
    normalization_params = {
        'mu_real': mu_real,
        'std_real': std_real,
        'mu_imag': mu_imag,
        'std_imag': std_imag,
    }

    return normalized_tensor, normalization_params

def denormalize_core(tensor_core, S_shape, normalization_params):
    """
    对归一化的核心张量进行解归一化，以恢复其原始值。

    参数：
    - tensor_core: 归一化后的复数核心张量。
    - S_shape: 原始完整张量（在切比雪夫变换之前）的形状。
    - normalization_params: 包含实部和虚部均值与标准差的字典。

    返回：
    - denormalized_tensor: 解归一化后的切比雪夫张量核心。
    """
    # 获取归一化参数
    mu_real = normalization_params['mu_real']
    std_real = normalization_params['std_real']
    mu_imag = normalization_params['mu_imag']
    std_imag = normalization_params['std_imag']

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
