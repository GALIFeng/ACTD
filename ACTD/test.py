import numpy as np
import torch
from tqdm import tqdm
from Recon.kaczmarzReg_cheb import kaczmarz_tensor_cheb
from ACTD import ACTD_optimization,tensor_cheb_transform

from utils.DataProcess import normalize,denormalize_core

def generate_observed_tensor(tensor, sparsity=0.1, sampling_method="random", downsample_factor=None):
    np.random.seed(0)
    if sampling_method == "random":
        # 随机生成观测掩码
        mask = np.random.rand(*tensor.shape[1:]) < sparsity
    elif sampling_method == "downsample":
        # 生成下采样掩码：每隔downsample_factor取一个值
        mask = np.zeros(tensor.shape[1:], dtype=bool)
        # 遍历张量每个维度，进行下采样
        mask[::downsample_factor[0], ::downsample_factor[1], ::downsample_factor[2]] = True
    else:
        raise ValueError(f"Unsupported sampling_method '{sampling_method}'. Use 'random' or 'downsample'.")
    # 构建部分观测张量
    observed_tensor = np.copy(tensor)
    observed_tensor.imag[:, ~mask] = np.nan  # 用 NaN 表示缺失值
    observed_tensor.real[:, ~mask] = np.nan
    return observed_tensor, mask

def tensor_completion_ACTD_pytorch(S_downsampled, mask, ranks, lr): 
    if not isinstance(S_downsampled, torch.Tensor):
        raise ValueError("Input matrix must be a PyTorch Tensor.")

    if S_downsampled.shape[1:] != mask.shape:
        raise ValueError("Mask shape must match the spatial dimensions of the tensor.")

    def completion(obS, mask):
        # 这里调训练参数，主要调ranks, ref_F和reg_PQR
        tensor_filled_Cheb = ACTD_optimization(obS, mask, ranks, lr=lr, max_iter=5000,
                                                 device="cuda:0", reg_F=1, reg_PQR=1)
        return tensor_filled_Cheb

    S = [completion(S_downsampled[i], mask) for i in tqdm(range(S_downsampled.shape[0]))]
    

    S_completed_cheb = np.stack(S,axis=0)
    return S_completed_cheb


if __name__ == "__main__":
    #  这里改标定序号和模体，C=5，M=2或者C=6，M=3
    Calibration_num = 6
    Meas_num = 3
    Phantom = "Shape"   # 切换模体：Shape, Resolution, Concentration

    scale = 1       # 改尺度重建
    snr = 7
    lr = 0.05
    ranks = [15, 15, 15]

    sparsity = 0.0625   # 改观测稀疏度：0.015625
    downsample = 1      # 采样方式
    sampling_method = "random"if downsample else "downsample"
    downsample_factor = None if downsample else (2, 2, 2)

    S = np.load(f"Data\\S_Calibration{Calibration_num}_snr{snr}.npy")  # 读取原始数据
    u = np.load(f"Data\\{Phantom}{Meas_num}_snr{snr}.npy")
    S0_shape = S.shape
    Core_shape = tuple(ranks)

    # 归一化
    _, mask = generate_observed_tensor(S, sparsity=sparsity, sampling_method=sampling_method, downsample_factor=downsample_factor)
    S_downsampled_flat, norm_param = normalize(S[:, mask])    
    S_downsampled = np.zeros_like(S)
    S_downsampled[:, mask] = S_downsampled_flat
    S_completed_cheb_ = tensor_completion_ACTD_pytorch(torch.from_numpy(S_downsampled), torch.from_numpy(mask),ranks,lr=lr) 

    # 系统矩阵去归一化
    S_completed_cheb = denormalize_core(S_completed_cheb_, S0_shape, norm_param)   
    # S_completed = tensor_cheb_transform(S_completed_cheb, Core_shape, S0_shape)
    # S_completed_HR = tensor_cheb_transform(S_completed_cheb, Core_shape, S0_shape, scale=scale)

    # 进行Kaczmarz重建
    C = kaczmarz_tensor_cheb(
        S_completed_cheb, u,
        Core_shape=Core_shape,
        S0_shape=S0_shape,
        iterations=3,
        lambd=0.75,
        enforceReal=False,
        enforcePositive=True,
        shuffle=True,
        tv_iters=1,
        dt=0.1,
        eps=1,
        scale=scale
    ).real







    


    
