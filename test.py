import argparse

import numpy as np
import torch
from tqdm import tqdm

from ACTD import ACTD_optimization
from Dataset.SystemMatrixLoad import SystemMatrixLoad, generate_mask

# from Recon.KaczmarzReg import kaczmarz_tensor
from Recon.KaczmarzReg import kaczmarz_tensor, kaczmarz_tensor_cheb
from result_save import result_plot_save
from utils.DataProcess import denormalize_core, normalize, tensor_cheb_transform


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ACTD system matrix completion and reconstruction."
    )
    parser.add_argument(
        "--calibration-num",
        type=int,
        default=5,
        help="Calibration file number (default: 5).",
    )
    parser.add_argument(
        "--meas-num",
        type=int,
        default=2,
        help="Measurement file number (default: 2).",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.0625,
        help="Sampling sparsity between 0 and 1 (default: 0.0625).",
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=7,
        help="Signal-to-noise ratio (default: 7).",
    )
    args = parser.parse_args()
    if not 0 < args.sparsity <= 1:
        parser.error("--sparsity must be greater than 0 and at most 1")
    return args


def tensor_completion_ACTD_pytorch(S_downsampled, mask, ranks, lr):
    if not isinstance(S_downsampled, torch.Tensor):
        raise TypeError("Input matrix must be a PyTorch Tensor.")

    if S_downsampled.shape[1:] != mask.shape:
        raise ValueError("Mask shape must match the spatial dimensions of the tensor.")

    def completion(obS, mask):
        # 这里调训练参数，主要调ranks, ref_F和reg_PQR
        tensor_filled_Cheb = ACTD_optimization(
            obS, mask, ranks, lr=lr, max_iter=5000, device="cuda:0", reg_F=1, reg_PQR=1
        )
        return tensor_filled_Cheb

    S = [
        completion(S_downsampled[i], mask) for i in tqdm(range(S_downsampled.shape[0]))
    ]

    S_completed_cheb = np.stack(S, axis=0)
    return S_completed_cheb


if __name__ == "__main__":
    args = parse_args()
    Calibration_num = args.calibration_num
    Meas_num = args.meas_num
    sparsity = args.sparsity
    snr = args.snr
    Phantom_type = ["Shape", "Resolution", "Concentration"]
    ## 原始数据中读取系统矩阵和观测数据
    SMdata = SystemMatrixLoad(
        f"OpenMPIData\\Calibrations\\{Calibration_num}.mdf",
        [
            f"OpenMPIData\\Measurements\\shape\\{Meas_num}.mdf",
            f"OpenMPIData\\Measurements\\resolution\\{Meas_num}.mdf",
            f"OpenMPIData\\Measurements\\concentration\\{Meas_num}.mdf",
        ],
        snr,
        sparsity,
        "random",
        None,
        [True, True, False],  # (x, y, z)-coil是否使用
        Phantom_type=Phantom_type,
    )
    GridSize = SMdata.GridSize
    S = SMdata.Su.copy()  # 系统矩阵
    u = SMdata.u  # 观测数据

    ## 生成采样掩码并对观测数据进行归一化
    mask, _ = generate_mask(GridSize, sparsity=sparsity, sampling_method="random")
    S_downsampled_flat, norm_param = normalize(S[:, mask], "zscore")
    S_downsampled = np.zeros_like(S)
    S_downsampled[:, mask] = S_downsampled_flat

    # 进行ACTD优化
    lr = 0.05
    ranks = [15, 15, 15]
    S_completed_cheb_ = tensor_completion_ACTD_pytorch(
        torch.from_numpy(S_downsampled), torch.from_numpy(mask), ranks, lr=lr
    )
    Core_shape = tuple(ranks)
    S_completed_cheb = denormalize_core(S_completed_cheb_, S.shape, norm_param)
    print("Completed tensor_cheb shape:", S_completed_cheb.shape)
    S_completed = tensor_cheb_transform(S_completed_cheb, Core_shape, S.shape)
    print("Completed tensor shape:", S_completed.shape)

    phantoms_gt = {}
    phantoms = {}
    for phantom in Phantom_type:
        C_gt = kaczmarz_tensor(
            S,
            u[phantom],
            iterations=3,
            lambd=0.75,
            enforceReal=False,
            enforcePositive=True,
            shuffle=True,
            tv_iters=1,
            dt=0.1,
            eps=1,
        ).real

        # 在变换域中重建
        C = kaczmarz_tensor_cheb(
            S_completed_cheb,
            u[phantom],
            Core_shape=Core_shape,
            S0_shape=S.shape,
            iterations=3,
            lambd=0.75,
            enforceReal=False,
            enforcePositive=True,
            shuffle=True,
            tv_iters=1,
            dt=0.1,
            eps=1,
            scale=1,
        ).real

        # 原域中重建
        # C = kaczmarz_tensor(
        #     S_completed, u,
        #     iterations=3,
        #     lambd=0.75,
        #     enforceReal=False,
        #     enforcePositive=True,
        #     shuffle=True,
        #     tv_iters=1,
        #     dt=0.1,
        #     eps=1,
        # ).real

        phantoms_gt[phantom] = C_gt
        phantoms[phantom] = C

    result_plot_save(
        SMdata,
        C_gt=phantoms_gt,
        C=phantoms,
        save_path=f"Test Results/snr-{snr}/Calibration-{Calibration_num}/result_{sparsity}",
        plot_plane=None,  # None表示不绘制，19表示绘制系统矩阵第19层对比
        C_num=f"{Calibration_num}",
        S_completed=S_completed,
        mask=mask,
        recon=True,  # True表示绘制重建图像与基准对比图
    )
