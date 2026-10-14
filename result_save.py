import os

import numpy as np

from utils.TestResultLogger import (
    ReconResultMetricCalculation,
    ReconResultSave,
    ResultSave,
    SurfacePlot,
    SystemMatrixMetricCalculation,
    switchReconRange,
)


def result_plot_save(
    SMdata,
    C_gt,
    C,
    save_path,
    plot_plane=None,
    C_num="5",
    S_completed=None,
    mask=None,
    recon=False,
):
    """保存重建图像对比结果，并可选地保存系统矩阵对比结果。

    ``C_gt`` 和 ``C`` 应为以 ``Shape``、``Resolution``、``Concentration``
    为键的三维图像字典。系统矩阵相关指标和按频率/线圈绘图只有在同时
    传入 ``S_completed`` 和 ``mask`` 时才会执行。
    """
    if not isinstance(C_gt, dict) or not isinstance(C, dict):
        raise TypeError("C_gt and C must be dictionaries keyed by phantom name")
    if set(C_gt) != set(C):
        raise ValueError("C_gt and C must contain the same phantom names")

    c_gt = {phantom: np.asarray(value) for phantom, value in C_gt.items()}
    c_recon = {phantom: np.asarray(C[phantom]) for phantom in c_gt}

    for phantom in c_gt:
        if c_gt[phantom].shape != c_recon[phantom].shape:
            raise ValueError(f"{phantom}: C_gt and C must have the same shape")

    os.makedirs(save_path, exist_ok=True)
    if recon:
        for phantom, c_gt_value in c_gt.items():
            recon_plot_range_x, recon_plot_range_y, recon_plot_range_z = (
                switchReconRange(phantom, C_num)
            )
            ReconMetric = ReconResultMetricCalculation(
                c_recon[phantom],
                dx=recon_plot_range_x,
                dy=recon_plot_range_y,
                dz=recon_plot_range_z,
                phantom=phantom,
                c={phantom: c_gt_value},
            )

            savepath = f"{save_path}/StandardView/{phantom}"
            ReconResultSave(
                savepath=savepath,
                ReconMetric_dict=ReconMetric,
                cfm=c_recon[phantom],
                phantom=phantom,
                c_gt={phantom: c_gt_value},
                recon_plot_range_x=recon_plot_range_x,
                recon_plot_range_y=recon_plot_range_y,
                recon_plot_range_z=recon_plot_range_z,
            )
            print(
                f"{phantom}-PSNR-SSIM:{ReconMetric['psnr_3d_batch'][0]}/{ReconMetric['ssim_3d_batch'][0]}"
            )

    if S_completed is not None:
        if mask is None:
            raise ValueError("mask is required when S_completed is provided")
        SMMetric = SystemMatrixMetricCalculation(S_completed, mask=mask, Su=SMdata.Su)
        ResultSave(
            save_path,
            completed_tensor=S_completed,
            SMMetric_dict=SMMetric,
            save_sm=True,
        )
        print(
            f"nRMSE-PSNR-R2:{SMMetric['total_nrmse'][-1].split('=')[-1]}\t"
            f"{SMMetric['total_psnr'][-1].split('=')[-1]}\t"
            f"{SMMetric['total_r2'][0].split('=')[-1]}"
        )
        if plot_plane is not None:
            K = SMdata.K_select if SMdata.K_select is not None else SMdata.K_train
            SurfacePlot(
                S_completed,
                SMMetric,
                plot_plane,
                K,
                SMdata.Su,
                mask,
                SMdata.GridSize,
                SMdata.SNR,
                SMdata.freq,
                f"{save_path}/",
            )
