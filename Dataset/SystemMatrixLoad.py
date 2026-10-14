import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.DataProcess import normalize


def generate_mask(
    GridSize, sparsity=0.1, sampling_method="random", downsample_factor=None
):
    """
    根据给定的稀疏度或下采样倍数生成观测掩码及其索引。

    参数：
    - GridSize: 原始张量的形状，应为包含三个维度的序列。
    - sparsity: "random"或"grouped_random"采样时的观测比例，取值范围为0到1。
    - sampling_method: 采样方式，可选"random"、"downsample"或"grouped_random"，默认为"random"。
    - downsample_factor: 均匀下采样的步长序列，例如(2, 4, 2)；
      仅当sampling_method="downsample"时有效。

    返回：
    - mask: 观测掩码（布尔数组），True表示已观测，False表示已缺失。
    - mask_indices: 观测位置的索引张量，形状为(N, 3)，每行对应一个True位置。
    """
    np.random.seed(0)
    if sampling_method == "random":
        # 随机生成观测掩码
        mask = np.random.rand(*GridSize) < sparsity
    elif sampling_method == "downsample":
        # 生成下采样掩码：每隔downsample_factor取一个值
        mask = np.zeros(GridSize, dtype=bool)
        # 遍历张量每个维度，进行下采样
        mask[
            :: downsample_factor[0], :: downsample_factor[1], :: downsample_factor[2]
        ] = True
    elif sampling_method == "grouped_random":
        n = 1 / sparsity
        assert n % 1 == 0
        n = int(n)
        total_slices = GridSize[0]
        mask = np.zeros(GridSize, dtype=bool)

        # 每组包含 n 个切片，分组采样，组内位置不重复
        for group_start in range(0, total_slices, n):
            group_end = min(group_start + n, total_slices)
            group_masks = np.zeros((n, GridSize[1], GridSize[2]), dtype=bool)

            for slice_idx in range(group_end - group_start):
                # 每个切片随机采样 int[1/sparsity] 的点，并确保组内切片不重叠
                num_points = int(np.prod(GridSize[1:]) * sparsity)
                available_indices = np.argwhere(
                    ~group_masks.any(axis=0)
                )  # 获取当前未被占用的点
                if len(available_indices) < num_points:
                    raise ValueError(
                        "Not enough points available to sample without overlap within a group."
                    )
                selected_indices = available_indices[
                    np.random.choice(len(available_indices), num_points, replace=False)
                ]
                group_masks[
                    slice_idx, selected_indices[:, 0], selected_indices[:, 1]
                ] = True

            # 将组内的掩码合并到总掩码中
            mask[group_start:group_end, :, :] = group_masks[
                : group_end - group_start, :, :
            ]

    else:
        raise ValueError(
            f"Unsupported sampling_method '{sampling_method}'. "
            "Use 'random', 'downsample', or 'grouped_random'."
        )
    mask_indices = torch.nonzero(torch.from_numpy(mask), as_tuple=False)

    return mask, mask_indices


class CustomArray:
    def __init__(self, dataset: h5py.Dataset, isBG):
        """
        dataset: h5py.Dataset, 存储在 HDF5 文件中的数据
        isBG: 1D 布尔数组，表示哪些点是背景点
        """
        self.dataset = dataset  # h5py.Dataset (未加载到内存)
        self.isBG = np.asarray(isBG, dtype=bool)  # 确保 isBG 是布尔数组
        self.valid_indices = np.where(~self.isBG)[0]  # 预计算非背景点的索引

    def __getitem__(self, key):
        """
        访问 s[c, k] 时，支持 k 为整数或数组，等价于 S[:, c, k, ~isBG].squeeze()
        """
        if isinstance(key, tuple) and len(key) == 2:
            c, k = key
            k = np.atleast_1d(k)  # 确保 k 是 1D 数组

            # 从 h5py 读取完整列，但只选取指定的 k 频率
            full_data = self.dataset[:, c, k, :]  # (N, len(k), M)

            # 过滤掉背景点
            filtered_data = full_data[:, :, ~self.isBG]  # 只保留非背景点
            return np.squeeze(filtered_data)  # squeeze 以去除单一维度
        else:
            raise IndexError("Only indexing as s[c, k] is supported.")


class SystemMatrixLoad(Dataset):
    def __init__(
        self,
        SMPath,
        MeasPath,
        SNRThreshold,
        sparsity,
        sampling_method,
        K_select,
        use_coil,
        Phantom_type=None,
        downsample_factor=None,
    ):
        super().__init__()
        self.datatype = "train"
        self.filenameSM = SMPath
        self.Threshold = SNRThreshold
        self.loadSMh5()
        self.mask, _ = generate_mask(
            self.GridSize, sparsity, sampling_method, downsample_factor
        )
        self.loadSMData(K_select, use_coil)
        if MeasPath is not None:
            self.loadMeasData(MeasPath, Phantom_type=Phantom_type)
        self.Nk = len(self)

    def loadSMData(self, K_select, use_coil):
        self.use_coil = use_coil
        self.K_select = K_select
        K_use__ = []
        for coil, kset in enumerate(self.K_filter):
            for ks in kset:
                K_use__.append(f"coil:{coil}-k:{ks}")
        use_coil_expand = (
            [use_coil[0]] * len(self.K_filter[0])
            + [use_coil[1]] * len(self.K_filter[1])
            + [use_coil[2]] * len(self.K_filter[2])
        )
        K_use = [sublist for sublist, flag in zip(K_use__, use_coil_expand) if flag]
        Su = []
        if K_select is not None:
            assert set(K_select).issubset(K_use), (
                f"Assertion failed: Some elements in K_select are not in K_use: {K_select}"
            )
            for i, ck in enumerate(K_select):
                c = int(ck.split("-")[0].split(":")[1])
                k = int(ck.split("-")[1].split(":")[1])
                Su.append(self.S[c, k].copy())
            Su = np.vstack(Su).reshape(len(K_select), *self.GridSize)
        else:
            Su = [
                self.S[idx, ku].copy().reshape(self.S[idx, ku].shape[0], *self.GridSize)
                for idx, (ku, flag) in enumerate(zip(self.K_filter, use_coil))
                if flag
            ]
            Su = np.vstack(Su)
        self.Su = Su
        self.Su_observed_norm, _ = normalize(Su[:, self.mask], normalize_type="zscore")
        self.K_train = K_use
        self.K_train_Num = [len(k) for k in self.K_filter]

    def loadSMh5(self):
        fSM = h5py.File(self.filenameSM, "r")
        # read the full system matrix
        S = fSM["/measurement/data"]
        # reinterpret to complex data
        isBG = fSM["/measurement/isBackgroundFrame"][:].view(bool)
        self.S = CustomArray(S, isBG)
        # get rid of background frames

        # generate frequency vector
        numFreq = round(fSM["/acquisition/receiver/numSamplingPoints"][()] / 2) + 1
        rxBandwidth = fSM["/acquisition/receiver/bandwidth"][()]
        self.freq = np.arange(0, numFreq) / (numFreq - 1) * rxBandwidth
        # # remove frequencies below 80 kHz and use only x/y receive channels
        idxMin = np.nonzero(np.ravel(self.freq[:] > 30e3))[0][0]
        SM_SNR_x = 10 * np.log10(fSM["/calibration/snr"][0, 0])
        SM_SNR_y = 10 * np.log10(fSM["/calibration/snr"][0, 1])
        SM_SNR_z = 10 * np.log10(fSM["/calibration/snr"][0, 2])
        self.GridSize = fSM["/calibration/size"][:]
        self.SNR = [SM_SNR_x, SM_SNR_y, SM_SNR_z]
        K_filter = [
            np.where(SM_SNR_x >= self.Threshold)[0],
            np.where(SM_SNR_y >= self.Threshold)[0],
            np.where(SM_SNR_z >= self.Threshold)[0],
        ]
        self.K_filter = [
            K_filter[0][K_filter[0] > idxMin],
            K_filter[1][K_filter[1] > idxMin],
            K_filter[2][K_filter[2] > idxMin],
        ]

    def loadMeasData(self, MeasPath, Phantom_type):
        self.u = {}
        assert len(MeasPath) == len(Phantom_type)
        for mea_path, phantomType in zip(MeasPath, Phantom_type):
            filenameMeas = mea_path
            fMeas = h5py.File(filenameMeas, "r")
            u = fMeas["/measurement/data"]
            u = u[:, :, :, :].squeeze()
            if len(u.shape) == 4:
                u_t = np.mean(u[0], axis=0)
            else:
                u_t = np.mean(u, axis=0)
            uk = np.fft.rfft(
                u_t
            )  # 默认最后一个轴，返回长度为：奇数：（N+1）/2，偶数：N/2 +1
            u = [
                uk[0, self.K_filter[0]].copy(),
                uk[1, self.K_filter[1]].copy(),
                uk[2, self.K_filter[2]].copy(),
            ]
            self.u[phantomType] = np.hstack(
                [sublist for sublist, flag in zip(u, self.use_coil) if flag]
            )

    def __getitem__(self, idx):
        if self.datatype == "train":
            return self.Su_observed_norm[idx], self.K_train[idx]
        elif self.datatype == "test":
            return self.Su[idx], self.K_train[idx]

    def __len__(self):
        return len(self.K_train)

    def DataTypeSwitch(self, datatype):
        self.datatype = datatype

    @staticmethod
    def collate_fn(batch):
        S = [torch.from_numpy(item[0]) for item in batch]
        K = [item[1] for item in batch]
        return torch.stack(S), K


if __name__ == "__main__":
    SMdata = SystemMatrixLoad(
        SMPath=r"OpenMPIData\calibrations\5.mdf",
        MeasPath=None,
        SNRThreshold=3,
        sparsity=0.125,
        sampling_method="random",
        K_select=None,
        use_coil=[True, True, False],
    )
    SMdata.DataTypeSwitch("test")
    SMdata[0]
    loader = DataLoader(
        SMdata, batch_size=64, shuffle=True, collate_fn=SMdata.collate_fn
    )
    data = next(iter(loader))
