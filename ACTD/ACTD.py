import torch
import tensorly as tl
from tensorly.tenalg.core_tenalg import multi_mode_dot
import numpy as np
from numpy.polynomial import chebyshev as cheb

tl.set_backend('pytorch')

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
    return C    # C^plus = C.T，即C的伪逆等于C的转置

def tensor_cheb_transform(C, Core_shape, S0_shape, device='cpu', scale=1):
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
    is_batch = (C.ndim == 4)
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
            raise ValueError(f"输入 C 的形状 {C.shape} 与 rank {ranks} 或 S_shape {S_shape} 都不匹配")
    else:
        # 取第一个切片确定 inv
        sample = C[0]
        inv = (sample.shape == ranks)

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


class ACTD(torch.nn.Module):
    def __init__(self, tensor_shape, rank, device=None):
        super(ACTD, self).__init__()
        # 初始化参数
        self.tensor_shape = tensor_shape
        self.rank = rank
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        # 低秩张量核
        self.core = torch.nn.Parameter(
            torch.randn(rank, dtype=torch.complex64, device=self.device),
            requires_grad=True
        )

        # 自适应变换矩阵（对每个模式）
        self.transforms = torch.nn.ParameterList([
            torch.nn.Parameter(
                torch.randn((r, r), dtype=torch.complex64, device=self.device),
                requires_grad=True
            ) for r in rank
        ])

        # 预计算 LR 和 HR 的切比雪夫矩阵伪逆
        for i, (dim, r) in enumerate(zip(tensor_shape, rank)):
            # LR
            C = cheb_transform_matrix(dim, r)
            C_tensor = torch.tensor(C, dtype=torch.complex64, device=self.device)
            C_pinv = torch.linalg.pinv(C_tensor)        # C_pinv = C_tensor.T
            self.register_buffer(f'C_pinv_{i}', C_pinv)
            

    def forward(self):
        factors = []
        for i in range(len(self.transforms)):
            C_pinv = getattr(self, f'C_pinv_{i}')
            transform = self.transforms[i]
            factor = torch.matmul(C_pinv, transform)
            factors.append(factor)
        return multi_mode_dot(self.core, factors)

def ACTD_optimization(tensor, mask, ranks, lr=0.05, max_iter=5000,
                        device="cuda:0", reg_F=1, reg_PQR=1):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    tensor = tensor.to(device)
    mask = mask.to(device)

    model = ACTD(tensor.shape, ranks, device=device).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    best_loss = float('inf')
    best_state = None
    loss_history = []

    for _ in range(max_iter):
        optimizer.zero_grad()
        X_hat = model()
        # MSE on LR
        mse = (torch.norm((X_hat - tensor) * mask) ** 2) / mask.sum()
        # 正则项
        reg_term = reg_F * (torch.abs(model.core.real).mean() + torch.abs(model.core.imag).mean())
        reg_trans_terms = sum(torch.norm(T) ** 2 for T in model.transforms) / sum(r * r for r in ranks)
        loss = mse + reg_term + reg_PQR * reg_trans_terms
        loss_history.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone().detach() for k, v in model.state_dict().items()}

        loss.backward()
        optimizer.step()

    # 加载最优参数
    if best_state is not None:
        model.load_state_dict(best_state)

    # 输出 LR 重建结果
    with torch.no_grad():
        Reconstructed_Cheb = multi_mode_dot(model.core, model.transforms).detach().cpu().numpy()
    return Reconstructed_Cheb





