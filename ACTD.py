import torch
from tensorly.tenalg.core_tenalg import multi_mode_dot

from utils.DataProcess import cheb_transform_matrix


class ACTD(torch.nn.Module):
    def __init__(self, tensor_shape, rank, device=None):
        super().__init__()
        # 初始化参数
        self.tensor_shape = tensor_shape
        self.rank = rank
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # 低秩张量核
        self.core = torch.nn.Parameter(
            torch.randn(rank, dtype=torch.complex64, device=self.device),
            requires_grad=True,
        )

        # 自适应变换矩阵（对每个模式）
        self.transforms = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.randn((r, r), dtype=torch.complex64, device=self.device),
                    requires_grad=True,
                )
                for r in rank
            ]
        )

        # 预计算 LR 和 HR 的切比雪夫矩阵伪逆
        for i, (dim, r) in enumerate(zip(tensor_shape, rank)):
            # LR
            C = cheb_transform_matrix(dim, r)
            C_tensor = torch.tensor(C, dtype=torch.complex64, device=self.device)
            C_pinv = torch.linalg.pinv(C_tensor)  # C_pinv = C_tensor.T
            self.register_buffer(f"C_pinv_{i}", C_pinv)

    def forward(self):
        factors = []
        for i in range(len(self.transforms)):
            C_pinv = getattr(self, f"C_pinv_{i}")
            transform = self.transforms[i]
            factor = torch.matmul(C_pinv, transform)
            factors.append(factor)
        return multi_mode_dot(self.core, factors)


def ACTD_optimization(
    tensor, mask, ranks, lr=0.05, max_iter=5000, device="cuda:0", reg_F=1, reg_PQR=1
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    tensor = tensor.to(device)
    mask = mask.to(device)

    model = ACTD(tensor.shape, ranks, device=device).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    best_loss = float("inf")
    best_state = None
    loss_history = []

    for _ in range(max_iter):
        optimizer.zero_grad()
        X_hat = model()
        # MSE on LR
        mse = (torch.norm((X_hat - tensor) * mask) ** 2) / mask.sum()
        # 正则项
        reg_term = reg_F * (
            torch.abs(model.core.real).mean() + torch.abs(model.core.imag).mean()
        )
        reg_trans_terms = sum(torch.norm(T) ** 2 for T in model.transforms) / sum(
            r * r for r in ranks
        )
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
        Reconstructed_Cheb = (
            multi_mode_dot(model.core, model.transforms).detach().cpu().numpy()
        )
    return Reconstructed_Cheb
