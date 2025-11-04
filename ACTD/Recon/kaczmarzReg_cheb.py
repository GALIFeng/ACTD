import numpy as np
from ACTD import tensor_cheb_transform

def computeSliceEnergy(S):
    Nf = S.shape[0]
    energy = np.zeros(Nf, dtype=np.float32)
    for f in range(Nf):
        energy[f] = np.linalg.norm(S[f].ravel())
    return energy

def TV3D(vol, iter=1, dt=0.1, epsilon=1e-3, lamb=0.0):
    I_t = vol
    ep2 = epsilon ** 2

    for t in range(iter):
        # 在每个方向上各填充一层边界
        I_pad = np.pad(I_t, ((1,1),(1,1),(1,1)), mode='edge')

        # 一阶导数（中心差分）
        Ix = (I_pad[1:-1,1:-1,2:] - I_pad[1:-1,1:-1,:-2]) / 2.0
        Iy = (I_pad[1:-1,2:,1:-1] - I_pad[1:-1,:-2,1:-1]) / 2.0
        Iz = (I_pad[2:,1:-1,1:-1] - I_pad[:-2,1:-1,1:-1]) / 2.0

        # 二阶导数
        Ixx = I_pad[1:-1,1:-1,2:] + I_pad[1:-1,1:-1,:-2] - 2 * I_t
        Iyy = I_pad[1:-1,2:,1:-1] + I_pad[1:-1,:-2,1:-1] - 2 * I_t
        Izz = I_pad[2:,1:-1,1:-1] + I_pad[:-2,1:-1,1:-1] - 2 * I_t

        # 交叉导数
        Ixy = ( I_pad[1:-1,2:,2:] + I_pad[1:-1,:-2,:-2]
              - I_pad[1:-1,2:,:-2] - I_pad[1:-1,:-2,2:] ) / 4.0
        Ixz = ( I_pad[2:,1:-1,2:] + I_pad[:-2,1:-1,:-2]
              - I_pad[2:,1:-1,:-2] - I_pad[:-2,1:-1,2:] ) / 4.0
        Iyz = ( I_pad[2:,2:,1:-1] + I_pad[:-2,:-2,1:-1]
              - I_pad[2:,:-2,1:-1] - I_pad[:-2,2:,1:-1] ) / 4.0

        # 分子：包括各向的二阶导数与一阶导数的组合
        num = (
            Iyy*(Ix**2 + ep2) + Ixx*(Iy**2 + ep2) + Izz*(Iz**2 + ep2)
            - 2*Ix*Iy*Ixy - 2*Ix*Iz*Ixz - 2*Iy*Iz*Iyz
        )

        # 分母
        den = (Ix**2 + Iy**2 + Iz**2 + ep2)**1.5

        # 更新
        I_t += dt * ( num/den + (0.5 + lamb)*(vol - I_t) )

    return I_t



def kaczmarz_tensor_cheb(S, U,
                        Core_shape = None,
                        S0_shape = None,
                        iterations = 10,
                        lambd = 0.0,
                        enforceReal = False,
                        enforcePositive = False,
                        shuffle = True,
                        tv_iters=1,
                        dt=0.1,
                        eps=1,
                        scale = 1,
                        ):
                            
    Nf = S.shape[0]
    idx = np.arange(Nf)
    dtype = S.dtype if np.iscomplexobj(S) else np.complex128
    C = np.zeros(tuple(map(int, np.array(S0_shape[-3:]) * scale)), dtype=np.float64) 
    rho = np.zeros(Nf, dtype=dtype)
    energy = computeSliceEnergy(S)
    for _ in range(iterations):

        Z_cheb = tensor_cheb_transform(C, Core_shape, S0_shape, scale=scale)
        Z_cheb = Z_cheb.astype(dtype) # 确保为复数类型

        if shuffle:
            np.random.shuffle(idx)
        # 1. 更新切比雪夫系数 Z_cheb
        for f in idx:
            if energy[f] == 0: continue
            inner_prod = np.tensordot(S[f], Z_cheb, axes=3)
            beta = (U[f] - inner_prod - np.sqrt(lambd) * rho[f]) / (energy[f]**2 + lambd)
            Z_cheb += beta * S[f].conjugate()
            rho[f] += np.sqrt(lambd) * beta

        C = tensor_cheb_transform(Z_cheb, Core_shape, S0_shape, scale=scale)
        # 2. 添加约束条件
        if enforceReal:
            C = C.real
        if enforcePositive:
            C = np.maximum(C, 0)
        # 3. 应用 TV 正则化并更新 C 以用于下一次迭代
        C = TV3D(C, tv_iters, dt, eps)
    return C




def kaczmarz_tensor(S, U,
                        iterations = 3,
                        lambd = 0.0,
                        enforceReal = False,
                        enforcePositive = False,
                        shuffle = True,
                        tv_iters=1,
                        dt=0.1,
                        eps=1):
    """
    Tensor-form Kaczmarz with Tikhonov and TV regularization.

    Args:
      S: ndarray, shape (Nf, Nx, Ny, Nz)
      U: ndarray, shape (Nf,)
      C0: initial tensor (Nx,Ny,Nz)
      lambd: Tikhonov weight
      tv_weight: TV regularization weight
      tv_iters: TV iterations per slice
    """
    # 预先计算每个切片的能量
    energy = computeSliceEnergy(S)
    Nf, Nx, Ny, Nz = S.shape
    idx = np.arange(Nf)
    # initialize C and residual
    dtype = S.dtype if np.iscomplexobj(S) else np.complex128
    C = np.zeros((Nx, Ny, Nz), dtype=dtype)
    rho = np.zeros(Nf, dtype=dtype)

    for it in range(iterations):
        if shuffle:
            np.random.shuffle(idx)
        # Kaczmarz updates
        for f in idx:
            C = C.astype(dtype)  # Ensure C is in the correct dtype
            E = energy[f]
            if E == 0: continue
            inner = np.tensordot(S[f], C, axes=3)
            rf = U[f] - inner
            beta = (rf - np.sqrt(lambd)*rho[f])/(E*E + lambd)
            C += beta * S[f].conjugate()
            rho[f] += np.sqrt(lambd)*beta

        # enforce constraints
        if enforceReal:
            C = C.real.astype(dtype)
        if enforcePositive:
            C = np.maximum(C.real, 0).astype(dtype)

        # TV regularization: denoise each 2D slice
        C_real = C.real
        # 直接对整个三维体调用 TV3D
        C = TV3D(C_real, tv_iters, dt, eps)
        
    return C

