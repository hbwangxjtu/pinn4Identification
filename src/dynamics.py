import torch
import numpy as np

def transform_matrix(alpha, a, d, theta):
    """Calculate the homogeneous transformation matrix based on MDH parameters"""
    ct = torch.cos(theta)
    st = torch.sin(theta)
    ca = torch.cos(alpha)
    sa = torch.sin(alpha)
    
    T = torch.zeros(4, 4, dtype=theta.dtype, device=theta.device)
    T[0, 0] = ct
    T[0, 1] = -st
    T[0, 2] = 0
    T[0, 3] = a
    
    T[1, 0] = st * ca
    T[1, 1] = ct * ca
    T[1, 2] = -sa
    T[1, 3] = -sa * d
    
    T[2, 0] = st * sa
    T[2, 1] = ct * sa
    T[2, 2] = ca
    T[2, 3] = ca * d
    
    T[3, 3] = 1.0
    return T

def get_jacobian(q):
    """Calculate the geometric Jacobian matrix"""
    batch_size = q.shape[0]
    J = torch.zeros(batch_size, 6, 7, dtype=q.dtype, device=q.device)
    
    # MDH parameters
    mdh_params = torch.tensor([
        [0, 0, 0.404, 0],
        [-np.pi/2, 0, 0, np.pi],
        [-np.pi/2, 0, 0.4375, np.pi],
        [-np.pi/2, 0, 0, np.pi],
        [-np.pi/2, 0, 0.4125, np.pi],
        [-np.pi/2, 0, 0, np.pi],
        [-np.pi/2, 0, 0.2755, np.pi]
    ], dtype=q.dtype, device=q.device)
    
    for b in range(batch_size):
        T_0_i = torch.eye(4, dtype=q.dtype, device=q.device)
        z_prev = torch.tensor([0, 0, 1], dtype=q.dtype, device=q.device)
        p_prev = torch.zeros(3, dtype=q.dtype, device=q.device)
        
        for i in range(7):
            alpha, a, d, offset = mdh_params[i]
            theta = q[b, i] + offset
            
            T_i = transform_matrix(alpha, a, d, theta)
            T_0_i = torch.mm(T_0_i, T_i)
            
            z_i = T_0_i[:3, 2]
            p_i = T_0_i[:3, 3]
            
            J[b, :3, i] = torch.cross(z_prev, p_i - p_prev, dim=0)
            J[b, 3:, i] = z_prev
            
            z_prev = z_i
            p_prev = p_i
            
            
    return J

def get_M(q, params):
    """Calculate the inertia matrix"""
    batch_size = q.shape[0]
    n_joints = 7
    M = torch.zeros(batch_size, n_joints, n_joints, dtype=q.dtype, device=q.device)
    
    for b in range(batch_size):
        J = get_jacobian(q[b:b+1])  # J shape: [1, 6, 7]
        
        for i in range(n_joints):
            offset = i * 13
            mass = params[offset]
            com = params[offset+1:offset+4]
            inertia = params[offset+4:offset+13].reshape(3, 3)
            
            H_i = torch.zeros(6, 6, dtype=q.dtype, device=q.device)
            H_i[:3, :3] = inertia
            
            com_skew = torch.zeros(3, 3, dtype=q.dtype, device=q.device)
            com_skew[0, 1] = -com[2]
            com_skew[0, 2] = com[1]
            com_skew[1, 0] = com[2]
            com_skew[1, 2] = -com[0]
            com_skew[2, 0] = -com[1]
            com_skew[2, 1] = com[0]
            
            H_i[:3, 3:] = com_skew
            H_i[3:, :3] = -com_skew
            H_i[3:, 3:] = mass * torch.eye(3, dtype=q.dtype, device=q.device)
            
            # 修改索引方式
            J_i = J[0, :, i].reshape(-1, 1)  # 取第一个batch，所有行，第i列
            H_i_J_i = torch.mm(H_i, J_i)
            contribution = torch.mm(J_i.T, H_i_J_i)
            M[b, i, i] = contribution.item()
    
    return M

def get_C(q, dq, params):
    """Calculate the Coriolis matrix"""
    batch_size = q.shape[0]
    n_joints = 7
    C = torch.zeros(batch_size, n_joints, n_joints, dtype=q.dtype, device=q.device)
    
    if not q.requires_grad:
        q = q.detach().requires_grad_(True)
    
    for b in range(batch_size):
        M = get_M(q[b:b+1], params)[0]
        if M.requires_grad:  # 确保M是可微的、
            dM = torch.autograd.grad(M.sum(), q, create_graph=True)[0]
            dM = dM[b]  # 正确的批次索引位置
            
            for i in range(n_joints):
                for j in range(n_joints):
                    for k in range(n_joints):
                        c_ijk = 0.5 * (dM[i, j, k] + dM[i, k, j] - dM[k, j, i])
                        C[b, i, j] += c_ijk * dq[b, k]
    
    return C

def get_G(q, params):
    """Calculate the gravity vector"""
    batch_size = q.shape[0]
    n_joints = 7
    g = torch.tensor([0, 0, -9.81], dtype=q.dtype, device=q.device)
    G = torch.zeros(batch_size, n_joints, dtype=q.dtype, device=q.device)
    
    for b in range(batch_size):
        J = get_jacobian(q[b:b+1])  # J shape: [1, 6, 7]
        
        for i in range(n_joints):
            offset = i * 13
            mass = params[offset]
            com = params[offset+1:offset+4]
            
            # 修改这里的索引方式
            J_i = J[0, :3, i]  # 只取第一个 batch，前3行（平移部分），第 i 列
            G[b, i] = -mass * torch.dot(g, J_i)
    
    return G