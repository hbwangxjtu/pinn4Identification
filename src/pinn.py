import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 12)
        )
        self.dyn_params = nn.Parameter(torch.randn(10))
        
    def forward(self, q, dq):
        x = torch.cat([q, dq], dim=-1)
        return self.net(x)
    
    def physics_loss(self, q, dq, ddq):
        from src.dynamics import get_M, get_C, get_G
        
        M = get_M(q, self.dyn_params)
        C = get_C(q, dq, self.dyn_params)
        G = get_G(q, self.dyn_params)
        
        tau_physics = torch.bmm(M, ddq.unsqueeze(-1)).squeeze(-1) + \
                     torch.bmm(C, dq.unsqueeze(-1)).squeeze(-1) + G
                     
        pred = self.forward(q, dq)
        pred_ddq = pred[:, :6]
        pred_tau = pred[:, 6:]
        
        physics_loss = torch.mean((pred_ddq - ddq)**2)
        dynamics_loss = torch.mean((pred_tau - tau_physics)**2)
        
        return physics_loss + dynamics_loss