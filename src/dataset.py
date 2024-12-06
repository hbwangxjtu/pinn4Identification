import torch
from torch.utils.data import Dataset

class RobotDataset(Dataset):
    def __init__(self, q, dq, ddq, tau):
        """
        构造数据集
        Args:
            q: 关节位置数据 [N, 6]
            dq: 关节速度数据 [N, 6] 
            ddq: 关节加速度数据 [N, 6]
            tau: 关节力矩数据 [N, 6]
        """
        self.q = torch.FloatTensor(q)
        self.dq = torch.FloatTensor(dq)
        self.ddq = torch.FloatTensor(ddq)
        self.tau = torch.FloatTensor(tau)
        
    def __len__(self):
        return len(self.q)
    
    def __getitem__(self, idx):
        return self.q[idx], self.dq[idx], self.ddq[idx], self.tau[idx]
