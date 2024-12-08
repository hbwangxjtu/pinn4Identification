import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

class RobotDataset(Dataset):
    def __init__(self, q, dq, ddq, tau):
        """
        构造数据集
        Args:
            q: 关节位置数据 [N, 7]
            dq: 关节速度数据 [N, 7] 
            ddq: 关节加速度数据 [N, 7]
            tau: 关节力矩数据 [N, 7]
        """
        self.q = torch.FloatTensor(q)
        self.dq = torch.FloatTensor(dq)
        self.ddq = torch.FloatTensor(ddq)
        self.tau = torch.FloatTensor(tau)
        
    def __len__(self):
        return len(self.q)
    
    def __getitem__(self, idx):
        q = self.q[idx].clone().detach().requires_grad_(True)
        dq = self.dq[idx].clone().detach()
        ddq = self.ddq[idx].clone().detach()
        tau = self.tau[idx].clone().detach()
        return q, dq, ddq, tau

def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    巴特沃斯低通滤波器
    Args:
        data: 输入数据
        cutoff: 截止频率
        fs: 采样频率
        order: 滤波器阶数
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def compute_derivatives(q, dt, cutoff_freq=10, fs=1000):
    """
    计算关节速度和加速度
    Args:
        q: 位置数据 [N, 7]
        dt: 时间步长
        cutoff_freq: 截止频率
        fs: 采样频率
    Returns:
        dq: 速度数据
        ddq: 加速度数据
    """
    # 对位置数据进行滤波
    q_filtered = np.zeros_like(q)
    for i in range(q.shape[1]):
        q_filtered[:, i] = butter_lowpass_filter(q[:, i], cutoff_freq, fs)
    
    # 计算速度（使用中心差分）
    dq = np.zeros_like(q)
    dq[1:-1] = (q_filtered[2:] - q_filtered[:-2]) / (2 * dt)
    dq[0] = (q_filtered[1] - q_filtered[0]) / dt
    dq[-1] = (q_filtered[-1] - q_filtered[-2]) / dt
    
    # 对速度进行滤波
    dq_filtered = np.zeros_like(dq)
    for i in range(dq.shape[1]):
        dq_filtered[:, i] = butter_lowpass_filter(dq[:, i], cutoff_freq, fs)
    
    # 计算加速度
    ddq = np.zeros_like(q)
    ddq[1:-1] = (dq_filtered[2:] - dq_filtered[:-2]) / (2 * dt)
    ddq[0] = (dq_filtered[1] - dq_filtered[0]) / dt
    ddq[-1] = (dq_filtered[-1] - dq_filtered[-2]) / dt
    
    # 对加速度进行滤波
    ddq_filtered = np.zeros_like(ddq)
    for i in range(ddq.shape[1]):
        ddq_filtered[:, i] = butter_lowpass_filter(ddq[:, i], 2, fs)
    
    return dq_filtered, ddq_filtered

def plot_results(t, q, dq, ddq):
    """
    绘制结果图像
    """
    plt.figure(figsize=(15, 10))
    
    # 颜色列表，用于区分不同关节
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
    
    plt.subplot(3, 1, 1)
    for i in range(7):  # 7个关节
        plt.plot(t, q[:, i], color=colors[i], label=f'Joint {i+1}')
    plt.title('Joint Positions')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [rad]')
    plt.legend()
    plt.grid(True)
    
    # 绘制速度数据
    plt.subplot(3, 1, 2)
    for i in range(7):
        plt.plot(t, dq[:, i], color=colors[i], label=f'Joint {i+1}')
    plt.title('Joint Velocities')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [rad/s]')
    plt.legend()
    plt.grid(True)
    
    # 绘制加速度数据
    plt.subplot(3, 1, 3)
    for i in range(7):
        plt.plot(t, ddq[:, i], color=colors[i], label=f'Joint {i+1}')
    plt.title('Joint Accelerations')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [rad/s^2]')
    plt.legend()
    plt.grid(True)
    
    # 调整子图之间的间距
    plt.tight_layout()
    plt.show()
    
def process_robot_data(q_file, tau_file, dt=0.001):
    """
    处理机器人数据
    Args:
        q_file: 位置数据文件路径
        tau_file: 力矩数据文件路径
        dt: 采样时间间隔
    Returns:
        dataset: RobotDataset对象
    """
    # 加载数据
    q = np.loadtxt(q_file)
    tau = np.loadtxt(tau_file)
    
    # 生成时间序列
    t = np.arange(len(q)) * dt
    
    # 计算速度和加速度
    dq, ddq = compute_derivatives(q, dt)
    
    # 绘制结果
    # plot_results(t, q, dq, ddq)
    
    # 创建数据集
    dataset = RobotDataset(q, dq, ddq, tau)
    
    return dataset

if __name__ == "__main__":
    # 文件路径
    q_file = "./data/g2/q_mea.txt"      
    tau_file = "./data/g2/tau_mea.txt"  
    
    # 处理数据
    dataset= process_robot_data(q_file, tau_file)
    
    print(f"数据集大小: {len(dataset)}")
    # print(f"位置数据形状: {q.shape}")
    # print(f"速度数据形状: {dq.shape}")
    # print(f"加速度数据形状: {ddq.shape}")
    # print(f"力矩数据形状: {tau.shape}")
    
    # 获取第一个样本
    q_sample, dq_sample, ddq_sample, tau_sample = dataset[0]
    print("\n第一个样本的数据:")
    print(f"位置: {q_sample}")
    print(f"速度: {dq_sample}")
    print(f"加速度: {ddq_sample}")
    print(f"力矩: {tau_sample}")