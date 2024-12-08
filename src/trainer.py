import torch
from torch.utils.data import DataLoader
from src.dataset import RobotDataset
from src.pinn import PINN
from datetime import datetime
import os
import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 显式设置后端为TkAgg
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torch.utils.tensorboard import SummaryWriter
import time

class TrainingMonitor:
    def __init__(self, log_dir=None, num_epochs=1000, optimizer=None):
        """
        初始化训练监视器
        Args:
            log_dir: 日志保存目录
            num_epochs: 总训练轮数
        """
        
        # 创建日志目录
        if log_dir is None:
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_dir = os.path.join('runs', current_time)
        os.makedirs(log_dir, exist_ok=True)
        print(f"TensorBoard log directory: {log_dir}")
        print(f"To start TensorBoard, run: tensorboard --logdir={log_dir} --port=6006")
        
        self.writer = SummaryWriter(log_dir)
        # 记录一个初始值以确保writer正常工作
        self.writer.add_scalar('Initialization/test', 0.0, 0)
        self.writer.flush()
        
        self.log_dir = log_dir
        self.num_epochs = num_epochs
        self.optimizer = optimizer  # 保存为实例变量
        
        # 初始化历史记录
        self.loss_history = {
            'total_loss': [], 'physics_loss': [], 
            'data_loss': [], 'ddq_loss': [], 'dynamics_loss': []
        }
        self.param_history = []
        self.start_time = time.time()
    
    def log_epoch(self, epoch, avg_losses, params):
        """记录每个epoch的结果"""
        try:
            elapsed_time = time.time() - self.start_time
            
            # 记录损失到TensorBoard
            for key, value in avg_losses.items():
                value_float = float(value)
                self.loss_history[key].append(value_float)
                # 添加到TensorBoard
                self.writer.add_scalar(f'Losses/{key}', value_float, epoch)
            
            # 记录参数到TensorBoard
            for i, param in enumerate(params):
                param_float = float(param)  # 确保转换为Python float
                self.writer.add_scalar(f'Parameters/param_{i+1}', param_float, epoch)
            self.param_history.append(params)
            
            # 计算和记录学习率
            if self.optimizer:
                    for i, param_group in enumerate(self.optimizer.param_groups):
                        self.writer.add_scalar('Learning_rate/group_{i}', 
                                            float(param_group['lr']), epoch)
           
            # 每10个epoch保存一次历史数据
            if (epoch + 1) % 10 == 0:
                self.save_history()
                
            # 确保数据被写入
            self.writer.flush()
            
            # 计算训练时间
            elapsed_time = time.time() - self.start_time
            
            # 添加参数的直方图
            self.writer.add_histogram('Parameters/distribution', 
                                    torch.tensor(params), epoch)
            
            # 打印训练进度
            print(f"\rEpoch {epoch+1}/{self.num_epochs} | "
                f"Time: {elapsed_time:.2f}s | "
                f"Total Loss: {avg_losses['total_loss']:.4e} | "
                f"Physics Loss: {avg_losses['physics_loss']:.4e} | "
                f"Data Loss: {avg_losses['data_loss']:.4e}", 
                flush=True)
            
            
        except Exception as e:
            print(f"Error in log_epoch: {str(e)}")
            import traceback
            traceback.print_exc()

    def save_history(self):
        """保存训练历史数据"""
        history_data = {
            'losses': {k: [float(v) for v in vals] for k, vals in self.loss_history.items()},
            'parameters': [p.tolist() for p in self.param_history]
        }
        with open(os.path.join(self.log_dir, 'training_history.json'), 'w') as f:
            json.dump(history_data, f)
    
    def close(self):
        """关闭监视器"""
        try:
            # 最后一次确保数据写入
            self.writer.flush()
            self.writer.close()
            print("TensorBoard writer closed successfully")
        except Exception as e:
            print(f"Error closing writer: {str(e)}")

def train_pinn(model, train_loader, optimizer, num_epochs, device, monitor=None):
    """训练PINN网络"""
    model.train()
    best_loss = float('inf')
    
    try:
        for epoch in range(num_epochs):
            epoch_losses = {
                'total_loss': 0, 'physics_loss': 0, 'data_loss': 0,
                'ddq_loss': 0, 'dynamics_loss': 0
            }
            
            batch_count = 0
            for batch_idx, (q, dq, ddq, tau) in enumerate(train_loader):
                q = q.to(device).requires_grad_(True)
                dq = dq.to(device)
                ddq = ddq.to(device)
                tau = tau.to(device)
                
                # 前向传播
                pred = model(q, dq)
                pred_tau = pred[:, 7:]
                
                # 计算损失
                physics_results = model.physics_loss(q, dq, ddq)
                physics_loss = physics_results['total_physics_loss']
                data_loss = torch.mean((pred_tau - tau)**2)
                total_loss = physics_loss + data_loss
                
                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # 记录损失
                batch_losses = {
                    'total_loss': total_loss.item(),
                    'physics_loss': physics_loss.item(),
                    'data_loss': data_loss.item(),
                    'ddq_loss': physics_results['ddq_loss'].item(),
                    'dynamics_loss': physics_results['dynamics_loss'].item()
                }
                
                for key in epoch_losses:
                    epoch_losses[key] += batch_losses[key]
                batch_count += 1
            
            # 计算平均损失
            if batch_count > 0:
                for key in epoch_losses:
                    epoch_losses[key] /= batch_count
            
            # 记录epoch结果
            if monitor is not None:
                monitor.log_epoch(epoch, epoch_losses, 
                                model.dyn_params.detach().cpu().numpy())
            
            # 保存最佳模型
            if epoch_losses['total_loss'] < best_loss:
                best_loss = epoch_losses['total_loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, os.path.join(monitor.log_dir, 'best_model.pth'))
            
            # 每50个epoch保存一个检查点
            if (epoch + 1) % 50 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'losses': epoch_losses,
                }, os.path.join(monitor.log_dir, f'checkpoint_epoch_{epoch+1}.pth'))
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if monitor is not None:
            monitor.close()

def identify_parameters(dataset, hidden_dim=128, num_epochs=1000, batch_size=32):
    """参数辨识主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 初始化模型和优化器
    model = PINN(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 创建训练监控器
    monitor = TrainingMonitor(optimizer=optimizer, num_epochs=num_epochs)
    
    print("\nStarting training...")
    print("To monitor training progress:")
    print("1. Open a new terminal")
    print("2. Navigate to your project directory")
    print("3. Run: tensorboard --logdir=runs --port=6006")
    print("4. Open http://localhost:6006 in your web browser")
    
    # 训练模型
    train_pinn(model, train_loader, optimizer, num_epochs, device, monitor)
    
    return model.dyn_params.detach().cpu().numpy()