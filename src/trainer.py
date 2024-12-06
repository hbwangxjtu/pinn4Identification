import torch
from torch.utils.data import DataLoader
from src.dataset import RobotDataset
from src.pinn import PINN

def train_pinn(model, train_loader, optimizer, num_epochs, device):
    """训练PINN网络"""
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for q, dq, ddq, tau in train_loader:
            q, dq, ddq, tau = q.to(device), dq.to(device), ddq.to(device), tau.to(device)
            
            pred = model(q, dq)
            pred_tau = pred[:, 6:]
            
            physics_loss = model.physics_loss(q, dq, ddq)
            data_loss = torch.mean((pred_tau - tau)**2)
            loss = physics_loss + data_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')

def identify_parameters(robot_data, hidden_dim=128, num_epochs=1000, batch_size=32):
    """参数辨识主函数"""
    dataset = RobotDataset(
        robot_data['q'],
        robot_data['dq'],
        robot_data['ddq'],
        robot_data['tau']
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PINN(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    train_pinn(model, train_loader, optimizer, num_epochs, device)
    
    return model.dyn_params.detach().cpu().numpy()
