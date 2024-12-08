import numpy as np
from src.trainer import identify_parameters
from src.dataset import process_robot_data

def main():

    # 文件路径
    q_file = "./data/g2/q_mea.txt"      # 使用 ./ 而不是 ../
    tau_file = "./data/g2/tau_mea.txt"  # 使用 ./ 而不是 ../
    
    # 处理数据
    dataset = process_robot_data(q_file, tau_file)   
    
    # 执行参数辨识
    try:
        identified_params = identify_parameters(
            dataset,
            hidden_dim=128,
            num_epochs=1000,
            batch_size=32
        )
    except KeyboardInterrupt:
        print("\nTraining stopped by user")
        
    print("Identified parameters:", identified_params)

if __name__ == "__main__":
    main()