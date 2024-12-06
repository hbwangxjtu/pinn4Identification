import numpy as np
from src.trainer import identify_parameters

def main():
    # 示例数据
    robot_data = {
        'q': np.random.randn(1000, 6),
        'dq': np.random.randn(1000, 6),
        'ddq': np.random.randn(1000, 6),
        'tau': np.random.randn(1000, 6)
    }
    
    # 执行参数辨识
    identified_params = identify_parameters(robot_data)
    print("Identified parameters:", identified_params)

if __name__ == "__main__":
    main()