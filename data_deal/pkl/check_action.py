#读取某个pkl的若干帧action
import pickle
import numpy as np
import os

# ================= 配置区域 =================
# 1. 指定要读取的文件路径
file_path = '/work/wmx/openpi/data_clean/single_grasp/episode_0.pkl'

# 2. 指定要读取的帧数范围 (例如：前 5 帧)
start_frame = 0
num_frames = 250
# ===========================================

def check_actions():
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return

    print(f"正在读取: {file_path} ...")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    total_len = len(data)
    print(f"✅ 读取成功，该 Episode 总长度: {total_len} 帧")

    # 防止索引越界
    end_frame = min(start_frame + num_frames, total_len)
    
    print(f"\n--- 展示第 {start_frame} 到 {end_frame - 1} 帧的 Action 数据 ---")

    for i in range(start_frame, end_frame):
        frame_data = data[i]
        
        # 获取 action
        if 'action' in frame_data:
            action = frame_data['action']
            print(f"[Frame {i}] Action Shape: {action.shape}")
            print(f"   Values: {action}")
        else:
            print(f"[Frame {i}] ⚠️ 警告: 该帧没有 'action' 键")

if __name__ == '__main__':
    check_actions()