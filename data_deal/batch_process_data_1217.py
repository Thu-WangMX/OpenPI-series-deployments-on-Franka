import os
import glob
import pickle
import numpy as np
import sys
from tqdm import tqdm

# ================= 配置路径 =================
# 1. 引入转换工具的路径
TOOL_PATH = '/work/wmx/openpi/data_deal'
if TOOL_PATH not in sys.path:
    sys.path.append(TOOL_PATH)

# 2. 导入你之前生成的转换函数
try:
    # 假设文件名为 quat_2_6drotation.py
    from quat_2_6drotation import quaternion_to_rotation6d
except ImportError:
    print(f"❌ 错误: 在 {TOOL_PATH} 下找不到 quat_2_6drotation.py，请检查文件名！")
    sys.exit(1)

# 3. 输入和输出文件夹
INPUT_DIR = '/work/wmx/dataset_1227/dataset_Pick_the_red_chili_pepper_doll_into_the_basket._2025-12-27_14-24-32'
OUTPUT_DIR = '/work/wmx/openpi/dataset_1225/data_red_1227_70'

# 确保输出文件夹存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_single_episode(file_path, save_path):
    """读取、处理并保存单个 pickle 文件"""
    with open(file_path, 'rb') as f:
        data_list = pickle.load(f)

    # 你的数据看起来是一个列表，每一项是一帧的字典
    # 如果是其他结构（如大字典包含数组），代码需要微调。
    # 这里假设 data_list 是 [frame0_dict, frame1_dict, ...]
    
    processed_data = []

    for frame in data_list:
        # ========================================================
        # 1. 提取原始数据
        # ========================================================
        obs = frame['observations']
        next_obs = frame['next_observations']
        
        # 获取 tcp_pose (7维: [x,y,z, w,x,y,z])
        curr_tcp = obs['orin_state']['tcp_pose']
        next_tcp = next_obs['orin_state']['tcp_pose']
        
        # 获取 gripper (1维)
        # 注意：有时候是标量，需要转成数组以便拼接
        curr_gripper = np.array([obs['orin_state']['gripper_pose']]).flatten()
        next_gripper = np.array([next_obs['orin_state']['gripper_pose']]).flatten()

        # ========================================================
        # 步骤 1: 四元数 -> 6D Rotation
        # ========================================================
        # 分离 Pos (前3) 和 Quat (后4, xyzw)
        curr_pos = curr_tcp[:3]
        curr_quat = curr_tcp[3:] 
        
        next_pos = next_tcp[:3]
        next_quat = next_tcp[3:]

        # 调用工具转换
        curr_rot6d = quaternion_to_rotation6d(curr_quat) # shape (6,)
        next_rot6d = quaternion_to_rotation6d(next_quat) # shape (6,)

        # ========================================================
        # 步骤 2: 拼接 State 和 Action
        # ========================================================
        # State: Pos(3) + Rot6D(6) + Gripper(1) = 10维
        new_state = np.concatenate([curr_pos, curr_rot6d, curr_gripper])
        
        # Action: Next Pos(3) + Next Rot6D(6) + Next Gripper(1) = 10维
        new_action = np.concatenate([next_pos, next_rot6d, next_gripper])

        # 写入字典
        # 写为 observation/state
        obs['state'] = new_state.astype(np.float32)
        
        # 写为 action (覆盖最外层的 action)
        frame['action'] = new_action.astype(np.float32)

        # ========================================================
        # 步骤 3: 拼接 Wrench (Force + Torque)
        # ========================================================
        force = obs['orin_state']['tcp_force']
        torque = obs['orin_state']['tcp_torque']
        # 拼接为 6维
        wrench = np.concatenate([force, torque])
        
        # 写为 observation/tcp_wrench
        obs['tcp_wrench'] = wrench.astype(np.float32)

        # ========================================================
        # 步骤 4: 重命名 tau_J 为 effort
        # ========================================================
        tau_j = obs['orin_state']['tau_J']
        
        # 写为 observation/effort
        obs['effort'] = tau_j.astype(np.float32)

        # 将修改后的 frame 加入列表
        processed_data.append(frame)

    # 保存处理后的文件
    with open(save_path, 'wb') as f:
        pickle.dump(processed_data, f)

def main():
    # 获取所有 pkl 文件
    pkl_files = glob.glob(os.path.join(INPUT_DIR, '*.pkl'))
    print(f"📂 发现 {len(pkl_files)} 个文件，准备处理...")
    print(f"🚀 输出目录: {OUTPUT_DIR}")

    # 使用 tqdm 显示进度条
    for pkl_file in tqdm(pkl_files):
        file_name = os.path.basename(pkl_file)
        save_path = os.path.join(OUTPUT_DIR, file_name)
        
        try:
            process_single_episode(pkl_file, save_path)
        except Exception as e:
            print(f"\n❌ 处理文件 {file_name} 时出错: {e}")

    print("\n✅ 所有文件处理完成！")

if __name__ == "__main__":
    main()