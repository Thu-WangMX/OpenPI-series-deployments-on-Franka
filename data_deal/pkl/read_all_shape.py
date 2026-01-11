import pickle
import numpy as np
from pathlib import Path
import sys

# ================= 配置区域 =================
# 你的目标文件夹路径
TARGET_DIR = "/work/wmx/openpi/data_1213/merged_all_episodes"
# ===========================================

def print_separator(title=""):
    print("\n" + "=" * 60)
    if title:
        print(f"📢 {title}")
    print("=" * 60)

def inspect_pkl_structure(folder_path):
    path = Path(folder_path)
    
    # 1. 检查路径和文件
    if not path.exists():
        print(f"❌ 错误: 文件夹不存在 -> {folder_path}")
        return

    pkl_files = sorted(list(path.glob("*.pkl")))
    if not pkl_files:
        print(f"❌ 错误: 在该目录下没有找到 .pkl 文件 -> {folder_path}")
        return

    total_files = len(pkl_files)
    print(f"✅ 目录检查通过: 发现 {total_files} 个 .pkl 文件")
    
    # 2. 读取第一个文件作为样本
    sample_file = pkl_files[0]
    print_separator(f"正在分析样本文件: {sample_file.name}")

    try:
        with open(sample_file, 'rb') as f:
            # 假设数据是 list 类型的 episode 数据
            episode_data = pickle.load(f)
        
        print(f"🔹 数据总类型: {type(episode_data)}")
        print(f"🔹 序列总长度 (Frames): {len(episode_data)}")
        
        if len(episode_data) == 0:
            print("⚠️ 警告: 数据列表为空")
            return

        # 3. 获取第一帧数据进行详细维度分析
        first_frame = episode_data[0]
        
        # 提取关键部分
        obs = first_frame.get('observations', {})
        orin_state = obs.get('orin_state', {})
        pixels = obs.get('pixels', {})
        action = first_frame.get('action', None)
        
        print_separator("核心维度检查 (基于第 0 帧)")

        # (A) 机械臂状态 (orin_state)
        print("🔧 [Robot State Dimensions]")
        state_keys = ['q', 'dq', 'tau_J', 'tau_ext', 'tcp_pose', 'tcp_vel', 'tcp_force', 'tcp_torque', 'gripper_pose']
        
        if not orin_state:
            print("   ⚠️ 未找到 'orin_state' 数据！")
        else:
            for key in state_keys:
                if key in orin_state:
                    val = orin_state[key]
                    # 尝试获取 shape，如果是 list 则获取长度
                    shape_info = val.shape if hasattr(val, 'shape') else (f"len={len(val)}" if isinstance(val, list) else type(val))
                    print(f"   - {key:<15}: {shape_info}")
                else:
                    print(f"   - {key:<15}: ❌ Missing")

        # (B) 动作 (Action)
        print("\n🎯 [Action Dimension]")
        if action is not None:
            shape_info = action.shape if hasattr(action, 'shape') else type(action)
            print(f"   - action         : {shape_info}")
        else:
            print("   ⚠️ 未找到 'action' 字段")

        # (C) 视觉数据 (Images)
        print("\n📷 [Image Dimensions]")
        if not pixels:
            print("   ⚠️ 未找到 'pixels' 数据！")
        else:
            for cam_name, img_data in pixels.items():
                shape_info = img_data.shape if hasattr(img_data, 'shape') else type(img_data)
                print(f"   - {cam_name:<15}: {shape_info}")

        # (D) 其他信息
        print("\n📝 [Other Info]")
        if 'task_description' in obs:
            print(f"   - Task           : \"{obs['task_description']}\"")
        if 'language_instruction' in first_frame:
            print(f"   - Instruction    : \"{first_frame['language_instruction']}\"")

    except Exception as e:
        print(f"\n❌ 读取或解析文件时发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_pkl_structure(TARGET_DIR)