from glob import glob
import os
import pickle

import numpy as np
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================
# 数据路径
DATA_DIR = "/work/wmx/openpi/data_1213/merged_all_episodes"

# 是否备份原文件 (建议 True，防止写错)
BACKUP_ORIGINAL = True

# 是否包含夹爪? (强烈建议 True，否则抓取任务没法做)
# True:  Action = [x, y, z, qx, qy, qz, qw, gripper] (8维)
# False: Action = [x, y, z, qx, qy, qz, qw]          (7维)
INCLUDE_GRIPPER = True
# ===============================================


def get_ee_state(frame_data):
    """从一帧数据中提取 EE 状态 (Pose + Gripper)"""
    try:
        orin_state = frame_data["observations"]["orin_state"]

        # 1. 提取 TCP Pose (7维: xyz + quat)
        tcp_pose = orin_state["tcp_pose"]

        if not INCLUDE_GRIPPER:
            return np.array(tcp_pose, dtype=np.float32)

        # 2. 提取 Gripper (1维)
        gripper = orin_state.get("gripper_pose", 0.0)
        # 确保 gripper 是数组形式以便拼接
        if np.isscalar(gripper):
            gripper = np.array([gripper], dtype=np.float32)
        else:
            gripper = np.array(gripper, dtype=np.float32).reshape(1)

        # 3. 拼接: [Pose(7), Gripper(1)] -> 8维
        return np.concatenate([tcp_pose, gripper]).astype(np.float32)

    except KeyError as e:
        raise ValueError(f"数据结构缺失: {e}")


def process_episode(file_path):
    """
    逻辑:
    1. Action[t] = State[t+1] (其中 State = TCP_Pose + Gripper)
    2. 最后一帧 Action = 倒数第二帧 Action
    """
    try:
        # 读取
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        total_steps = len(data)
        if total_steps < 2:
            return False

        # --- 核心重写逻辑 ---

        # 1. 遍历前 N-1 帧 (0 到 N-2)
        # 用 t+1 帧的状态作为 t 帧的动作
        for i in range(total_steps - 1):
            next_frame_state = get_ee_state(data[i + 1])  # 获取 t+1 的状态
            data[i]["action"] = next_frame_state

        # 2. 处理最后一帧 (索引 N-1)
        # 因为没有 t+1 了，所以复制前一帧的 action (即刚刚算出来的 State_last)
        # 保持动作静止/延续
        data[-1]["action"] = data[-2]["action"].copy()

        # --------------------

        # 备份
        if BACKUP_ORIGINAL:
            backup_path = file_path + ".bak"
            if not os.path.exists(backup_path):
                os.rename(file_path, backup_path)

        # 写入
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

        return True

    except Exception as e:
        print(f"\n❌ 处理出错 {os.path.basename(file_path)}: {e}")
        return False


def main():
    pkl_files = glob(os.path.join(DATA_DIR, "*.pkl"))
    pkl_files.sort()

    print(f"📂 目标路径: {DATA_DIR}")
    print(f"📄 文件数量: {len(pkl_files)}")
    print(f"🔧 动作模式: {'EE Pose + Gripper (8维)' if INCLUDE_GRIPPER else 'Only EE Pose (7维)'}")
    print("🔄 逻辑: Action[t] <- State[t+1]")

    input("按 Enter 开始处理 (Ctrl+C 取消)...")

    success_count = 0
    for pkl_path in tqdm(pkl_files, desc="Processing"):
        if pkl_path.endswith(".bak"):
            continue

        if process_episode(pkl_path):
            success_count += 1

    print("\n" + "=" * 30)
    print(f"✅ 完成! 成功处理: {success_count} / {len(pkl_files)}")


if __name__ == "__main__":
    main()
