# 12.17晚上用于验证数据处理是否正确
import glob
import os
import pickle
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

# ================= 配置路径 =================
# 原始数据文件夹
ORIG_DIR = "/work/wmx/dataset/dataset_Pick_the_red_chili_pepper_doll_into_the_basket._2025-12-18_22-17-52"
# 处理后的数据文件夹
PROCESSED_DIR = "/work/wmx/openpi/dataset_1218/data_red_fixed_50"


# ================= 辅助函数 (用于重新计算验证) =================
def matrix_to_rotation6d(matrix):
    batch_dim = matrix.shape[0]
    return matrix[:, :, :2].swapaxes(1, 2).reshape(batch_dim, 6)


def quaternion_to_rotation6d_verify(quat):
    """验证用的独立转换函数"""
    # 确保输入是 (N, 4)
    quat = np.atleast_2d(quat)
    # 你的数据是 [w, x, y, z]，Scipy 需要 [x, y, z, w]
    scipy_quat = np.concatenate([quat[:, 1:], quat[:, 0:1]], axis=1)
    r = R.from_quat(scipy_quat)
    matrix = r.as_matrix()
    return matrix_to_rotation6d(matrix).flatten()


def check_single_file(filename):
    orig_path = os.path.join(ORIG_DIR, filename)
    proc_path = os.path.join(PROCESSED_DIR, filename)

    if not os.path.exists(proc_path):
        print(f"❌ 错误: 找不到处理后的文件: {proc_path}")
        return False

    print(f"🔍 正在验证文件: {filename}")

    with open(orig_path, "rb") as f:
        orig_data = pickle.load(f)
    with open(proc_path, "rb") as f:
        proc_data = pickle.load(f)

    if len(orig_data) != len(proc_data):
        print(f"❌ 帧数不匹配! 原文件: {len(orig_data)}, 新文件: {len(proc_data)}")
        return False

    # 随机抽查 3 帧 (开头, 中间, 结尾)
    indices = [0, len(orig_data) // 2, len(orig_data) - 1]

    for idx in indices:
        orig = orig_data[idx]
        new = proc_data[idx]

        # --- 1. 验证 State (10维: Pos(3) + Rot6D(6) + Gripper(1)) ---
        orig_tcp = orig["observations"]["orin_state"]["tcp_pose"]
        orig_grip = orig["observations"]["orin_state"]["gripper_pose"]

        # 重新计算预期值
        expected_pos = orig_tcp[:3]
        expected_rot6d = quaternion_to_rotation6d_verify(orig_tcp[3:])
        expected_state = np.concatenate([expected_pos, expected_rot6d, [orig_grip]])

        actual_state = new["observations"]["state"]

        if actual_state.shape != (10,):
            print(f"❌ [Frame {idx}] State 维度错误! 期望 (10,), 实际 {actual_state.shape}")
            return False

        if not np.allclose(actual_state, expected_state, atol=1e-5):
            print(f"❌ [Frame {idx}] State 数值不匹配!")
            print(f"   期望 (前4): {expected_state[:4]}")
            print(f"   实际 (前4): {actual_state[:4]}")
            return False

        # --- 2. 验证 Action (10维: Next Pos + Next Rot6D + Next Gripper) ---
        # 注意：Action 应该来自原数据的 next_observations
        next_tcp = orig["next_observations"]["orin_state"]["tcp_pose"]
        next_grip = orig["next_observations"]["orin_state"]["gripper_pose"]

        expected_action_pos = next_tcp[:3]
        expected_action_rot6d = quaternion_to_rotation6d_verify(next_tcp[3:])
        expected_action = np.concatenate([expected_action_pos, expected_action_rot6d, [next_grip]])

        actual_action = new["action"]

        if actual_action.shape != (10,):
            print(f"❌ [Frame {idx}] Action 维度错误! 期望 (10,), 实际 {actual_action.shape}")
            return False

        if not np.allclose(actual_action, expected_action, atol=1e-5):
            print(f"❌ [Frame {idx}] Action 数值不匹配!")
            return False

        # --- 3. 验证 TCP Wrench (6维: Force + Torque) ---
        orig_force = orig["observations"]["orin_state"]["tcp_force"]
        orig_torque = orig["observations"]["orin_state"]["tcp_torque"]
        expected_wrench = np.concatenate([orig_force, orig_torque])

        actual_wrench = new["observations"]["tcp_wrench"]

        if actual_wrench.shape != (6,):
            print(f"❌ [Frame {idx}] Wrench 维度错误! 期望 (6,), 实际 {actual_wrench.shape}")
            return False

        if not np.allclose(actual_wrench, expected_wrench, atol=1e-5):
            print(f"❌ [Frame {idx}] Wrench 数值不匹配!")
            return False

        # --- 4. 验证 Effort (Tau_J) ---
        orig_tau = orig["observations"]["orin_state"]["tau_J"]
        actual_effort = new["observations"]["effort"]

        if not np.allclose(actual_effort, orig_tau, atol=1e-5):
            print(f"❌ [Frame {idx}] Effort (Tau_J) 数值不匹配!")
            return False

    print("✅ 该文件所有检查项目通过!")
    return True


def main():
    # 查找所有处理过的 pkl 文件
    pkl_files = glob.glob(os.path.join(PROCESSED_DIR, "*.pkl"))
    if not pkl_files:
        print(f"⚠️ 目录 {PROCESSED_DIR} 下没有找到 .pkl 文件。请先运行处理脚本。")
        sys.exit(1)

    print(f"📂 找到 {len(pkl_files)} 个文件。将抽查其中 3 个...")

    # 随机抽查 3 个文件（如果文件少于3个则全查）
    files_to_check = pkl_files[:3] if len(pkl_files) > 3 else pkl_files

    all_passed = True
    for pkl_path in files_to_check:
        filename = os.path.basename(pkl_path)
        if not check_single_file(filename):
            all_passed = False
            break

    if all_passed:
        print("\n🎉🎉🎉 验证成功！所有抽查文件数据结构和数值均正确！ 🎉🎉🎉")
        print(f"数据位置: {PROCESSED_DIR}")
        print("Observation Keys: ['state', 'tcp_wrench', 'effort', ...]")
        print("Action Shape: (10,)")
    else:
        print("\n🚫🚫🚫 验证失败，请检查处理脚本逻辑。")


if __name__ == "__main__":
    main()
