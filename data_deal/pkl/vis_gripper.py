# import pickle
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# # ================= 配置区域 =================
# # 请将此路径替换为你 episode_234.pkl 实际所在的完整路径
# FILE_PATH = "/work/wmx/dataset_1222_processed/1222_processed/dataset_Pick_the_red_chili_pepper_doll_into_the_basket._2025-12-22_14-49-06/episode_1.pkl"
# DIM_INDEX = 9  # 第10维 (索引为9)
# # ===========================================

# def plot_gripper_action(pkl_path, dim_idx):
#     if not os.path.exists(pkl_path):
#         print(f"❌ 错误: 找不到文件 {pkl_path}")
#         return

#     print(f"正在读取文件: {pkl_path} ...")

#     try:
#         with open(pkl_path, 'rb') as f:
#             data = pickle.load(f)

#         gripper_values = []

#         # 遍历数据提取指定维度的值
#         if isinstance(data, list):
#             for i, step in enumerate(data):
#                 # 尝试获取 'action' 或 'actions'
#                 action = step.get('action', step.get('actions'))

#                 if action is None:
#                     print(f"⚠️ 第 {i} 步没有找到 'action' 数据")
#                     continue

#                 # 转换为 numpy array 以便处理
#                 action = np.array(action)

#                 if action.shape[0] > dim_idx:
#                     gripper_values.append(action[dim_idx])
#                 else:
#                     print(f"⚠️ 第 {i} 步 action 维度不足 (长度 {action.shape[0]}, 需要索引 {dim_idx})")
#         else:
#             print("❌ 数据格式错误: 期望是一个 list")
#             return

#         if len(gripper_values) == 0:
#             print("❌ 未提取到任何数据，无法绘图")
#             return

#         # 开始绘图
#         plt.figure(figsize=(10, 6))
#         plt.plot(gripper_values, marker='o', markersize=3, linestyle='-', label=f'Action Dim {dim_idx+1}')

#         # 添加辅助线 (Open/Close 阈值)
#         plt.axhline(y=0.04, color='g', linestyle='--', alpha=0.5, label='Open Threshold (0.04)')
#         plt.axhline(y=0.02, color='r', linestyle='--', alpha=0.5, label='Close Threshold (0.02)')

#         plt.title(f'Action Dimension {dim_idx + 1} (Gripper) - {os.path.basename(pkl_path)}')
#         plt.xlabel('Time Step')
#         plt.ylabel('Value')
#         plt.legend()
#         plt.grid(True, alpha=0.3)

#         # 保存图片
#         save_name = "gripper_plot.png"
#         plt.savefig(save_name)
#         print(f"✅ 绘图完成！图片已保存为: {save_name}")

#         # 如果是在本地运行支持 GUI，可以取消下面这行的注释来显示图片
#         # plt.show()

#     except Exception as e:
#         print(f"❌ 读取或处理时发生错误: {e}")

# if __name__ == "__main__":
#     plot_gripper_action(FILE_PATH, DIM_INDEX)

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

# ================= ⚙️ 配置区域 =================
# 请将此路径替换为你 pickle 文件实际所在的完整路径
FILE_PATH = "/work/wmx/dataset_1222_processed/1222_processed/dataset_Pick_the_red_chili_pepper_doll_into_the_basket._2025-12-22_15-57-15/episode_1.pkl"
# 标志位：如果 gripper_pose 是个数组，取哪一维？
# 通常 gripper_pose 可能是 [width] (1维) 或者 [x,y,z,qx,qy,qz,width] (7维)
# 设为 -1 表示自动取最后一个值（通常是宽度）
GRIPPER_DIM_INDEX = -1
# ===============================================


def inspect_structure(data_sample):
    """辅助函数：打印数据结构帮助调试"""
    print("\n🔍 --- 数据结构探测 ---")
    # 优先检查复数形式
    if "observations" in data_sample:
        obs = data_sample["observations"]
        print(f"Dataset keys in 'observations': {list(obs.keys())}")
        if "orin_state" in obs and isinstance(obs["orin_state"], dict):
            print(f"Dataset keys in 'observations' -> 'orin_state': {list(obs['orin_state'].keys())}")
    # 兼容检查单数形式
    elif "observation" in data_sample:
        obs = data_sample["observation"]
        print(f"Dataset keys in 'observation': {list(obs.keys())}")
    else:
        print(f"Top level keys: {list(data_sample.keys())}")
    print("-----------------------\n")


def plot_gripper_observation(pkl_path):
    if not os.path.exists(pkl_path):
        print(f"❌ 错误: 找不到文件 {pkl_path}")
        return

    print(f"📂 正在读取文件: {pkl_path} ...")

    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, list) or len(data) == 0:
            print("❌ 数据格式错误: 期望是一个非空 list")
            return

        # 1. 探测阶段：尝试找到正确的 key
        first_step = data[0]
        obs_key_path = None
        main_obs_key = "observations"  # 默认为复数

        # 打印结构供参考
        inspect_structure(first_step)

        # 确定主 Key 是单数还是复数
        if "observations" in first_step:
            main_obs_key = "observations"
        elif "observation" in first_step:
            main_obs_key = "observation"
        else:
            print("❌ 未在顶层找到 'observations' 或 'observation'")
            return

        obs = first_step[main_obs_key]

        # 路径尝试逻辑
        # 路径 A: observations -> orin_state -> gripper_pose (这是你数据目前的结构)
        if "orin_state" in obs and isinstance(obs["orin_state"], dict) and "gripper_pose" in obs["orin_state"]:
            obs_key_path = "nested_orin_state"
            print(f"✅ 锁定路径: data['{main_obs_key}']['orin_state']['gripper_pose']")

        # 路径 B: observations -> gripper_pose (备用)
        elif "gripper_pose" in obs:
            obs_key_path = "direct_obs"
            print(f"✅ 锁定路径: data['{main_obs_key}']['gripper_pose']")

        if obs_key_path is None:
            print("❌ 未在 observation 中找到 'gripper_pose'")
            return

        # 2. 提取数据
        gripper_values = []
        for i, step in enumerate(data):
            try:
                obs = step[main_obs_key]

                # 根据探测到的路径提取
                if obs_key_path == "nested_orin_state":
                    val = obs["orin_state"]["gripper_pose"]
                else:
                    val = obs["gripper_pose"]

                # 3. 数据处理 (处理标量或数组)
                val = np.array(val)

                if val.ndim == 0:  # 标量
                    gripper_values.append(val.item())
                elif val.ndim >= 1:  # 数组
                    # 如果只有一个元素，直接取
                    if val.size == 1:
                        gripper_values.append(val.item())
                    else:
                        # 如果是多维数组，取配置的维度
                        gripper_values.append(val[GRIPPER_DIM_INDEX])

            except KeyError as e:
                print(f"⚠️ 第 {i} 步缺少 key: {e}")
                continue

        if len(gripper_values) == 0:
            print("❌ 未提取到任何数据")
            return

        # 4. 绘图
        plt.figure(figsize=(12, 6))
        plt.plot(
            gripper_values, color="#ff7f0e", marker=".", markersize=4, linestyle="-", label="Observed Gripper State"
        )

        # 添加辅助线 (参考)
        plt.axhline(y=0.04, color="g", linestyle="--", alpha=0.5, label="Open Ref (0.04)")
        plt.axhline(y=0.015, color="r", linestyle="--", alpha=0.5, label="Close Ref (0.015)")

        plt.title(f"Observation: Gripper Pose\nFile: {os.path.basename(pkl_path)}")
        plt.xlabel("Time Step")
        plt.ylabel("Value (Width)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_name = "gripper_obs_plot.png"
        plt.savefig(save_name)
        print(f"✅ 绘图完成！图片已保存为: {save_name}")

    except Exception as e:
        print(f"❌ 发生未预期的错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    plot_gripper_observation(FILE_PATH)
