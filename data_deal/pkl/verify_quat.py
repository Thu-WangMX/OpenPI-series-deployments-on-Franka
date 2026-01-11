import os
import pickle

import numpy as np

# 文件路径
FILE_PATH = "/work/wmx/dataset/dataset_1222/data_red_300/episode_66.pkl"


def check_quaternion_convention():
    if not os.path.exists(FILE_PATH):
        print(f"❌ 错误: 文件不存在 -> {FILE_PATH}")
        return

    print(f"📂 正在加载文件: {FILE_PATH} ...")
    with open(FILE_PATH, "rb") as f:
        data = pickle.load(f)

    # 假设数据是列表结构
    if not isinstance(data, list):
        print(f"⚠️ 警告: 数据不是列表，而是 {type(data)}，可能需要调整代码结构。")
        return

    print(f"📊 总帧数: {len(data)}")
    print("-" * 60)
    print(f"{'Frame':<5} | {'XYZ (Position)':<25} | {'Quaternion (Last 4)':<35}")
    print("-" * 60)

    # 统计 W 分量可能出现的位置
    # w_index_counts 记录每一帧中绝对值最大的那个分量的索引（相对于4维四元数向量）
    # 0 -> wxyz (w在首位)
    # 3 -> xyzw (w在末位)
    w_index_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    # 采样前 10 帧（或者全部帧）进行观察
    sample_frames = min(10, len(data))

    for i in range(sample_frames):
        frame = data[i]

        # 获取 tcp_pose
        # 根据你之前的代码，路径是 ['observations']['orin_state']['tcp_pose']
        try:
            if "observations" in frame:
                obs = frame["observations"]
            else:
                obs = frame  # 兼容可能的结构差异

            tcp_pose = obs["orin_state"]["tcp_pose"]
            tcp_pose = np.array(tcp_pose)

            # 分割 XYZ 和 Quat
            pos = tcp_pose[:3]
            quat = tcp_pose[3:]  # 取后4位

            # 格式化打印
            pos_str = f"[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"
            quat_str = f"[{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]"
            print(f"{i:<5} | {pos_str:<25} | {quat_str:<35}")

            # 简单的启发式判断：
            # 在大多数标准姿态下，四元数通常接近 [1,0,0,0] (WXYZ) 或 [0,0,0,1] (XYZW)
            # 或者某些分量显著大于其他分量。
            # 找出绝对值最大的索引
            max_idx = np.argmax(np.abs(quat))
            w_index_counts[max_idx] += 1

        except KeyError as e:
            print(f"❌ 帧 {i} 读取失败，缺少键值: {e}")
            break

    print("-" * 60)
    print("🕵️‍♂️ 自动分析结果:")

    # 打印最大值分布
    print(f"在采样帧中，绝对值最大的分量索引分布: {w_index_counts}")

    # 结论
    if w_index_counts[0] > w_index_counts[3]:
        print("✅ 结论: 看起来是 [W, X, Y, Z] 格式 (Scalar-First)")
        print("   -> 因为第1个数值 (Index 0) 的绝对值最大。")
    elif w_index_counts[3] > w_index_counts[0]:
        print("✅ 结论: 看起来是 [X, Y, Z, W] 格式 (Scalar-Last)")
        print("   -> 因为第4个数值 (Index 3) 的绝对值最大。")
    else:
        print("⚠️ 无法确定: 最大值分布不明显（可能是复杂的旋转姿态）。")
        print("   -> 请人工检查上面的数值，看哪一位接近 1.0 或 -1.0。")


if __name__ == "__main__":
    check_quaternion_convention()
