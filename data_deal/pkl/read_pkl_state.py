# 读取pkl文件的state并打印
import pickle

import numpy as np


def load_states_from_pkl(file_path):
    """
    读取 pkl 文件并提取 'observation.state'
    """
    try:
        with open(file_path, "rb") as f:
            # 1. 加载数据，data 是一个 list，长度约为 251
            data = pickle.load(f)

        print(f"成功加载文件: {file_path}")
        print(f"总时间步数 (Steps): {len(data)}")

        # 2. 提取 state
        # 根据你的结构: step -> 'observations' -> 'observation.state'
        all_states = []

        for i, step in enumerate(data):
            # 获取当前步的 observation 字典
            obs = step.get("observations", {})

            # 提取 state (根据你的 log，key 是 'observation.state')
            # 如果你的 state 定义是 agent_pos，可以将下行改为 obs.get('agent_pos')
            state = obs.get("observation.state")

            if state is not None:
                all_states.append(state)
            else:
                print(f"Warning: Step {i} 缺少 observation.state")

        # 3. 转换为 numpy 数组方便后续处理
        # 形状应该是 (T, 8), T=251
        states_np = np.array(all_states)

        return states_np

    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return None
    except Exception as e:
        print(f"读取过程中发生错误: {e}")
        return None


# =================使用示例=================
if __name__ == "__main__":
    pkl_file = "/work/wmx/openpi/data_1213/merged_all_episodes/episode_0.pkl"  # 替换为你的实际文件路径

    states = load_states_from_pkl(pkl_file)

    if states is not None:
        print("\n=== 数据统计 ===")
        print(f"State 数组形状: {states.shape}")  # 预期输出: (251, 8)
        print(f"数据类型: {states.dtype}")

        # 打印前 3 帧的 state 查看数值
        print("\n前 3 帧 State 数据:")
        print(states[:3])
