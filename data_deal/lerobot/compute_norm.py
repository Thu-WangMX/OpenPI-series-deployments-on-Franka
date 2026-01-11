# 为state和action计算norm
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ================= 配置区域 =================
# 直接指向你的数据集最终文件夹 (不需要分 root 和 repo_id)
DATASET_DIR = Path("/work/wzh/huggingface/lerobot/wmx/openpi_red_1223_274_clean")
# ===========================================


def safe_stack(series):
    """辅助函数：将 pandas 中的 list/array 列堆叠成 numpy 矩阵"""
    # 过滤掉 None
    valid_data = [x for x in series if x is not None]
    if not valid_data:
        return None
    # 某些时候 parquet 存的是 numpy array，某些时候是 list
    return np.stack(valid_data)


def main():
    print(f"📂 目标目录: {DATASET_DIR}")

    # 1. 搜索所有的 parquet 文件
    # LeRobot 的数据通常在 data/chunk-xxx/*.parquet
    parquet_files = sorted(list(DATASET_DIR.glob("data/**/*.parquet")))

    if not parquet_files:
        print("❌ 未找到 .parquet 文件！请检查路径是否正确。")
        return

    print(f"✅ 发现 {len(parquet_files)} 个数据文件，准备读取...")

    # 2. 仅读取 Action 和 State 列 (只读数值，极大节省内存)
    # columns=['action', 'observation.state'] 是关键，它会跳过图像数据的加载
    all_actions = []
    all_states = []

    # 动态检测是否存在 observation.state
    first_df = pd.read_parquet(parquet_files[0])
    cols_to_read = ["action"]
    has_state = "observation.state" in first_df.columns
    if has_state:
        cols_to_read.append("observation.state")

    print(f"🚀 正在读取列: {cols_to_read}")

    for p_file in tqdm(parquet_files, desc="Loading Parquet"):
        try:
            # 只加载指定列
            df = pd.read_parquet(p_file, columns=cols_to_read)

            # 收集 Action
            # Parquet 读取出来的通常是 array wrapper，需要 stack
            # 这里先存 list，最后统一 stack
            for act in df["action"]:
                all_actions.append(act)

            # 收集 State
            if has_state:
                for st in df["observation.state"]:
                    all_states.append(st)

        except Exception as e:
            print(f"⚠️ 读取 {p_file.name} 失败: {e}")

    # 3. 转换为 Numpy 进行计算
    print("⚡ 正在计算统计量...")

    # Action 处理
    actions_np = np.stack(all_actions).astype(np.float32)
    print(f"   Action Matrix Shape: {actions_np.shape}")

    stats = {}
    stats["action"] = {
        "mean": actions_np.mean(axis=0).tolist(),
        "std": actions_np.std(axis=0).tolist(),
        "min": actions_np.min(axis=0).tolist(),
        "max": actions_np.max(axis=0).tolist(),
    }

    # State 处理
    if has_state and all_states:
        states_np = np.stack(all_states).astype(np.float32)
        print(f"   State Matrix Shape:  {states_np.shape}")
        stats["observation.state"] = {
            "mean": states_np.mean(axis=0).tolist(),
            "std": states_np.std(axis=0).tolist(),
            "min": states_np.min(axis=0).tolist(),
            "max": states_np.max(axis=0).tolist(),
        }

    # 4. 填充图像默认值 (不做耗时计算)
    print("🖼️ 填充图像默认统计值...")
    # 扫描元数据或利用已知信息推断图像 key
    # 这里我们暴力扫描一下第一帧的 key 即可，不用加载数据
    # 假设标准命名
    img_keys = ["observation.images.image", "observation.images.image2"]

    for img_key in img_keys:
        # 默认 RGB 3通道
        c = 3
        stats[img_key] = {"mean": [0.5] * c, "std": [0.5] * c, "min": [0.0] * c, "max": [1.0] * c}

    # 5. 保存 stats.json
    meta_dir = DATASET_DIR / "meta"
    meta_dir.mkdir(exist_ok=True)
    stats_path = meta_dir / "stats.json"

    print(f"💾 写入文件: {stats_path}")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    print("\n🎉 搞定！Action 统计预览:")
    print(f"   Mean: {stats['action']['mean'][:3]}")
    print(f"   Std : {stats['action']['std'][:3]}")


if __name__ == "__main__":
    main()
