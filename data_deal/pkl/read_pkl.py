# 读取pkl文件夹中的文件结构并批量验证是否结构一致
from pathlib import Path
import pickle

import numpy as np
import torch

# ================= 配置 =================
# 您的数据文件夹路径
DATA_DIR = Path("/work/wmx/dataset_1227_205")
# =======================================


def inspect_structure(data, indent=0, max_depth=3):
    """递归打印数据结构 (类型, 形状, 键名)"""
    if indent > max_depth:
        return

    prefix = "  " * indent
    data_type = type(data)

    if isinstance(data, dict):
        print(f"{prefix}Type: {data_type.__name__}, Keys: {list(data.keys())}")
        for key in data:
            # 只展示第一层或特定层的 key，防止过多
            print(f"{prefix}  Key: '{key}'")
            inspect_structure(data[key], indent + 1, max_depth)

    elif isinstance(data, (list, tuple)):
        print(f"{prefix}Type: {data_type.__name__}, Length: {len(data)}")
        if len(data) > 0:
            print(f"{prefix}  [Element 0 Sample]:")
            inspect_structure(data[0], indent + 1, max_depth)

    elif isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
        print(f"{prefix}Type: {data_type.__name__}, Shape: {data.shape}, Dtype: {data.dtype}")

    else:
        # 打印标量值（如果是简单的数字或布尔值）
        print(f"{prefix}Type: {data_type.__name__}, Value: {data}")


def get_keys_recursive(data):
    """提取第一帧的所有 Keys 用于一致性对比"""
    if isinstance(data, list) and len(data) > 0:
        return get_keys_recursive(data[0])
    if isinstance(data, dict):
        keys = set(data.keys())
        for k, v in data.items():
            if isinstance(v, dict):
                # 简单处理嵌套字典 key，例如 'observations/state'
                sub_keys = get_keys_recursive(v)
                if isinstance(sub_keys, set):
                    for sk in sub_keys:
                        keys.add(f"{k}/{sk}")
        return keys
    return None


def main():
    if not DATA_DIR.exists():
        print(f"错误: 路径不存在 {DATA_DIR}")
        return

    pkl_files = sorted(list(DATA_DIR.glob("*.pkl")))
    print(f"在 {DATA_DIR} 下共找到 {len(pkl_files)} 个 .pkl 文件。\n")

    if not pkl_files:
        return

    # --- 1. 深度分析第一个文件 ---
    first_file = pkl_files[0]
    print("=" * 60)
    print(f"【深度分析样本】: {first_file.name}")
    print("=" * 60)

    try:
        with open(first_file, "rb") as f:
            first_data = pickle.load(f)

        inspect_structure(first_data)

        # 获取基准 Keys 用于后续对比
        baseline_keys = get_keys_recursive(first_data)
        print("\n" + "-" * 60)
        print(f"基准 Keys 集合: {baseline_keys}")
        print("-" * 60 + "\n")

    except Exception as e:
        print(f"读取第一个文件失败: {e}")
        return

    # --- 2. 批量扫描剩余文件 ---
    print(f"【批量扫描剩余 {len(pkl_files)-1} 个文件】...")
    print(f"{'文件名':<40} | {'帧数 (Frames)':<15} | {'结构一致性':<10}")
    print("-" * 75)

    for pkl_path in pkl_files[1:]:
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)

            length = len(data) if isinstance(data, list) else "N/A"

            # 简单的一致性检查
            current_keys = get_keys_recursive(data)
            is_consistent = current_keys == baseline_keys

            status = "✅ 匹配" if is_consistent else "❌ 异常"

            print(f"{pkl_path.name:<40} | {length!s:<15} | {status}")

            if not is_consistent:
                print(f"   >>> 警告: {pkl_path.name} 的 Keys 与基准不一致！")
                # 可选：打印差异
                # print(f"   差异: {current_keys ^ baseline_keys}")

        except Exception as e:
            print(f"{pkl_path.name:<40} | {'读取失败':<15} | {e}")


if __name__ == "__main__":
    main()
