# 打印lerobot数据集结构
from pathlib import Path

from datasets import load_dataset

# ================= 配置 =================
# 你的数据集路径
DATASET_PATH = Path("/work/wzh/huggingface/lerobot/wmx/openpi_red_1227_205_clean")
# =======================================


def get_shape_or_len(obj):
    """辅助函数：获取数据的形状或长度"""
    if hasattr(obj, "shape"):
        return obj.shape
    if isinstance(obj, list):
        return f"List (len={len(obj)})"
    if isinstance(obj, (str, int, float, bool)):
        return type(obj).__name__
    if isinstance(obj, dict):
        return "Dict (Image bytes?)"
    return type(obj)


def main():
    print(f"📂 正在检查数据集结构: {DATASET_PATH}")

    # 1. 使用 Pandas 读取第一行 (极速，仅读取 Schema)
    # LeRobot 的数据存储在 data/ 目录下的 parquet 文件中
    parquet_files = list(DATASET_PATH.glob("data/**/*.parquet"))

    if not parquet_files:
        print("❌ 未找到 Parquet 文件")
        return

    print(f"📄 找到数据文件: {parquet_files[0].name} (共 {len(parquet_files)} 个)")

    try:
        # 使用 HuggingFace Datasets 读取 (最标准的方式)
        ds = load_dataset("parquet", data_files=str(parquet_files[0]), split="train", streaming=True)
        sample = next(iter(ds))  # 获取第一个样本

        print("\n" + "=" * 40)
        print("🔍 数据集 Keys 结构一览")
        print("=" * 40)

        # 排序 Key 以便查看
        keys = sorted(sample.keys())

        for key in keys:
            value = sample[key]
            info = get_shape_or_len(value)

            # 特殊处理：如果是 Image，通常是 dict {'bytes': ..., 'path': ...}
            if "image" in key and isinstance(value, dict) and "bytes" in value:
                info = "Image (Encoded Bytes)"

            print(f"🔑 {key:<35} | 📦 {info!s}")

        print("=" * 40)

        # 2. 重点检查 Action 和 State
        print("\n🤖 核心向量维度检查:")
        if "action" in sample:
            print(f"   ► Action: {sample['action'][:3]} ... (Total len: {len(sample['action'])})")

        if "observation.state" in sample:
            print(f"   ► State : {sample['observation.state'][:3]} ... (Total len: {len(sample['observation.state'])})")

    except Exception as e:
        print(f"❌ 读取出错: {e}")


if __name__ == "__main__":
    main()
