# 检查lerobot结构是否正确
import json
from pathlib import Path

from datasets import load_dataset

ROOT_PATH = Path(" /work/wzh/huggingface/lerobot/wmx/openpi_red_1222_300")


def load_task_map(root_path):
    """读取 meta/tasks.jsonl 构建 index -> text 的映射"""
    task_map = {}
    task_file = root_path / "meta/tasks.jsonl"
    if task_file.exists():
        with open(task_file) as f:
            for line in f:
                item = json.loads(line)
                # 通常结构是 {"task_index": 0, "task": "description..."}
                task_map[item["task_index"]] = item["task"]
    return task_map


def main():
    print(f"📂 数据集: {ROOT_PATH}")

    # 1. 检查 Stats
    if (ROOT_PATH / "meta/stats.json").exists():
        print("✅ meta/stats.json 存在")
    else:
        print("❌ meta/stats.json 依然缺失！请运行 compute_stats.py")

    # 2. 加载 Task 映射
    task_map = load_task_map(ROOT_PATH)
    print(f"✅ 加载了 {len(task_map)} 个任务描述")

    # 3. 加载数据
    data_files = str(ROOT_PATH / "data/**/*.parquet")
    ds = load_dataset("parquet", data_files=data_files, split="train")
    print(f"📊 总帧数: {len(ds)}")

    # 4. 抽样检查
    indices = [0, len(ds) // 2, len(ds) - 1]

    print("\n🔍 内容抽样:")
    for i in indices:
        item = ds[i]

        # --- 获取 Task ---
        t_idx = item.get("task_index")
        # 如果有 index，去 map 里查；如果没有，尝试直接读 string
        task_str = task_map.get(t_idx, "Unknown Task") if t_idx is not None else item.get("task", "N/A")

        # --- 获取 Action ---
        act = item["action"]

        print(f"Frame [{i}]:")
        print(f'  📝 Task Index: {t_idx} -> "{task_str}"')
        print(f"  🦾 Action[:3]: {act[:3]}")
        print("-" * 30)


if __name__ == "__main__":
    main()
