"""
LeRobot 数据集修复脚本
功能：
1. 手动计算并生成 stats.json (Fix Missing Stats)
2. 验证 task_index 是否正确区分了任务
"""

from pathlib import Path

from datasets import load_dataset
import numpy as np
import torch

# ================= 配置 =================
# 数据集根目录
ROOT_PATH = Path("/work/wzh/huggingface/lerobot/wmx/openpi_merged_single_grasp_newest")
# =======================================


def compute_stats(dataset):
    print("🧮 正在计算统计信息 (这可能需要几秒钟)...")

    # 提取所有 action (N, 7)
    # 注意：如果内存不够，可以分批读取。这里 3.6万帧应该没问题。
    actions = torch.tensor(dataset["action"])

    # 计算统计值
    stats = {
        "action": {
            "mean": actions.mean(dim=0).tolist(),
            "std": actions.std(dim=0).tolist(),
            "min": actions.min(dim=0).values.tolist(),
            "max": actions.max(dim=0).values.tolist(),
        }
    }

    # 如果有 state，也计算 state
    if "observation.state" in dataset.features:
        states = torch.tensor(dataset["observation.state"])
        stats["observation.state"] = {
            "mean": states.mean(dim=0).tolist(),
            "std": states.std(dim=0).tolist(),
            "min": states.min(dim=0).values.tolist(),
            "max": states.max(dim=0).values.tolist(),
        }

    return stats


def main():
    print(f"📂 目标路径: {ROOT_PATH}")

    # 1. 加载数据
    data_files = str(ROOT_PATH / "data/**/*.parquet")
    try:
        ds = load_dataset("parquet", data_files=data_files, split="train")
        print(f"✅ 数据加载成功: {len(ds)} 帧")
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return

    # 2. 验证 Task Index 分布
    print("\n🔍 检查 Task Index 分布...")
    if "task_index" in ds.features:
        task_indices = np.array(ds["task_index"])
        unique_tasks, counts = np.unique(task_indices, return_counts=True)
        print(f"   发现 {len(unique_tasks)} 种任务 ID: {unique_tasks}")
        for task_id, count in zip(unique_tasks, counts):
            print(f"   🆔 Task {task_id}: {count} 帧")

        if len(unique_tasks) >= 2:
            print("   ✅ 成功检测到混合任务 (Chili vs Dolls)！")
        else:
            print("   ⚠️ 警告: 只发现 1 种任务 ID，请确认合并是否成功。")
    else:
        print("   ❌ 未找到 task_index 列！")

    # 3. 计算并保存 Stats
    # stats = compute_stats(ds)

    # # 打印预览
    # print("\n📊 统计信息预览 (Action):")
    # print(f"   Mean: {np.array(stats['action']['mean'])[:3]} ...")
    # print(f"   Std:  {np.array(stats['action']['std'])[:3]} ...")

    # # 检查是否有非零 Std (如果全是0，说明数据有问题)
    # if np.all(np.array(stats['action']['std']) < 1e-6):
    #     print("   ❌ [严重警告] Action Std 接近 0！这说明所有数据的动作可能都是静止的，或者读取错误！")
    # else:
    #     print("   ✅ Action 数据分布正常。")

    # # 4. 写入文件
    # meta_dir = ROOT_PATH / "meta"
    # meta_dir.mkdir(exist_ok=True)
    # stats_path = meta_dir / "stats.json"

    # with open(stats_path, 'w') as f:
    #     json.dump(stats, f, indent=4)

    # print(f"\n💾 已保存统计信息至: {stats_path}")
    # print("🎉 修复完成！现在可以开始训练了。")


if __name__ == "__main__":
    main()
