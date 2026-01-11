# 读取meta/stats.json
import json
import os
from pathlib import Path

# ================= 配置 =================
REPO_ID = "wmx/openpi_single_force_185_1214"
TARGET_FILE = "meta/stats.json"
# =======================================


def main():
    # 1. 确定 LEROBOT_HOME (复用之前的逻辑)
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        LEROBOT_HOME = Path(hf_home) / "lerobot"
    else:
        LEROBOT_HOME = Path.home() / ".cache/huggingface/lerobot"

    # 2. 构造完整路径
    file_path = LEROBOT_HOME / REPO_ID / TARGET_FILE

    print(f"🔍 正在尝试读取: {file_path}")

    if not file_path.exists():
        print("❌ 错误: 文件不存在！")

        # 尝试寻找其他可能的统计文件位置
        possible_paths = [
            LEROBOT_HOME / REPO_ID / "stats.json",
            LEROBOT_HOME / REPO_ID / "meta_data.json",
            LEROBOT_HOME / REPO_ID / "data_info.json",
        ]
        print("💡 建议检查以下文件是否存在:")
        for p in possible_paths:
            if p.exists():
                print(f"   [存在] {p.name}")
            else:
                print(f"   [缺少] {p.name}")
        return

    # 3. 读取并打印 JSON
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        print("\n✅ 文件内容:")
        print("=" * 40)
        print(json.dumps(data, indent=4, ensure_ascii=False))
        print("=" * 40)

    except Exception as e:
        print(f"❌ 读取失败: {e}")


if __name__ == "__main__":
    main()
