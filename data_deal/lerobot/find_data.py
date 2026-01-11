#寻找指定的lerobot数据集位置和结构
import os
from pathlib import Path

# ================= 核心修复区域 =================
# 尝试导入，如果失败则手动根据 HF_HOME 构造路径
try:
    from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
except (ImportError, ModuleNotFoundError):
    # 获取环境变量 HF_HOME，如果没设则用默认的 ~/.cache/huggingface
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        # 您的环境会走这里 -> /work/wzh/huggingface/lerobot
        LEROBOT_HOME = Path(hf_home) / "lerobot"
    else:
        LEROBOT_HOME = Path.home() / ".cache/huggingface/lerobot"
    
    print(f"⚠️ [兼容模式] 无法导入 LEROBOT_HOME，已手动定位至: {LEROBOT_HOME}")
# ===============================================

# ================= 配置 =================
REPO_ID = "wmx/openpi_merged_single_grasp_newest"
# =======================================

def main():
    # 1. 构造目标路径
    repo_path = LEROBOT_HOME / REPO_ID
    
    print(f"🔍 正在查找路径: {repo_path}")
    
    if not repo_path.exists():
        print(f"❌ 错误: 目录不存在！")
        print(f"  当前搜索路径: {repo_path}")
        print("  可能原因：数据集尚未转换成功，或者 Repo ID 写错了。")
        return

    print("\n📦 目录内容:")
    print("=" * 60)
    
    # 2. 遍历并打印文件
    file_count = 0
    total_size = 0
    
    # 使用 rglob 递归查找所有文件
    for p in sorted(repo_path.rglob("*")):
        if p.is_file():
            file_count += 1
            size_mb = p.stat().st_size / (1024 * 1024)
            total_size += size_mb
            
            # 打印相对路径和大小
            rel_path = p.relative_to(repo_path)
            print(f"📄 {str(rel_path):<40} | {size_mb:>8.2f} MB")
            
    print("=" * 60)
    print(f"✅ 总计: {file_count} 个文件, 共 {total_size:.2f} MB")

    # 3. 关键文件检查
    required_files = ["data_info.json", "meta_data.json"] 
    print("\n🧐 完整性检查:")
    for f in required_files:
        if (repo_path / f).exists():
            print(f"  [OK] 发现 {f}")
        else:
            print(f"  [MISSING] ⚠️ 缺少 {f} (LeRobot 无法加载)")
            
    data_files = list(repo_path.rglob("*.arrow")) + list(repo_path.rglob("*.parquet"))
    if data_files:
        print(f"  [OK] 发现 {len(data_files)} 个数据文件 (Arrow/Parquet)")
    else:
        print("  [MISSING] ⚠️ 没找到数据文件！转换可能未完成。")

if __name__ == "__main__":
    main()