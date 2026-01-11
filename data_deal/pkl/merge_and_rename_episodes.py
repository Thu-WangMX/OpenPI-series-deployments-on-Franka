import os
import glob
import shutil
import re
from tqdm import tqdm  # 如果没有装 tqdm，可以 pip install tqdm，或者把下面的 tqdm() 去掉

# ================= ⚙️ 配置区域 =================
# 来源文件夹列表 (按顺序合并：列表第一个文件夹的数据排在前面)
SOURCE_DIRS = [
    "/work/wmx/dataset/dataset_1225/data_red_300",
    "/work/wmx/openpi/dataset_1225/data_red_1227_70"
]

# 目标文件夹
DST_DIR = "/work/wmx/dataset_1227_200"
# ===============================================

def get_file_index(filename):
    """从文件名中提取数字索引，用于排序"""
    match = re.search(r'episode_(\d+)\.pkl', filename)
    if match:
        return int(match.group(1))
    return float('inf') # 如果没找到数字，排到最后

def main():
    # 1. 检查源目录
    for d in SOURCE_DIRS:
        if not os.path.exists(d):
            print(f"❌ 错误: 源目录不存在 -> {d}")
            return

    # 2. 准备目标目录
    if not os.path.exists(DST_DIR):
        os.makedirs(DST_DIR)
        print(f"📁 创建目标目录: {DST_DIR}")
    else:
        print(f"⚠️  警告: 目标目录已存在: {DST_DIR}")
        print("    新文件将混入其中，如果这不是你想要的，请先清空目标目录。")
        # 简单防呆：如果里面有文件，询问是否继续？这里默认继续，但在生产环境最好检查

    all_files_ordered = []

    # 3. 收集并排序文件
    print("🔍 正在扫描源文件...")
    for src_dir in SOURCE_DIRS:
        # 获取该目录下所有pkl文件
        files = glob.glob(os.path.join(src_dir, "episode_*.pkl"))
        
        # 按文件名中的数字大小排序 (关键步骤)
        files.sort(key=lambda x: get_file_index(os.path.basename(x)))
        
        print(f"   -> 在 {os.path.basename(src_dir)} 中找到 {len(files)} 个文件")
        all_files_ordered.extend(files)

    total_files = len(all_files_ordered)
    print(f"📊 总计需要合并的文件数: {total_files}")
    print("-" * 50)

    # 4. 执行复制并重命名
    print("🚀 开始复制并重命名...")
    
    for new_idx, src_path in enumerate(tqdm(all_files_ordered, desc="Merging")):
        # 定义新文件名: episode_0.pkl, episode_1.pkl ...
        new_filename = f"episode_{new_idx}.pkl"
        dst_path = os.path.join(DST_DIR, new_filename)
        
        try:
            # 复制文件 (保留元数据)
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            print(f"❌ 复制失败: {src_path} -> {e}")

    print("-" * 50)
    print("🎉 合并完成！")
    print(f"📂 新数据集位置: {DST_DIR}")
    print(f"🔢 索引范围: episode_0.pkl ~ episode_{total_files - 1}.pkl")

    # 简单的验证
    dst_files = glob.glob(os.path.join(DST_DIR, "episode_*.pkl"))
    print(f"✅ 目标文件夹内实际文件数: {len(dst_files)}")

if __name__ == "__main__":
    main()