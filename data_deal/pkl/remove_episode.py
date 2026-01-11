import os
import glob
import re

# ================= ⚙️ 配置区域 =================
DATA_DIR = "/work/wmx/dataset/dataset_1225/data_red_300"

# 需要删除的序号区间 (闭区间，包含起始和结束)
# 格式: (开始, 结束)
DELETE_RANGES = [
    (50, 69),
    (88, 91),
    (235, 302),
    (70, 87),
    (176, 218)
]
# ===============================================

def get_file_index(filename):
    """从文件名中提取数字索引"""
    match = re.search(r'episode_(\d+)\.pkl', filename)
    if match:
        return int(match.group(1))
    return None

def main():
    if not os.path.exists(DATA_DIR):
        print(f"❌ 目录不存在: {DATA_DIR}")
        return

    print(f"📂 正在处理目录: {DATA_DIR}")
    
    # 1. 生成所有需要删除的序号集合
    delete_indices = set()
    for start, end in DELETE_RANGES:
        for i in range(start, end + 1):
            delete_indices.add(i)
    
    print(f"🔍 计划删除的索引范围涵盖: {len(delete_indices)} 个序号")

    # 2. 扫描文件
    pkl_files = glob.glob(os.path.join(DATA_DIR, "episode_*.pkl"))
    files_to_keep = []
    deleted_count = 0

    print("🚀 开始执行删除操作...")
    
    for file_path in pkl_files:
        idx = get_file_index(os.path.basename(file_path))
        
        if idx is None:
            continue # 跳过不符合格式的文件

        if idx in delete_indices:
            try:
                os.remove(file_path)
                # print(f"   🗑️ 已删除: episode_{idx}.pkl") # 如果文件太多，可以注释掉这行
                deleted_count += 1
            except OSError as e:
                print(f"   ❌ 删除失败 {file_path}: {e}")
        else:
            files_to_keep.append((idx, file_path))

    print(f"✅ 删除完成! 共删除了 {deleted_count} 个文件。")
    print(f"📊 剩余文件数量: {len(files_to_keep)}")

    # 3. 重新排序与重命名
    # 必须按旧索引从小到大排序，保证时间顺序
    files_to_keep.sort(key=lambda x: x[0])

    print("🔄 开始重新排序命名 (从 episode_0.pkl 开始)...")
    
    # 第一步：先全部重命名为临时文件，防止命名冲突 (例如把 10 改成 5，而 5 还存在时)
    temp_files = []
    for i, (old_idx, old_path) in enumerate(files_to_keep):
        dir_name = os.path.dirname(old_path)
        temp_name = os.path.join(dir_name, f"temp_reindex_{i}.tmp")
        os.rename(old_path, temp_name)
        temp_files.append(temp_name)

    # 第二步：将临时文件重命名为最终目标
    for i, temp_path in enumerate(temp_files):
        dir_name = os.path.dirname(temp_path)
        final_name = os.path.join(dir_name, f"episode_{i}.pkl")
        os.rename(temp_path, final_name)
    
    print(f"🎉 全部完成！")
    print(f"   现在目录中共有 {len(temp_files)} 个文件。")
    print(f"   索引范围: episode_0.pkl -> episode_{len(temp_files)-1}.pkl")

if __name__ == "__main__":
    main()