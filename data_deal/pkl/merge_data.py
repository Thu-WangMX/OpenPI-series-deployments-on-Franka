# 把多个pkl文件夹下的文件合并成一个文件夹并同一按顺序命名
import os
import shutil

# 1. 定义源文件夹路径 (注意：这里保留了你路径中的 'tow' 拼写)
source_dirs = [
    "/work/wmx/openpi/data_clean/pick_red_chili_peppers",
    "/work/wmx/openpi/data_clean/pick_tow_of_the_dolls_1",
    "/work/wmx/openpi/data_clean/pick_tow_of_the_dolls_2",
]

# 2. 定义目标文件夹路径
target_dir = "/work/wmx/openpi/data_clean/single_grasp"


def merge_and_rename():
    # 如果目标目录不存在，则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"创建目标目录: {target_dir}")

    global_idx = 0

    print("-" * 50)
    print("开始合并数据...")

    for src_dir in source_dirs:
        if not os.path.exists(src_dir):
            print(f"⚠️ 警告: 源路径不存在，跳过: {src_dir}")
            continue

        # 获取源目录下的所有 pkl 文件并排序，保证顺序确定性
        files = sorted([f for f in os.listdir(src_dir) if f.endswith(".pkl")])

        if not files:
            print(f"目录为空: {src_dir}")
            continue

        start_idx = global_idx

        for file_name in files:
            src_file = os.path.join(src_dir, file_name)

            # 构造新的文件名: episode_0.pkl, episode_1.pkl ...
            new_file_name = f"episode_{global_idx}.pkl"
            dst_file = os.path.join(target_dir, new_file_name)

            # 执行复制操作 (使用 copy2 保留文件元数据)
            shutil.copy2(src_file, dst_file)

            global_idx += 1

        end_idx = global_idx - 1
        print(f"已处理: {os.path.basename(src_dir)}")
        print(f"   └── 映射范围: episode_{start_idx} -> episode_{end_idx} (共 {len(files)} 个)")

    print("-" * 50)
    print("✅ 合并完成！")
    print(f"📁 总文件数: {global_idx}")
    print(f"📂 保存位置: {target_dir}")


if __name__ == "__main__":
    merge_and_rename()
