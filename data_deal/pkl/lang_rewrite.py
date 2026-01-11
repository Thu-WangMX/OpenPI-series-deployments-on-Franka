# 修改了120条抓橙色玩偶的pkl的task描述
import os
import pickle

# 定义需要处理的文件夹路径
target_dirs = [
    "/work/wmx/openpi/data_clean/pick_tow_of_the_dolls_1",
    "/work/wmx/openpi/data_clean/pick_tow_of_the_dolls_2",
]

# 定义新的任务描述
new_description = "Put the dolls on the table into the basket."


def process_pkl_files():
    count = 0
    for folder_path in target_dirs:
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            print(f"Warning: 路径不存在，跳过: {folder_path}")
            continue

        print(f"正在处理文件夹: {folder_path} ...")

        # 获取文件夹下所有文件
        files = os.listdir(folder_path)

        for file_name in files:
            if file_name.endswith(".pkl"):
                file_path = os.path.join(folder_path, file_name)

                try:
                    # 1. 读取数据
                    with open(file_path, "rb") as f:
                        data = pickle.load(f)

                    # 2. 修改数据
                    # 数据结构是一个 List，我们需要遍历 list 中的每一帧 (dict)
                    if isinstance(data, list):
                        for step_data in data:
                            if isinstance(step_data, dict):
                                # 修改 observations 中的 task_description
                                if "observations" in step_data and isinstance(step_data["observations"], dict):
                                    step_data["observations"]["task_description"] = new_description

                                # 修改 next_observations 中的 task_description
                                if "next_observations" in step_data and isinstance(
                                    step_data["next_observations"], dict
                                ):
                                    step_data["next_observations"]["task_description"] = new_description

                                # 修改 language_instruction (通常与 task_description 保持一致)
                                if "language_instruction" in step_data:
                                    step_data["language_instruction"] = new_description
                    else:
                        print(f"跳过文件 {file_name}: 数据结构不是 list")
                        continue

                    # 3. 保存回文件
                    with open(file_path, "wb") as f:
                        pickle.dump(data, f)

                    count += 1

                except Exception as e:
                    print(f"处理文件 {file_name} 时出错: {e}")

    print(f"\n处理完成！共修改了 {count} 个 pkl 文件。")


if __name__ == "__main__":
    process_pkl_files()
