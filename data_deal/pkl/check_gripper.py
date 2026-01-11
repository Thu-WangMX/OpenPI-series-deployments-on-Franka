from collections import Counter  # <--- 新增 defaultdict
from collections import defaultdict  # <--- 新增 defaultdict
import glob
import os
import pickle

import numpy as np

# ================= 配置区域 =================
DATA_DIR = "/work/wmx/dataset/dataset_1225/data_red_300"
GRIPPER_DIM_INDEX = 9  # 第10维 (索引9)
# 你的阈值设置
OPEN_THRESHOLD = 0.06
CLOSE_THRESHOLD = 0.015
# ===========================================


def check_gripper_trend(gripper_values, filename):
    """
    返回: (是否符合标准, 趋势字符串, 详细信息)
    """
    if len(gripper_values) == 0:
        return False, "Empty", "数据为空"

    max_val = np.max(gripper_values)
    min_val = np.min(gripper_values)

    # 简单的状态机
    states = []
    for v in gripper_values:
        if v > OPEN_THRESHOLD:
            states.append("1")  # Open
        elif v < CLOSE_THRESHOLD:
            states.append("0")  # Closed

    # 状态压缩 (例如 "1110011" -> "101")
    if not states:
        return False, "NoTrigger", f"数值范围异常 ({min_val:.4f} ~ {max_val:.4f})，未触发阈值"

    compressed = states[0]
    for s in states[1:]:
        if s != compressed[-1]:
            compressed += s

    has_pattern = "101" in compressed

    # 详细反馈
    info = f"范围[{min_val:.3f}, {max_val:.3f}] -> 趋势[{compressed}]"

    # 返回: (Is_Valid, Trend_String, Detail_Info)
    return has_pattern, compressed, info


def main():
    pkl_files = glob.glob(os.path.join(DATA_DIR, "*.pkl"))
    pkl_files.sort()

    print(f"找到 {len(pkl_files)} 个 pkl 文件")
    print(f"{'文件名':<40} | {'结果':<8} | {'详情'}")
    print("-" * 80)

    valid_count = 0
    invalid_count = 0

    # 用于统计各种趋势的数量
    trend_counter = Counter()
    # <--- 新增：用于存储每种趋势对应的文件名列表
    trend_files = defaultdict(list)

    for pkl_path in pkl_files:
        filename = os.path.basename(pkl_path)
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)

            # 提取轨迹
            gripper_traj = []
            if isinstance(data, list):
                for step in data:
                    # 兼容 'action' 和 'actions'
                    action = step.get("action", step.get("actions"))

                    if action is not None and len(action) > GRIPPER_DIM_INDEX:
                        gripper_traj.append(action[GRIPPER_DIM_INDEX])

            gripper_traj = np.array(gripper_traj)

            if len(gripper_traj) == 0:
                print(f"{filename:<40} | ERROR    | 没找到 action 数据")
                trend_str = "ErrorData"  # 标记错误类型
                trend_counter[trend_str] += 1
                trend_files[trend_str].append(filename)  # <--- 记录错误文件
                continue

            # 检查趋势
            is_valid, trend_str, reason = check_gripper_trend(gripper_traj, filename)

            # 记录趋势类型和对应的文件名
            trend_counter[trend_str] += 1
            trend_files[trend_str].append(filename)  # <--- 记录文件名

            status = "✅ OK" if is_valid else "❌ FAIL"
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1

            # 这里可以选择是否打印每一行，或者只打印错误的
            # print(f"{filename:<40} | {status:<8} | {reason}")

        except Exception as e:
            print(f"{filename:<40} | ERROR    | 读取失败: {e}")
            trend_counter["ReadError"] += 1
            trend_files["ReadError"].append(filename)

    print("-" * 80)
    print(f"基础统计: 合格 {valid_count}, 不合格 {invalid_count}")

    # 打印详细的趋势统计
    print("\n" + "=" * 30 + " 趋势类型分布统计 " + "=" * 30)

    # 定义不需要打印详情的“白名单”趋势
    whitelist_trends = ["101", "10101"]

    if trend_counter:
        for trend, count in trend_counter.most_common():
            desc = ""
            if trend == "101":
                desc = "(完美: 开->合->开)"
            elif trend == "10":
                desc = "(只抓未放: 开->合)"
            elif trend == "1":
                desc = "(全程张开)"
            elif trend == "0":
                desc = "(全程闭合)"
            elif trend == "01":
                desc = "(开始闭合: 合->开)"
            elif trend == "NoTrigger":
                desc = "(数值未触发阈值)"

            print(f"趋势 [{trend:<10}]: {count:>3} 个 {desc}")

            # <--- 核心修改：如果趋势不在白名单中，打印对应的文件列表
            if trend not in whitelist_trends:
                file_list = trend_files[trend]
                # 简单格式化一下，每行打印几个，或者直接打印整个列表
                print(f"    ⚠️  涉及文件 ({len(file_list)}个):")
                # 将文件名列表转换为字符串打印，看起来更整洁
                print(f"       {file_list}")
                print("-" * 40)  # 分隔线

    else:
        print("没有统计到任何趋势数据")
    print("=" * 78)


if __name__ == "__main__":
    main()
