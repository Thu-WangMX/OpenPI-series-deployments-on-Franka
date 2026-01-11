# 读取单个pkl的前若干帧数据
import pickle

import numpy as np

# ================= 配置 =================
FILE_PATH = "/work/wmx/dataset_1217/data_red_125/episode_0.pkl"
FRAMES_TO_READ = 10
# ========================================


def format_value(val, indent=0):
    """递归格式化输出，数组只显示 Shape，其他显示完整值"""
    prefix = " " * indent

    if isinstance(val, dict):
        res = []
        for k, v in val.items():
            res.append(f"{prefix}{k}: {format_value(v, indent + 2).strip()}")
        return "\n" + "\n".join(res)

    if isinstance(val, (list, tuple)):
        # 如果列表太长，也简化显示
        if len(val) > 20:
            return f"<list length={len(val)}, first_element={type(val[0])}>"
        return str(val)

    if isinstance(val, np.ndarray):
        # 针对数组：显示形状、类型，如果元素少于10个则显示具体数值
        if val.size < 20:
            return f"array(shape={val.shape}, dtype={val.dtype}, val={val.tolist()})"
        # 大数组（如图像）只显示形状
        return f"array(shape={val.shape}, dtype={val.dtype}) [LARGE DATA HIDDEN]"

    return str(val)


def main():
    print(f"📂 正在读取文件: {FILE_PATH} ...")

    try:
        with open(FILE_PATH, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, list):
            print(f"❌ 数据格式错误: 期望是 list，实际是 {type(data)}")
            return

        total_frames = len(data)
        print(f"✅ 读取成功! 总帧数: {total_frames}")

        count = min(total_frames, FRAMES_TO_READ)
        print(f"👇 下面是前 {count} 帧的详细数据:\n")

        for i in range(count):
            print("=" * 60)
            print(f"🎥 Frame {i}")
            print("=" * 60)
            print(format_value(data[i]))
            print("\n")

    except FileNotFoundError:
        print(f"❌ 文件未找到: {FILE_PATH}")
    except Exception as e:
        print(f"❌ 发生错误: {e}")


if __name__ == "__main__":
    main()
