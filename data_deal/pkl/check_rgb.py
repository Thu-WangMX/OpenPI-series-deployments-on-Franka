# 检测pkl中的img是rgb还是bgr
import pickle

import cv2
import numpy as np

# 指向你的任意一个 pkl 文件
PKL_PATH = "/work/wmx/openpi/data_clean/single_grasp/episode_0.pkl"


def check_color():
    print(f"正在读取: {PKL_PATH}")
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)

    # 获取第一帧图像
    img = np.array(data[0]["observations"]["pixels"]["image"])

    print(f"图像形状: {img.shape}")

    # 测试 A：假设它是 BGR (直接保存)
    cv2.imwrite("check_pkl_as_is.jpg", img)

    # 测试 B：假设它是 RGB (转一下 BGR 再保存)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("check_pkl_converted.jpg", img_bgr)

    print("✅ 已保存 check_pkl_as_is.jpg 和 check_pkl_converted.jpg")
    print("请查看图片：")
    print("1. 如果 'as_is' 颜色正常 -> 你的 pkl 是 BGR 格式 (需要转换！)")
    print("2. 如果 'converted' 颜色正常 -> 你的 pkl 是 RGB 格式 (无需修改)")


if __name__ == "__main__":
    check_color()
