import io
from pathlib import Path

import cv2
from datasets import load_dataset
import numpy as np
from PIL import Image
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================
# 1. 数据集路径 (你的转换后数据路径)
DATASET_PATH = Path("/work/wzh/huggingface/lerobot/wmx/openpi_red_1223_274_clean")

# 2. 目标 Episode (可以改，比如 0, 1, 2...)
TARGET_EPISODE_INDEX = 34

# 3. 输出视频文件名
OUTPUT_VIDEO = f"vis_episode_{TARGET_EPISODE_INDEX}_rot6d_wrench.mp4"

# 4. FPS (建议设置为 30，与你之前的设置一致)
FPS = 30

# 5. 仪表盘高度 (为了容纳更多数据，稍微调高)
INFO_PANEL_HEIGHT = 320
# ===========================================


def decode_image(img_entry):
    """解码 LeRobot 图像 (Bytes/Numpy/PIL -> BGR Numpy)"""
    try:
        if img_entry is None:
            return None
        if isinstance(img_entry, dict) and "bytes" in img_entry:
            image = Image.open(io.BytesIO(img_entry["bytes"]))
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if isinstance(img_entry, np.ndarray):
            return cv2.cvtColor(img_entry, cv2.COLOR_RGB2BGR)
        if isinstance(img_entry, Image.Image):
            return cv2.cvtColor(np.array(img_entry), cv2.COLOR_RGB2BGR)
        return None
    except Exception as e:
        print(f"解码失败: {e}")
        return None


def draw_info_panel(canvas, start_y, width, frame_idx, total_frames, action, state, wrench, effort, task_desc=""):
    """
    绘制详细数据面板，适配 10维 State/Action 和 7维 Effort
    """
    # 字体设置
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    line_spacing = 22
    x_offset = 15
    y = start_y + 20

    # 颜色定义 (BGR)
    C_WHITE = (255, 255, 255)
    C_GREEN = (0, 255, 0)  # Action (Next Step)
    C_CYAN = (255, 255, 0)  # State (Current)
    C_RED = (80, 80, 255)  # Wrench (Force)
    C_ORANGE = (0, 165, 255)  # Effort
    C_GRAY = (180, 180, 180)

    # --- 1. 标题 ---
    header = f"EP: {TARGET_EPISODE_INDEX} | Frame: {frame_idx}/{total_frames} | {task_desc[:40]}"
    cv2.putText(canvas, header, (x_offset, int(y)), font, 0.5, C_WHITE, 1)
    y += line_spacing * 1.5

    # --- 2. State (10维: Pos3 + Rot6D + Grip1) ---
    # 你的数据结构: [x,y,z, r1,r2,r3,r4,r5,r6, g]
    if state is not None and len(state) == 10:
        pos = state[:3]
        rot6d = state[3:9]
        grip = state[9]

        str_pos = f"[State] Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] Grip: {grip:.3f}"
        cv2.putText(canvas, str_pos, (x_offset, int(y)), font, font_scale, C_CYAN, thickness)
        y += line_spacing

        str_rot = f"        R6D: [{rot6d[0]:.2f} {rot6d[1]:.2f} {rot6d[2]:.2f} | {rot6d[3]:.2f} {rot6d[4]:.2f} {rot6d[5]:.2f}]"
        cv2.putText(canvas, str_rot, (x_offset, int(y)), font, font_scale, C_CYAN, thickness)
    else:
        cv2.putText(
            canvas,
            f"[State] Dim Error: {len(state) if state is not None else 'None'}",
            (x_offset, int(y)),
            font,
            font_scale,
            C_CYAN,
            thickness,
        )

    y += line_spacing * 1.2

    # --- 3. Action (10维: Next Pos3 + Next Rot6D + Next Grip1) ---
    if action is not None and len(action) == 10:
        pos = action[:3]
        rot6d = action[3:9]
        grip = action[9]

        str_act = f"[Act]   Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] Grip: {grip:.3f}"
        cv2.putText(canvas, str_act, (x_offset, int(y)), font, font_scale, C_GREEN, thickness)
        y += line_spacing

        str_rot = f"        R6D: [{rot6d[0]:.2f} {rot6d[1]:.2f} {rot6d[2]:.2f} | {rot6d[3]:.2f} {rot6d[4]:.2f} {rot6d[5]:.2f}]"
        cv2.putText(canvas, str_rot, (x_offset, int(y)), font, font_scale, C_GREEN, thickness)
    else:
        cv2.putText(canvas, "[Act]   N/A", (x_offset, int(y)), font, font_scale, C_GREEN, thickness)

    y += line_spacing * 1.2

    # --- 4. Wrench (6维: Force3 + Torque3) ---
    if wrench is not None and len(wrench) == 6:
        force = wrench[:3]
        torque = wrench[3:]
        str_wrench = f"[Wrench] F: [{force[0]:.1f}, {force[1]:.1f}, {force[2]:.1f}] T: [{torque[0]:.2f}, {torque[1]:.2f}, {torque[2]:.2f}]"
        cv2.putText(canvas, str_wrench, (x_offset, int(y)), font, font_scale, C_RED, thickness)

    y += line_spacing * 1.2

    # --- 5. Effort (7维) ---
    if effort is not None and len(effort) >= 7:
        # 显示前4个和后3个，避免太长
        e = effort
        str_eff = f"[Effort] [{e[0]:.1f}, {e[1]:.1f}, {e[2]:.1f}, {e[3]:.1f}, {e[4]:.1f}, {e[5]:.1f}, {e[6]:.1f}]"
        cv2.putText(canvas, str_eff, (x_offset, int(y)), font, font_scale, C_ORANGE, thickness)


def main():
    print(f"📂 加载数据集: {DATASET_PATH}")

    # 1. 加载 Parquet 数据
    # LeRobot 生成的数据通常在 data/ 目录下
    data_files = str(DATASET_PATH / "data/**/*.parquet")
    try:
        ds = load_dataset("parquet", data_files=data_files, split="train")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print("尝试直接加载 dataset_dict...")
        ds = load_dataset(str(DATASET_PATH), split="train")

    # 2. 筛选指定 Episode
    print(f"🔍 正在筛选 Episode {TARGET_EPISODE_INDEX} ...")
    # 注意：LeRobot 的 key 通常是 'episode_index'
    episode_frames = ds.filter(lambda x: x["episode_index"] == TARGET_EPISODE_INDEX)

    total_frames = len(episode_frames)
    if total_frames == 0:
        print(f"❌ 未找到 Episode {TARGET_EPISODE_INDEX}！请检查索引是否存在。")
        # 打印前几个存在的索引供参考
        print(f"前5个可用索引: {ds[:5]['episode_index']}")
        return

    print(f"✅ 找到 {total_frames} 帧，准备渲染...")

    # 3. 准备 Canvas
    first_frame = episode_frames[0]
    img1 = decode_image(first_frame.get("observation.images.image"))
    img2 = decode_image(first_frame.get("observation.images.image2"))

    if img1 is None:
        print("❌ 无法读取第一帧图像")
        return

    h, w, _ = img1.shape

    # 缩放因子：保证文字清晰，如果图太小就放大
    SCALE_FACTOR = 2.0 if w < 320 else 1.5
    display_w = int(w * SCALE_FACTOR)
    display_h = int(h * SCALE_FACTOR)

    # 双目并排
    canvas_w = display_w * 2 if img2 is not None else display_w
    canvas_h = display_h + INFO_PANEL_HEIGHT

    print(f"📺 视频分辨率: {canvas_w}x{canvas_h} | FPS: {FPS}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (canvas_w, canvas_h))

    # 4. 循环生成
    # LeRobot 数据集 column 名称通常带有点号，例如 'observation.state'
    for i, item in enumerate(tqdm(episode_frames, desc="Processing")):
        # A. 图像解码
        im1 = decode_image(item.get("observation.images.image"))
        im2 = decode_image(item.get("observation.images.image2"))

        # 缩放
        im1 = cv2.resize(im1, (display_w, display_h), interpolation=cv2.INTER_NEAREST)
        if im2 is not None:
            im2 = cv2.resize(im2, (display_w, display_h), interpolation=cv2.INTER_NEAREST)

        # B. 数据提取
        # 注意：HuggingFace dataset 返回的是 list，需要转 numpy
        action = np.array(item.get("action", []))
        state = np.array(item.get("observation.state", []))
        wrench = np.array(item.get("observation.tcp_wrench", []))
        effort = np.array(item.get("observation.effort", []))

        task_desc = item.get("task_index", "")  # 或者根据 mapping 转换文字
        # 你的数据里似乎没有直接的 task 文本 column，只有 task_index
        # 如果之前写入了 'task' 或 'language_instruction' 也可以在这里取

        # C. 绘图
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # 贴图
        canvas[0:display_h, 0:display_w] = im1
        if im2 is not None:
            canvas[0:display_h, display_w : display_w * 2] = im2

        # 绘制数据
        draw_info_panel(
            canvas,
            start_y=display_h,
            width=canvas_w,
            frame_idx=i,
            total_frames=total_frames,
            action=action,
            state=state,
            wrench=wrench,
            effort=effort,
            task_desc=f"TaskIdx: {task_desc}",
        )

        out.write(canvas)

    out.release()
    print(f"\n🎉 视频已保存: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
