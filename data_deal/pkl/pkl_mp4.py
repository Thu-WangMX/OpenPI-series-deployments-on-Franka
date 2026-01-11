import pickle
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================
# 输入文件路径 (确保这是你刚刚处理过的含 Wrench/State 的 pkl)
PKL_PATH = "/work/wmx/dataset/dataset_1225/data_red_300/episode_88.pkl"

# 输出视频路径 (自动生成在同目录下)
OUTPUT_VIDEO_NAME = "vis_episo_pkl.mp4"
FPS = 30  # 播放速度

# 仪表盘高度 (用于显示大量文字)
INFO_PANEL_HEIGHT = 220 
# ===========================================

def draw_info_panel(canvas, start_y, width, frame_idx, total_frames, action, state, wrench, task_desc=""):
    """
    在底部绘制详细的数据仪表盘 (Force, Torque, State, Action)
    """
    # 字体设置
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    thickness = 1
    line_spacing = 25
    x_offset = 15
    y = start_y + 20

    # 颜色定义 (BGR)
    COLOR_WHITE = (255, 255, 255)
    COLOR_GREEN = (0, 255, 0)    # Action (Target)
    COLOR_CYAN  = (255, 255, 0)  # State (Current)
    COLOR_RED   = (0, 0, 255)    # Wrench (Force)
    COLOR_GRAY  = (180, 180, 180)

    # 1. 基础信息
    header = f"Frame: {frame_idx}/{total_frames}"
    cv2.putText(canvas, header, (x_offset, int(y)), font, 0.6, COLOR_WHITE, 2)
    y += line_spacing * 1.2

    # 2. State (8维: 7 Pose + 1 Gripper)
    if state is not None and len(state) >= 8:
        # 第一行: TCP Pos
        state_str1 = f"State(Pos): [{state[0]:.3f}, {state[1]:.3f}, {state[2]:.3f}]"
        cv2.putText(canvas, state_str1, (x_offset, int(y)), font, font_scale, COLOR_CYAN, thickness)
        
        # 第二行: TCP Rot + Gripper
        state_str2 = f"     (Rot): [{state[3]:.2f}, {state[4]:.2f}, {state[5]:.2f}, {state[6]:.2f}] G:{state[-1]:.3f}"
        cv2.putText(canvas, state_str2, (x_offset + 200, int(y)), font, font_scale, COLOR_CYAN, thickness)
    else:
        cv2.putText(canvas, "State: N/A", (x_offset, int(y)), font, font_scale, COLOR_CYAN, thickness)
    y += line_spacing

    # 3. Action (8维: Next Pose + Gripper)
    if action is not None and len(action) >= 8:
        act_str1 = f"Act  (Pos): [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}]"
        cv2.putText(canvas, act_str1, (x_offset, int(y)), font, font_scale, COLOR_GREEN, thickness)
        
        act_str2 = f"     (Rot): [{action[3]:.2f}, {action[4]:.2f}, {action[5]:.2f}, {action[6]:.2f}] G:{action[-1]:.3f}"
        cv2.putText(canvas, act_str2, (x_offset + 200, int(y)), font, font_scale, COLOR_GREEN, thickness)
    else:
        cv2.putText(canvas, "Action: N/A", (x_offset, int(y)), font, font_scale, COLOR_GREEN, thickness)
    y += line_spacing

    # 4. Wrench (6维: 3 Force + 3 Torque)
    if wrench is not None and len(wrench) >= 6:
        # Force
        f_str = f"Force (N): [{wrench[0]:.1f}, {wrench[1]:.1f}, {wrench[2]:.1f}]"
        cv2.putText(canvas, f_str, (x_offset, int(y)), font, font_scale, COLOR_RED, thickness)
        
        # Torque
        t_str = f"Torque(Nm): [{wrench[3]:.2f}, {wrench[4]:.2f}, {wrench[5]:.2f}]"
        cv2.putText(canvas, t_str, (x_offset + 220, int(y)), font, font_scale, COLOR_RED, thickness)
    else:
        cv2.putText(canvas, "Wrench: N/A", (x_offset, int(y)), font, font_scale, COLOR_RED, thickness)
    y += line_spacing * 1.5

    # 5. 任务描述
    if task_desc:
        cv2.putText(canvas, f"Task: {task_desc}", (x_offset, int(y)), font, 0.4, COLOR_GRAY, 1)

def main():
    if not os.path.exists(PKL_PATH):
        print(f"❌ 找不到文件: {PKL_PATH}")
        return

    print(f"📂 正在读取: {PKL_PATH}")
    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)

    if len(data) == 0:
        print("数据为空！")
        return

    total_frames = len(data)

    # --- 1. 获取视频尺寸信息 ---
    first_frame = data[0]['observations']
    img1 = first_frame['pixels']['image'] # RGB
    img2 = first_frame['pixels']['image2'] # RGB
    
    # 转换为 BGR
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 画面逻辑：双摄左右拼接，如果太小则放大
    # 如果宽度小于 200，说明是小图 (128x128)，放大 2 倍方便看清文字
    SCALE_FACTOR = 2 if w1 < 200 else 1
    
    display_w = w1 * SCALE_FACTOR
    display_h = h1 * SCALE_FACTOR
    
    # 总画布尺寸
    canvas_w = display_w * 2
    canvas_h = display_h + INFO_PANEL_HEIGHT
    
    output_path = str(Path(PKL_PATH).parent / OUTPUT_VIDEO_NAME)
    
    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (canvas_w, canvas_h))

    print(f"🎥 开始生成视频: {output_path}")
    print(f"📺 画面分辨率: {canvas_w}x{canvas_h} (Scale: {SCALE_FACTOR}x)")

    # --- 2. 逐帧处理 ---
    for i, step in enumerate(tqdm(data, desc="Rendering")):
        obs = step['observations']
        
        # --- A. 图像处理 ---
        im1 = cv2.cvtColor(obs['pixels']['image'], cv2.COLOR_RGB2BGR)
        im2 = cv2.cvtColor(obs['pixels']['image2'], cv2.COLOR_RGB2BGR)
        
        # 放大
        if SCALE_FACTOR > 1:
            im1 = cv2.resize(im1, (display_w, display_h), interpolation=cv2.INTER_NEAREST)
            im2 = cv2.resize(im2, (display_w, display_h), interpolation=cv2.INTER_NEAREST)

        # --- B. 获取数值数据 ---
        # Action
        action = step.get('action')
        
        # State (优先用新生成的 state，否则回退 agent_pos)
        state = obs.get('state')
        if state is None:
            # 注意：agent_pos 可能是关节角，而 state 是 EE Pose，显示时要注意区分物理含义
            # 这里仅做数据展示
            state = obs.get('agent_pos')
            
        # Wrench (Force + Torque)
        wrench = obs.get('tcp_wrench')
        
        # Task
        task_desc = step.get('language_instruction', '')
        if not task_desc:
            task_desc = obs.get('task_description', '')

        # --- C. 绘制画布 ---
        # 1. 创建全黑背景
        frame = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # 2. 贴图
        frame[0:display_h, 0:display_w] = im1
        frame[0:display_h, display_w:display_w*2] = im2
        
        # 3. 绘制仪表盘
        draw_info_panel(
            frame, 
            start_y=display_h, 
            width=canvas_w,
            frame_idx=i, 
            total_frames=total_frames,
            action=action,
            state=state,
            wrench=wrench,
            task_desc=task_desc
        )

        # 写入视频
        out.write(frame)

    # 释放资源
    out.release()
    print("\n✅ 视频生成完毕！")
    print(f"👉 路径: {output_path}")

if __name__ == "__main__":
    main()