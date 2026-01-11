import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# ================= 配置区域 =================
REPO_ID = "wmx/openpi_merged_single_grasp"  # 你的数据集ID
NUM_EPISODES = 50          # 要合并的 Episode 数量
OUTPUT_FILENAME = "combined_50_episodes.mp4"
FPS = 30.0                 # 视频帧率 (建议设高一点，比如 30 或 60，否则 50 个 episode 会看很久)
# ===========================================

def get_image_key(item):
    """自动寻找主相机视角的 Key"""
    for k in item.keys():
        if "image" in k and "wrist" not in k:
            return k
    return "observation.images.image"

def main():
    print(f"🔄 正在加载数据集: {REPO_ID} ...")
    try:
        dataset = LeRobotDataset(repo_id=REPO_ID)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    print(f"🚀 准备合并前 {NUM_EPISODES} 个 Episode...")
    
    # 1. 初始化视频写入器 (只需一次)
    # 先读取第0帧来获取图像尺寸
    first_item = dataset[0]
    img_key = get_image_key(first_item)
    sample_img = first_item[img_key]
    H, W = sample_img.shape[1], sample_img.shape[2]
    
    print(f"📺 视频分辨率: {W}x{H}, 帧率: {FPS}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, FPS, (W, H))

    # 统计总帧数用于进度条
    total_frames = 0
    for i in range(min(NUM_EPISODES, dataset.num_episodes)):
        total_frames += (dataset.episode_data_index["to"][i] - dataset.episode_data_index["from"][i]).item()

    print(f"🎞️ 预计总帧数: {total_frames}")

    # 2. 遍历 Episode 并写入同一个视频文件
    pbar = tqdm(total=total_frames, unit="frame")
    
    for ep_idx in range(min(NUM_EPISODES, dataset.num_episodes)):
        # 获取当前 Episode 的起止帧
        from_idx = dataset.episode_data_index["from"][ep_idx].item()
        to_idx = dataset.episode_data_index["to"][ep_idx].item()
        
        # 遍历当前 Episode 的每一帧
        for i in range(from_idx, to_idx):
            item = dataset[i]
            
            # --- 处理图像 ---
            img_tensor = item[img_key]
            img_np = img_tensor.permute(1, 2, 0).numpy()
            if img_np.max() <= 1.05:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
            
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # --- 处理动作与叠加文字 ---
            action = item.get('action', item.get('actions'))
            if action is not None:
                act_np = action.float().numpy() if isinstance(action, torch.Tensor) else action
                z_val = act_np[2] # Z轴
                
                # 颜色：负数(红), 正数(绿)
                color = (0, 0, 255) if z_val < 0 else (0, 255, 0)
                
                # 在画面上叠加信息
                # 第一行：Z轴数值
                cv2.putText(img_bgr, f"Z: {z_val:.4f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                # 第二行：当前 Episode 和 帧号
                cv2.putText(img_bgr, f"Ep: {ep_idx} | Frame: {i}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 写入视频
            out.write(img_bgr)
            pbar.update(1)
        
        # (可选) 在 Episode 之间插入几帧黑屏或过渡，方便区分？
        # 这里为了保持连贯性，暂不插入，你可以看左上角的 Ep 编号变化。

    pbar.close()
    out.release()

    print("\n" + "="*50)
    print(f"✅ 合并视频已生成: {OUTPUT_FILENAME}")
    print("请下载并播放。你可以通过拖动进度条快速浏览这 50 个 Episode。")
    print("重点检查：是否每个 Episode 的抓取动作发生时，Z 值都是负数？")

if __name__ == "__main__":
    main()