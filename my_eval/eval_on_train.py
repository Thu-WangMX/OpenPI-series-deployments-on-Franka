import os  # 用于自动提取文件名
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from openpi.policies import policy_config
from openpi.training import config as _config

# ================= 配置区域 =================
# 1. 数据与模型路径
PKL_FILE_PATH = "/work/wmx/dataset_1217/data_red_125/episode_55.pkl"
CHECKPOINT_DIR = "/work/wmx/openpi/ckpt_torch/pi0_red_125_absolute_rot6d"
CONFIG_NAME = "pi0_franka_low_mem_finetune"

# 2. 验证设置
STRIDE = 1  # 采样步长 (1代表每帧都测)
ACTION_DIM = 10  # 7 关节 + 1 夹爪


# ================= 主流程 =================
def main():
    # 1. 加载数据
    print(f"📂 正在加载数据: {PKL_FILE_PATH}")
    with open(PKL_FILE_PATH, "rb") as f:
        episode = pickle.load(f)

    total_frames = len(episode)
    print(f"   数据总帧数: {total_frames}")

    # 2. 加载模型
    print(f"🔄 正在加载模型: {CHECKPOINT_DIR}")
    try:
        config = _config.get_config(CONFIG_NAME)
    except KeyError:
        print(f"⚠️  Config '{CONFIG_NAME}' 未找到，请确保已正确定义或导入。")
        return

    policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR)
    print("   模型加载完成。")

    # 3. 全轨迹推理
    gt_trajectory = []
    model_trajectory = []

    # 生成需要验证的帧索引
    frames_indices = range(0, total_frames, STRIDE)

    print(f"🚀 开始全轨迹推理 (总计 {len(frames_indices)} 步)...")

    for t in tqdm(frames_indices):
        sample = episode[t]

        # --- 准备输入 ---
        obs = sample["observations"]
        example_input = {
            "observation/image": obs["pixels"]["image"],
            "observation/wrist_image": obs["pixels"]["image2"],
            "observation/state": obs["state"],
            "prompt": obs["task_description"],
        }

        # --- 获取 GT (绝对值) ---
        gt_action = sample["action"]
        gt_trajectory.append(gt_action)

        # --- 模型推理 ---
        # 自动完成: Normalize -> Model -> Unnormalize -> AbsoluteActions
        with torch.no_grad():
            result = policy.infer(example_input)

        # --- 取 Chunk 的第 1 帧 ---
        action_chunk = result["actions"]

        # 转 Numpy
        if hasattr(action_chunk, "cpu"):
            action_chunk = action_chunk.cpu().numpy()

        pred_action_t = action_chunk[0]  # 取当前时刻动作
        model_trajectory.append(pred_action_t)

    # 转换为数组
    gt_trajectory = np.array(gt_trajectory)
    model_trajectory = np.array(model_trajectory)

    # 4. 自动生成文件名
    base_name = os.path.splitext(os.path.basename(PKL_FILE_PATH))[0]
    save_path = f"pi0_full_episode_trajectory_{base_name}.png"

    # 5. 计算指标并绘图
    print(f"📊 正在生成对比图: {save_path}")
    plot_full_trajectory(gt_trajectory, model_trajectory, frames_indices, save_path)


def plot_full_trajectory(gt_data, model_data, time_steps, save_path):
    # --- 【关键】计算 MSE ---
    # 计算整体均方误差 (所有维度、所有步长的平均值)
    total_mse = np.mean((gt_data - model_data) ** 2)
    print("\n" + "=" * 40)
    print(f"📈 整体均方误差 (Total MSE): {total_mse:.6f}")
    print("=" * 40 + "\n")

    # 开始绘图
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    file_label = os.path.basename(save_path)
    fig.suptitle(f"Whole Episode Trajectory Comparison\nFile: {file_label} | Total MSE: {total_mse:.6f}", fontsize=16)

    axes = axes.flatten()

    for dim in range(ACTION_DIM):
        ax = axes[dim]

        # 绘制 GT 和 Model
        ax.plot(time_steps, gt_data[:, dim], label="Ground Truth", color="green", linewidth=2, alpha=0.7)
        ax.plot(time_steps, model_data[:, dim], label="Model Prediction", color="blue", linewidth=1.5, linestyle="--")

        # 单独计算当前关节的 MSE
        dim_mse = np.mean((gt_data[:, dim] - model_data[:, dim]) ** 2)

        # 设置标题
        dim_name = f"Joint {dim}" if dim < 7 else "Gripper"
        ax.set_title(f"{dim_name} (MSE: {dim_mse:.5f})")
        ax.grid(True, alpha=0.3)

        if dim >= 4:
            ax.set_xlabel("Frame Index")

        # 仅在第一个子图显示图例
        if dim == 0:
            ax.legend()
            # 在图内也写一下 Total MSE
            ax.text(
                0.05,
                0.9,
                f"Total MSE: {total_mse:.5f}",
                transform=ax.transAxes,
                color="red",
                fontsize=12,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.8),
            )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"✅ 图片已保存至: {save_path}")


if __name__ == "__main__":
    main()
