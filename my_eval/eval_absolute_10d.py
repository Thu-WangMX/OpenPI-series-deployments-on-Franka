import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import glob
import shutil  # 新增：导入shutil用于删除目录/文件
from tqdm import tqdm
from openpi.training import config as _config
from openpi.policies import policy_config

# ================= ⚙️ 配置区域 =================
# 1. 数据目录 (注意：这里填目录路径)
DATA_DIR = "/work/wmx/openpi/dataset_1217/data_red_125"

# 2. 模型设置
CHECKPOINT_DIR = "/work/wmx/openpi/ckpt_torch/pi0_abs_6drot_red_300_bs192_6k"
CONFIG_NAME = "pi0_franka_low_mem_finetune"

# 3. 评估设置
MAX_EPISODES = 10    # 评估多少个文件？(None 代表全部, 整数代表数量)
STRIDE = 1          # 采样步长
ACTION_DIM = 10     # 3 Pos + 6 Rot + 1 Gripper
OUTPUT_DIR = "eval_results_10d_red_fixed_50"  # 图片保存目录

# ================= 辅助函数 =================
def plot_10d_trajectory(gt_data, model_data, time_steps, save_path, episode_name):
    """绘制单条轨迹对比图"""
    total_mse = np.mean((gt_data - model_data)**2)
    
    dim_names = [
        "Pos X", "Pos Y", "Pos Z",           
        "Rot6D_1 (xx)", "Rot6D_2 (xy)", "Rot6D_3 (xz)", 
        "Rot6D_4 (yx)", "Rot6D_5 (yy)", "Rot6D_6 (yz)", 
        "Gripper"                            
    ]

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle(f"Episode: {episode_name} | Total MSE: {total_mse:.6f}", fontsize=16)
    
    axes = axes.flatten()
    for dim in range(ACTION_DIM):
        ax = axes[dim]
        ax.plot(time_steps, gt_data[:, dim], label='GT', color='#2ca02c', linewidth=2, alpha=0.8)
        ax.plot(time_steps, model_data[:, dim], label='Pred', color='#1f77b4', linewidth=1.5, linestyle='--')
        
        dim_mse = np.mean((gt_data[:, dim] - model_data[:, dim])**2)
        name = dim_names[dim] if dim < len(dim_names) else f"Dim {dim}"
        ax.set_title(f"{name}\nMSE: {dim_mse:.5f}", fontsize=10)
        ax.grid(True, alpha=0.3)
        if dim == 0: ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig) # 关闭图像释放内存
    return total_mse

# ================= 主流程 =================
def main():
    # 0. 准备工作：先清空输出目录，再重建
    if os.path.exists(OUTPUT_DIR):
        # 遍历目录内所有内容并删除
        for item in os.listdir(OUTPUT_DIR):
            item_path = os.path.join(OUTPUT_DIR, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)  # 删除文件/符号链接
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)  # 删除子目录
                print(f"🗑️ 删除旧文件/目录: {item_path}")
            except Exception as e:
                print(f"⚠️ 删除 {item_path} 失败: {e}")
    else:
        # 目录不存在则创建
        os.makedirs(OUTPUT_DIR)
        print(f"📁 创建新目录: {OUTPUT_DIR}")
    
    # 确保目录存在（防止删除后意外丢失）
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 扫描文件
    pkl_files = sorted(glob.glob(os.path.join(DATA_DIR, "episode_*.pkl")))
    if not pkl_files:
        print(f"❌ 未在 {DATA_DIR} 找到 .pkl 文件")
        return
        
    # 截取指定数量
    if MAX_EPISODES is not None:
        pkl_files = pkl_files[:MAX_EPISODES]
    
    print(f"📂 发现文件总数: {len(pkl_files)} (将评估前 {len(pkl_files)} 个)")

    # 2. 加载模型 (只加载一次)
    print(f"🔄 正在加载模型: {CHECKPOINT_DIR}")
    try:
        config = _config.get_config(CONFIG_NAME)
    except KeyError:
        print(f"⚠️  Config '{CONFIG_NAME}' 未找到。")
        return

    policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR)
    print("✅ 模型加载完成")

    # 3. 批量循环
    all_mses = []
    
    # 外层进度条：遍历文件
    for pkl_path in tqdm(pkl_files, desc="Batch Eval"):
        episode_name = os.path.splitext(os.path.basename(pkl_path))[0]
        
        # --- A. 读取数据 ---
        with open(pkl_path, 'rb') as f:
            episode = pickle.load(f)
        
        total_frames = len(episode)
        frames_indices = range(0, total_frames, STRIDE)
        
        gt_traj = []
        model_traj = []

        # --- B. 单个文件推理 ---
        # 内层循环：遍历帧 (不显示进度条以免刷屏，或者用 leave=False)
        for t in frames_indices:
            sample = episode[t]
            obs = sample['observations']
            
            example_input = {
                "observation/image": obs['pixels']['image'],
                "observation/wrist_image": obs['pixels']['image2'],
                "observation/state": obs['state'],
                "prompt": obs['task_description']
            }

            # GT
            gt_traj.append(sample['action'])

            # Pred
            with torch.no_grad():
                result = policy.infer(example_input)
            
            action_chunk = result["actions"]
            if hasattr(action_chunk, 'cpu'):
                action_chunk = action_chunk.cpu().numpy()
            model_traj.append(action_chunk[0])

        gt_traj = np.array(gt_traj)
        model_traj = np.array(model_traj)

        # --- C. 绘图与记录 ---
        save_path = os.path.join(OUTPUT_DIR, f"eval_{episode_name}.png")
        mse = plot_10d_trajectory(gt_traj, model_traj, frames_indices, save_path, episode_name)
        
        all_mses.append(mse)
        # print(f"   Saved: {save_path} | MSE: {mse:.6f}")

    # 4. 最终总结
    avg_mse = np.mean(all_mses)
    print("\n" + "="*40)
    print(f"🎉 批量评估完成！")
    print(f"📊 评估文件数: {len(pkl_files)}")
    print(f"📉 平均 MSE: {avg_mse:.6f}")
    print(f"📂 结果保存在: {OUTPUT_DIR}")
    print("="*40)

if __name__ == "__main__":
    main()