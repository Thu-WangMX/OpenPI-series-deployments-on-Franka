"""
LeRobot 批量数据转换脚本 (v3: 适配 10维 Rot6D State/Action + Effort + Wrench)
对应数据集: /work/wmx/dataset_1217/data_used
"""
import pickle
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import shutil
import sys

# 尝试导入 LeRobotDataset
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ================= ⚙️ 配置区域 =================
# 1. 输入路径 (你刚刚批量处理后的文件夹)
RAW_DATA_DIR = Path("/work/wmx/dataset_1227_205")

# 2. 输出 Repo ID
REPO_ID = "wmx/openpi_red_1227_205_clean"

# 3. 其他参数
MIN_FRAMES = 15  
FPS = 30
ROBOT_TYPE = "FR3"  # 或 "Panda"
# ===============================================

def load_pkl(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def main():
    # --- 0. 强制清理旧的 HuggingFace 缓存 ---
    cache_dir = Path.home() / ".cache/huggingface/lerobot" / REPO_ID
    if cache_dir.exists():
        print(f"🧹 清理旧缓存目录: {cache_dir}")
        shutil.rmtree(cache_dir)
    
    # 扫描 pkl 文件
    # 假设文件名格式为 episode_0.pkl, episode_1.pkl ... 按数字排序
    pkl_files = sorted(list(RAW_DATA_DIR.glob("*.pkl")), key=lambda x: int(x.stem.split('_')[-1]) if '_' in x.stem else 0)
    
    if not pkl_files:
        print(f"❌ 错误: 在 {RAW_DATA_DIR} 未找到 .pkl 文件")
        return
    
    print(f"📂 找到 {len(pkl_files)} 个 episodes.")

    # --- 1. 探测特征 (Probe Features) ---
    print("🔍 正在探测数据特征维度...")
    try:
        sample_data = load_pkl(pkl_files[0])
        first_frame = sample_data[0]
        obs_sample = first_frame['observations']

        # [探测] 图像尺寸
        img1 = np.array(obs_sample['pixels']['image'])
        img2 = np.array(obs_sample['pixels']['image2'])
        h1, w1, c1 = img1.shape
        h2, w2, c2 = img2.shape
        
        # [探测] Action 维度 (应该是 10: Pos3 + Rot6D + Grip1)
        act_dim = first_frame['action'].shape[0]
        
        # [探测] State 维度 (应该是 10)
        state_dim = obs_sample['state'].shape[0]
        
        # [探测] Wrench 维度 (应该是 6)
        wrench_dim = obs_sample['tcp_wrench'].shape[0]
            
        # [探测] Effort 维度 (应该是 7, 对应之前的 tau_J)
        effort_dim = obs_sample['effort'].shape[0]

        print("="*40)
        print(f"✅ 特征探测结果:")
        print(f"   - Action Dim: {act_dim} (期望 10)")
        print(f"   - State  Dim: {state_dim} (期望 10)")
        print(f"   - Wrench Dim: {wrench_dim} (期望 6)")
        print(f"   - Effort Dim: {effort_dim} (期望 7)")
        print(f"   - Image Sizes: ({h1},{w1}) & ({h2},{w2})")
        print("="*40)
        
    except KeyError as e:
        print(f"❌ 探测失败，你的数据可能缺少键值: {e}")
        print("请确保你运行了之前的 batch_process_data.py 脚本！")
        return

    # --- 2. 定义 Feature Schema ---
    features = {
        # 动作 (Next State)
        "action": {
            "dtype": "float32",
            "shape": (act_dim,),
            "names": ["action"] * act_dim,
        },
        # 状态 (Current State: Pos + Rot6D + Gripper)
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"] * state_dim,
        },
        # 力/力矩 (Force + Torque)
        "observation.tcp_wrench": {
            "dtype": "float32",
            "shape": (wrench_dim,),
            "names": ["force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z"],
        },
        # 关节力矩 (Effort / Tau_J)
        "observation.effort": {
            "dtype": "float32",
            "shape": (effort_dim,),
            "names": [f"joint_{i}" for i in range(effort_dim)],
        },
        # 图像
        "observation.images.image": {
            "dtype": "image",
            "shape": (h1, w1, c1),
            "names": ["height", "width", "channel"],
        },
        "observation.images.image2": {
            "dtype": "image",
            "shape": (h2, w2, c2),
            "names": ["height", "width", "channel"],
        },
    }

    # --- 3. 创建数据集 ---
    print(f"📦 初始化 LeRobot 数据集: {REPO_ID}")
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        robot_type=ROBOT_TYPE,
        features=features,
    )

    # --- 4. 转换循环 ---
    count_success = 0

    for pkl_path in tqdm(pkl_files, desc="Converting"):
        try:
            episode_data = load_pkl(pkl_path)
            
            if len(episode_data) < MIN_FRAMES:
                continue

            # 获取任务描述
            task_desc = episode_data[0].get('language_instruction', '')
            if not task_desc:
                task_desc = episode_data[0].get('observations', {}).get('task_description', 'pick red chili pepper')

            for frame in episode_data:
                obs = frame['observations']
                
                # --- A. 提取并转为 Tensor ---
                # 注意：astype(np.float32) 很重要，否则 LeRobot 可能会报错
                
                # 1. Action (10维)
                action_tensor = torch.from_numpy(frame['action'].astype(np.float32))
                
                # 2. State (10维)
                state_tensor = torch.from_numpy(obs['state'].astype(np.float32))

                # 3. Wrench (6维)
                wrench_tensor = torch.from_numpy(obs['tcp_wrench'].astype(np.float32))

                # 4. Effort (7维)
                effort_tensor = torch.from_numpy(obs['effort'].astype(np.float32))

                # 5. Images
                img1_tensor = torch.from_numpy(np.array(obs['pixels']['image']))
                img2_tensor = torch.from_numpy(np.array(obs['pixels']['image2']))

                # --- B. 添加帧 ---
                dataset.add_frame({
                    "action": action_tensor,
                    "observation.state": state_tensor,
                    "observation.tcp_wrench": wrench_tensor,
                    "observation.effort": effort_tensor,
                    "observation.images.image": img1_tensor,
                    "observation.images.image2": img2_tensor,
                    "task": task_desc 
                })

            # 保存 Episode
            dataset.save_episode()
            count_success += 1

        except Exception as e:
            print(f"\n❌ [错误] 处理 {pkl_path.name} 失败: {e}")
            # 清理当前 buffer 防止污染下一个 episode
            if hasattr(dataset, 'clear_episode_buffer'):
                dataset.clear_episode_buffer()
            else:
                # 兼容旧版本 LeRobot
                dataset.episode_buffer = dataset.create_episode_buffer()
            continue

    # --- 5. 结束 ---
    print(f"\n💾 正在保存并最终化数据集...")
    if hasattr(dataset, 'finalize'):
        dataset.finalize()
    elif hasattr(dataset, 'consolidate'):
        dataset.consolidate()
    
    print("="*50)
    print(f"🎉 转换完成！")
    print(f"✅ 成功转换: {count_success} / {len(pkl_files)}")
    print(f"📂 数据集已保存至 HuggingFace Cache 或 本地路径")
    print("="*50)

if __name__ == "__main__":
    main()