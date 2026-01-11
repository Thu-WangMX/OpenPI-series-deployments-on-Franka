import os
import pickle
import numpy as np
from tqdm import tqdm

def process_pkl_files(directory):
    # 获取所有pkl文件
    files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    print(f"找到 {len(files)} 个文件，准备处理...")

    for filename in tqdm(files, desc="Processing"):
        filepath = os.path.join(directory, filename)
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            if not isinstance(data, list) or len(data) == 0:
                print(f"Skipping {filename}: Not a valid list or empty.")
                continue

            num_frames = len(data)

            # -------------------------------------------------
            # 第一步：遍历每一帧，构造 observations['state']
            # -------------------------------------------------
            for i in range(num_frames):
                # 1. 处理 Current Observations
                obs = data[i].get('observations', {})
                if 'orin_state' in obs:
                    orin = obs['orin_state']
                    tcp_pose = orin.get('tcp_pose')
                    gripper_pose = orin.get('gripper_pose')
                    
                    if tcp_pose is not None and gripper_pose is not None:
                        # 确保转换为 numpy 且维度正确
                        tcp_arr = np.array(tcp_pose, dtype=np.float32).flatten()
                        gripper_arr = np.array(gripper_pose, dtype=np.float32).flatten() # 兼容标量或数组
                        
                        # 拼接 (7,) + (1,) -> (8,)
                        new_state = np.concatenate([tcp_arr, gripper_arr])
                        data[i]['observations']['state'] = new_state

                # 2. 处理 Next Observations (同理)
                next_obs = data[i].get('next_observations', {})
                if 'orin_state' in next_obs:
                    orin = next_obs['orin_state']
                    tcp_pose = orin.get('tcp_pose')
                    gripper_pose = orin.get('gripper_pose')
                    
                    if tcp_pose is not None and gripper_pose is not None:
                        tcp_arr = np.array(tcp_pose, dtype=np.float32).flatten()
                        gripper_arr = np.array(gripper_pose, dtype=np.float32).flatten()
                        
                        new_state = np.concatenate([tcp_arr, gripper_arr])
                        data[i]['next_observations']['state'] = new_state

            # -------------------------------------------------
            # 第二步：修正 Action 的第7维 (Index 6)
            # -------------------------------------------------
            for i in range(num_frames):
                # 获取当前的 action (引用)
                current_action = data[i]['action']
                
                # 确保 action 是 numpy 数组且至少有7维
                if not isinstance(current_action, np.ndarray) or current_action.shape[0] < 7:
                    continue

                if i < num_frames - 1:
                    # === 情况 A: 第 t 帧 (非最后一帧) ===
                    # 逻辑: 用 t+1 帧的 gripper_pose 赋值给当前 action 的第7维
                    # 注意: data[i]['next_observations'] 实际上就是 data[i+1]['observations'] (在标准MDP中)
                    # 这里直接取本条数据的 next_observations 里的 gripper_pose 最方便
                    
                    next_gripper_val = data[i]['next_observations']['orin_state']['gripper_pose']
                    
                    # 提取数值 (处理可能是 array 或 tensor 的情况)
                    if isinstance(next_gripper_val, np.ndarray):
                        val = next_gripper_val.item()
                    else:
                        val = float(next_gripper_val)
                    
                    current_action[6] = val
                    
                else:
                    # === 情况 B: 最后一帧 ===
                    # 逻辑: 跟倒数第二帧保持一致
                    if num_frames > 1:
                        prev_action_val = data[i-1]['action'][6]
                        current_action[6] = prev_action_val
                    else:
                        # 极端情况：只有一帧的数据，没法取上一帧，保持原样或设为当前gripper
                        pass 

                # 将修改后的 action 写回字典 (如果是引用修改通常不需要这步，但为了保险)
                data[i]['action'] = current_action

            # -------------------------------------------------
            # 第三步：覆盖保存
            # -------------------------------------------------
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("所有文件处理完成。")

if __name__ == "__main__":
    target_dir = "/work/wmx/openpi/data_1213/merged_all_episodes_1215"
    if os.path.exists(target_dir):
        process_pkl_files(target_dir)
    else:
        print(f"目录不存在: {target_dir}")