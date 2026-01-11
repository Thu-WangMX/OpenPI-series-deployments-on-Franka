import os
import pickle
import numpy as np
import random

def verify_files(directory, num_samples=50):
    files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    
    if not files:
        print("❌ 目录下没有找到 .pkl 文件")
        return

    # 随机抽取几个文件进行检查
    sample_files = random.sample(files, min(num_samples, len(files)))
    
    print(f"🔍 将检查以下 {len(sample_files)} 个文件: {sample_files}\n")

    for filename in sample_files:
        filepath = os.path.join(directory, filename)
        print(f"------ 正在检查: {filename} ------")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        num_frames = len(data)
        errors = []

        for i in range(num_frames):
            obs = data[i].get('observations', {})
            orin = obs.get('orin_state', {})
            action = data[i].get('action')
            
            # === 验证 1: observation.state 的构造 ===
            if 'state' not in obs:
                errors.append(f"帧 {i}: 缺少 observations['state']")
            else:
                state = obs['state']
                tcp = np.array(orin['tcp_pose']).flatten()
                gripper = np.array(orin['gripper_pose']).flatten()
                expected_state = np.concatenate([tcp, gripper])
                
                if state.shape != (8,):
                     errors.append(f"帧 {i}: state 形状错误 {state.shape}，应为 (8,)")
                elif not np.allclose(state, expected_state, atol=1e-5):
                     errors.append(f"帧 {i}: state数值不匹配。\n实际: {state}\n期望: {expected_state}")

            # === 验证 2: Action 第7维的赋值逻辑 ===
            current_gripper_action = action[6]

            if i < num_frames - 1:
                # 检查非最后一帧：应等于下一帧的 gripper_pose
                next_obs_gripper = data[i]['next_observations']['orin_state']['gripper_pose']
                # 或者 data[i+1]['observations']['orin_state']['gripper_pose']
                
                if isinstance(next_obs_gripper, np.ndarray):
                    expected_val = next_obs_gripper.item()
                else:
                    expected_val = float(next_obs_gripper)

                if not np.isclose(current_gripper_action, expected_val, atol=1e-5):
                    errors.append(f"帧 {i} (Action): 动作第7维 ({current_gripper_action}) 不等于下一帧 gripper_pose ({expected_val})")
            
            else:
                # 检查最后一帧：应等于倒数第二帧的 Action
                if num_frames > 1:
                    prev_action_val = data[i-1]['action'][6]
                    if not np.isclose(current_gripper_action, prev_action_val, atol=1e-5):
                        errors.append(f"帧 {i} (Last Action): 动作第7维 ({current_gripper_action}) 不等于上一帧 Action ({prev_action_val})")

        # === 报告结果 ===
        if len(errors) == 0:
            print(f"✅ {filename}: 所有 {num_frames} 帧验证通过！")
        else:
            print(f"❌ {filename}: 发现错误！(显示前3个)")
            for e in errors[:3]:
                print(f"  - {e}")
            if len(errors) > 3:
                print(f"  ... 还有 {len(errors)-3} 个错误")
        print("\n")

if __name__ == "__main__":
    target_dir = "/work/wmx/openpi/data_1213/merged_all_episodes_1215"
    verify_files(target_dir)