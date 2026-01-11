import os
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================
# 数据路径 (请确认这是你刚刚处理完 Action 的那个路径)
DATA_DIR = "/work/wmx/openpi/data_1213/merged_all_episodes"

# 是否备份 (建议 True)
BACKUP_ORIGINAL = True
# ===============================================

def process_episode(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # 遍历每一帧
        for i in range(len(data)):
            obs = data[i].get('observations', {})
            orin_state = obs.get('orin_state', {})
            
            # 1. 获取 TCP Pose (7维)
            if 'tcp_pose' not in orin_state:
                print(f"⚠️ {os.path.basename(file_path)} 第 {i} 帧缺少 tcp_pose")
                return False
            tcp_pose = orin_state['tcp_pose']
            
            # 2. 获取 Gripper Pose (1维)
            gripper = orin_state.get('gripper_pose', 0.0)
            # 处理标量转数组
            if np.isscalar(gripper):
                gripper = np.array([gripper], dtype=np.float32)
            else:
                gripper = np.array(gripper, dtype=np.float32).reshape(1)
            
            # 3. 拼接生成 State (8维)
            # [x, y, z, qx, qy, qz, qw, g]
            ee_state = np.concatenate([tcp_pose, gripper]).astype(np.float32)
            
            # 4. 写入到 observations['state']
            # 注意：OpenPI 默认很多 config 用 observation.state，我们这里显式创建一个 key
            data[i]['observations']['state'] = ee_state

        # 5. 保存
        if BACKUP_ORIGINAL:
            backup_path = file_path + ".bak_state"
            if not os.path.exists(backup_path):
                os.rename(file_path, backup_path)
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
            
        return True

    except Exception as e:
        print(f"❌ 处理出错 {os.path.basename(file_path)}: {e}")
        return False

def main():
    pkl_files = glob(os.path.join(DATA_DIR, "*.pkl"))
    pkl_files.sort()
    
    print(f"📂 目标路径: {DATA_DIR}")
    print(f"📄 文件数量: {len(pkl_files)}")
    print(f"🔧 任务: 构造 observations['state'] = tcp_pose(7) + gripper(1)")
    
    # 过滤掉备份文件
    pkl_files = [f for f in pkl_files if not f.endswith('.bak') and not f.endswith('.bak_state')]
    
    input(f"即将处理 {len(pkl_files)} 个文件，按 Enter 开始...")
    
    success_count = 0
    for pkl_path in tqdm(pkl_files, desc="Adding EE State"):
        if process_episode(pkl_path):
            success_count += 1
            
    print("\n" + "="*30)
    print(f"✅ 完成! 成功修改: {success_count} / {len(pkl_files)}")

if __name__ == "__main__":
    main()