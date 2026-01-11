import os
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================
# 数据路径 (请确认这是您当前的最新数据路径)
DATA_DIR = "/work/wmx/openpi/data_1213/merged_all_episodes_1215"

# 是否备份 (建议 True，防止写错)
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
            
            # 1. 获取 tcp_force (3维)
            if 'tcp_force' not in orin_state:
                print(f"⚠️ {os.path.basename(file_path)} 第 {i} 帧缺少 tcp_force")
                return False
            force = orin_state['tcp_force']
            
            # 2. 获取 tcp_torque (3维)
            if 'tcp_torque' not in orin_state:
                print(f"⚠️ {os.path.basename(file_path)} 第 {i} 帧缺少 tcp_torque")
                return False
            torque = orin_state['tcp_torque']
            
            # 3. 拼接生成 Wrench (6维)
            # [Fx, Fy, Fz, Tx, Ty, Tz]
            # 确保转为 float32 以节省空间并适配模型
            wrench = np.concatenate([force, torque]).astype(np.float32)
            
            # 4. 写入到 observations['tcp_wrench']
            # 这样在 config 中可以用 "observations.tcp_wrench" 读取
            data[i]['observations']['tcp_wrench'] = wrench

        # 5. 保存
        if BACKUP_ORIGINAL:
            # 避免覆盖之前的备份，可以换个后缀或者检查是否存在
            backup_path = file_path + ".bak_wrench"
            if not os.path.exists(backup_path):
                os.rename(file_path, backup_path)
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
            
        return True

    except Exception as e:
        print(f"❌ 处理出错 {os.path.basename(file_path)}: {e}")
        return False

def main():
    # 查找所有 pkl 文件
    pkl_files = glob(os.path.join(DATA_DIR, "*.pkl"))
    pkl_files.sort()
    
    print(f"📂 目标路径: {DATA_DIR}")
    print(f"📄 文件数量: {len(pkl_files)}")
    print(f"🔧 任务: 构造 observations['tcp_wrench'] = force(3) + torque(3)")
    
    # 过滤掉备份文件
    pkl_files = [f for f in pkl_files if not f.endswith('.bak') and not f.endswith('.bak_state') and not f.endswith('.bak_wrench')]
    
    input(f"即将处理 {len(pkl_files)} 个文件，按 Enter 开始...")
    
    success_count = 0
    for pkl_path in tqdm(pkl_files, desc="Adding Wrench"):
        if process_episode(pkl_path):
            success_count += 1
            
    print("\n" + "="*30)
    print(f"✅ 完成! 成功修改: {success_count} / {len(pkl_files)}")

if __name__ == "__main__":
    main()