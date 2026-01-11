import numpy as np
from openpi.training import config as _config
from openpi.policies import policy_config

# 1. 加载配置
print("🔄 正在加载配置...")
config = _config.get_config("pi05_franka_low_mem_finetune")
checkpoint_dir = "/work/wmx/openpi/ckpt_torch/pi05_delta_joint_bs32_3w"
print(f"🧐 当前配置的动作维度: {config.model.action_dim}")
# 2. 加载策略 (自动检测 PyTorch 格式)
print(f"🔄 正在加载模型: {checkpoint_dir}")
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# 3. 【关键】构造示例数据 (Example)
# 这里必须模拟真实机器人的输入格式
print("🛠️ 构造虚拟输入数据...")
example = {
    # 图像：必须是 (H, W, 3) 的 uint8 数组
    # OpenPi 默认需要 224x224，但 policy 内部通常有 Resize 操作
    # 这里的 Key (observation/image) 必须和你训练配置里的 repack_transforms 匹配
    "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
    
    # 状态：通常是关节角度 (7自由度 + 1夹爪 = 8维)
    "observation/state": np.zeros(8, dtype=np.float32),
    
    # 文本指令
    "prompt": "Put the red chili peppers into the basket"
}

# 4. 运行推理
print("🚀 开始推理...")
result = policy.infer(example)
print("result:" , result)
action_chunk = result["actions"]

print("✅ 推理成功！")
print(f"输出动作形状: {action_chunk.shape}")
print(f"前5步动作:\n{action_chunk[:5]}")