import numpy as np
import torch
from collections import deque
from typing import Dict, Any, Optional, List

# OpenPi 依赖
from openpi.policies import policy_config
from openpi.training import config as _config
from openpi_client import image_tools

class OpenPiSFTWrapper:
    def __init__(
        self, 
        checkpoint_dir: str, 
        config_name: str, 
        action_dim: int = 7, 
        default_prompt: str = "Do the task."
    ):
        """
        全功能 OpenPi SFT 模型适配器 (兼容 LeRobot 接口)
        """
        print(f"[OpenPiWrapper] Initializing...")
        print(f"  - Config: {config_name}")
        print(f"  - Checkpoint: {checkpoint_dir}")
        
        # 1. 加载配置 & 策略
        self.cfg = _config.get_config(config_name)
        # create_trained_policy 会自动加载 assets/norm_stats 并处理归一化
        self.policy = policy_config.create_trained_policy(
            self.cfg, 
            checkpoint_dir, 
            default_prompt=default_prompt
        )
        
        # 2. 关键参数
        self.robot_action_dim = action_dim  # 实际物理维度 (7)
        self.chunk_size = self.cfg.model.action_horizon # 模型输出步数 (Horizon)
        
        # 3. 动作队列 (支持 Batch=1 的单环境)
        self.batch_size = 1
        self.action_queue = [deque() for _ in range(self.batch_size)]
        
        print(f"✓ OpenPi Policy Ready.")
        print(f"  - Chunk Size: {self.chunk_size}")
        print(f"  - Action Dim: {self.cfg.model.action_dim} -> Sliced to {self.robot_action_dim}")

    def reset(self):
        """重置策略状态和动作队列"""
        if hasattr(self.policy, 'reset'):
            self.policy.reset()
        
        if self.action_queue is not None:
            for q in self.action_queue:
                q.clear()
        print("[OpenPiWrapper] Reset done.")

    def _recursive_to_numpy(self, x):
        """安全转换 Tensor/List 为 Numpy (复刻同学代码功能)"""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        elif isinstance(x, dict):
            return {k: self._recursive_to_numpy(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [self._recursive_to_numpy(v) for v in x]
        return x

    def _process_observation(self, observation: Dict[str, Any], task: Optional[str] = None) -> Dict[str, Any]:
        """数据清洗与映射"""
        # 1. 确保全部是 Numpy
        obs = self._recursive_to_numpy(observation)
        processed_obs = {}
        
        # 2. 图像处理 (Resize 224 + Uint8)
        # 兼容 LeRobot 常见结构 pixels={'image':...} 或扁平结构
        pixels = obs.get('pixels', obs)
        
        def process_img(img):
            img = image_tools.convert_to_uint8(img)
            img = image_tools.resize_with_pad(img, 224, 224)
            return img

        if 'image' in pixels:
            processed_obs["observation/image"] = process_img(pixels['image'])
        if 'image2' in pixels: # Wrist
            processed_obs["observation/wrist_image"] = process_img(pixels['image2'])

        # 3. 状态处理
        if 'agent_pos' in obs:
            processed_obs["observation/state"] = obs['agent_pos']
        elif 'observation.state' in obs:
             processed_obs["observation/state"] = obs['observation.state']

        # 4. Prompt 处理 (优先用参数传入的 task，其次找 obs 里的)
        if task:
            processed_obs["prompt"] = task
        elif 'task_description' in obs:
            processed_obs["prompt"] = str(obs['task_description'])
        elif 'task' in obs:
            processed_obs["prompt"] = str(obs['task'])
            
        return processed_obs

    def get_action_chunk(self, observation: Dict[str, Any], task: Optional[str] = None) -> np.ndarray:
        """
        单次推理，返回完整 Chunk。
        """
        # 1. 预处理
        model_inputs = self._process_observation(observation, task)
        
        # 2. 增加 Batch 维度 (重要补丁!)
        # JAX 模型通常需要 (B, ...) 输入，而真机数据通常是 (H,W,C)
        batched_inputs = {}
        for k, v in model_inputs.items():
            if isinstance(v, np.ndarray):
                batched_inputs[k] = v[np.newaxis, ...] # Add Batch=1
            else:
                batched_inputs[k] = [v] # Strings list
        
        # 3. 推理 (JAX)
        result = self.policy.infer(batched_inputs)
        
        # 4. 提取结果 & 移除 Batch 维度
        # result['actions'] -> (1, Horizon, 32)
        actions = np.array(result['actions'])[0] # -> (Horizon, 32)
        
        # 5. 关键切片: 32维 -> 7维
        if actions.shape[-1] > self.robot_action_dim:
            actions = actions[:, :self.robot_action_dim]
            
        # 6. 为了保持接口一致，返回时再加回 Batch 维度 (同学的代码返回 [B, T, D])
        return actions[np.newaxis, ...]

    def __call__(self, observation: Dict[str, Any], tasks: list[str] = None, return_chunk=False):
        """策略执行入口"""
        # 兼容 tasks 列表输入
        task_str = tasks[0] if tasks and isinstance(tasks, list) else tasks

        # 1. 直接返回 Chunk 模式
        if return_chunk:
            return self.get_action_chunk(observation, task=task_str)

        # 2. 队列模式 (Action Chunking)
        # 初始化
        if self.action_queue is None:
            self.action_queue = [deque() for _ in range(self.batch_size)]

        # 检查队列
        if len(self.action_queue[0]) == 0:
            # 队列空，推理进货
            chunk = self.get_action_chunk(observation, task=task_str) # [1, T, D]
            chunk = chunk[0] # [T, D]
            
            for i in range(chunk.shape[0]):
                self.action_queue[0].append(chunk[i])
                
        # 出货
        action = self.action_queue[0].popleft()
        return action[np.newaxis, ...] # 返回 [1, D]


# 调用
# policy = OpenPiSFTWrapper(
#     checkpoint_dir="/work/wmx/openpi/checkpoints/pi0_franka_low_mem_finetune/franka_red_pepper_run2/30000",
#     config_name="pi0_franka_low_mem_finetune",
#     action_dim=7,  # 你的 Franka 是 7 维
#     default_prompt="Put the red chili peppers into the basket."
# )