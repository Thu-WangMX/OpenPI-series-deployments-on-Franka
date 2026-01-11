# import torch
# from torch import nn
# import numpy as np
# from vlaresidual.utils.train_utils import fold_time_dim, unfold_time_dim, decode_task_description_from_obs

# from lerobot.policies.factory import make_policy, make_pre_post_processors
# from lerobot.envs.utils import (
#     preprocess_observation,
# )


# class OpenPi(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.policy = make_policy(
#             cfg=cfg.policy,
#             env_cfg=cfg.env,
#             rename_map=cfg.rename_map,
#         )

#         # 创建输入/输出预处理器
#         preprocessor, postprocessor = make_pre_post_processors(
#             policy_cfg=cfg.policy,
#             pretrained_path=cfg.policy.pretrained_path,
#             preprocessor_overrides={
#                 "device_processor": {"device": str(self.policy.config.device)},
#                 "rename_observations_processor": {"rename_map": cfg.rename_map},
#             }
#         )

#         self.preprocessor = preprocessor
#         self.postprocessor = postprocessor

#         # Action queue 管理
#         self.action_queue = None
#         self.batch_size = None

#         # 获取设备
#         self.device = self.policy.config.device

#         print(f"✓ PI05Policy 初始化完成")
#         print(f"  - n_action_steps: {self.policy.config.n_action_steps}")
#         print(f"  - chunk_size: {self.policy.config.chunk_size}")

#     def _recursive_to_numpy(self, x):
#         """
#         安全地将 Tensor (GPU/CPU) 转为 NumPy (CPU)。
#         如果是 Dict，则递归转换。
#         """
#         if isinstance(x, torch.Tensor):
#             return x.detach().cpu().numpy()
#         elif isinstance(x, dict):
#             return {k: self._recursive_to_numpy(v) for k, v in x.items()}
#         elif isinstance(x, list):
#             return [self._recursive_to_numpy(v) for v in x]
#         return x

#     def _ensure_image_uint8(self, x):
#         """
#         递归地将图像数据强制转换为 uint8。
#         LeRobot 的 preprocess_observation 严格要求 uint8 输入。
#         """
#         if isinstance(x, dict):
#             return {k: self._ensure_image_uint8(v) for k, v in x.items()}

#         if isinstance(x, np.ndarray):
#             # 判定是否为图像：维度 >= 3 且 最后一维是 3
#             # 注意：这里的 x 可能是已经被折叠过的 (B*T, H, W, C)
#             if x.ndim >= 3 and x.shape[-1] == 3:
#                 # 如果是浮点数
#                 if x.dtype in [np.float32, np.float64]:
#                     # 检查是否被归一化到了 [0, 1]
#                     if x.max() <= 1.05: # 留点误差空间
#                         x = x * 255.0

#                     # 强制转为 uint8
#                     return x.astype(np.uint8)

#             # 也可以针对特定 key (如 'image') 做判断，但维度判断通常足够通用

#         return x

#     def _add_envs_task(self, observation, task):
#         """
#         将任务描述添加到观测中
#         参考lerobot中对pi调用的示例
#         """
#         if task is not None:
#             if isinstance(observation, dict):
#                 observation['task'] = task
#             else:
#                 raise ValueError("Observation should be a dict to add task description.")
#         return observation

#     def get_action_chunk(self, observation):
#         """
#         获取 action chunk，只返回前 n_action_steps 个
#         参考lerobot中对pi调用的示例
#         """
#         # 预处理观测

#         # 获取task
#         task = decode_task_description_from_obs(observation)

#         # 处理后的observation{'agent_pos', 'pixels'{'image', 'image2'}, 'task_description'}
#         # agent_pos: numpy.ndarray
#         # pixels: dict{ numpy.ndarray, dtype=uint8 }, [B, H, W, C]
#         # task_description: numpy.ndarray, dtype='<U82'
#         observation = self._recursive_to_numpy(observation)

#         # folded_obs{'agent_pos', 'pixels'{'image', 'image2'}, 'task_description'}, 只折叠了维度
#         folded_obs, original_shapes = fold_time_dim(observation)
#         b, t = original_shapes

#         # 确保图像是 uint8 格式
#         folded_obs = self._ensure_image_uint8(folded_obs)

#         # 此处调用lerobot的接口，处理后的observation不含task，后续需把task加进来
#         # processed_folded_obs{'observation.images.image', 'observation.images.image2', 'observation.state'}
#         # agent_pos -> observation.state, torch.Tensor
#         # image -> observation.images.image, torch.Tensor, 除255做归一化
#         # image2 -> observation.images.image2, torch.Tensor, 除255做归一化
#         processed_folded_obs = preprocess_observation(folded_obs)

#         observation = unfold_time_dim(processed_folded_obs, original_shapes)

#         # 处理后的observation{'observation.images.image', 'observation.images.image2', 'observation.state', 'task'}
#         observation = self._add_envs_task(observation, task)

#         batch = self.preprocessor(observation)

#         # 使用 predict_action_chunk 方法获取完整的 action chunk
#         with torch.inference_mode():
#             batch, _ = fold_time_dim(batch)  # 折叠时间维度
#             action_chunk = self.policy.predict_action_chunk(batch)  # List(torch.Tensor)

#         # 只取前 n_action_steps 个 actions
#         n_steps = self.policy.config.n_action_steps
#         action_chunk = action_chunk[:, :n_steps, :]

#         return action_chunk

#     def __call__(self, observation, tasks: list[str] = None, return_chunk=False):
#         """
#         运行策略推理

#         Args:
#             observation: 环境观测
#             env: 环境实例
#             return_chunk: 如果为True，返回完整的action chunk；否则返回单个action

#         Returns:
#             如果 return_chunk=True: [batch_size, n_action_steps, action_dim]
#             如果 return_chunk=False: [batch_size, action_dim]
#         """
#         # 如果请求返回完整 chunk，直接调用 get_action_chunk
#         if return_chunk:
#             return self.get_action_chunk(observation, tasks=tasks)

#         # 初始化 batch size 和 action queue
#         if self.batch_size is None:
#             if tasks is not None:
#                 self.batch_size = len(tasks)
#             else:
#                 # Fallback logic
#                 self.batch_size = 1 # TODO： 确认batch_size是否正确
#             self.action_queue = [[] for _ in range(self.batch_size)]

#         # 检查是否需要预测新的 action chunk
#         need_prediction = [len(queue) == 0 for queue in self.action_queue]

#         if any(need_prediction):
#             # 获取新的 action chunk
#             action_chunk = self.get_action_chunk(observation, tasks=tasks)

#             # 将 action chunk 分配到各个环境的队列中
#             for env_idx in range(self.batch_size):
#                 if need_prediction[env_idx]:
#                     # 将该环境的所有 actions 加入队列
#                     n_steps = action_chunk.shape[1]
#                     for step_idx in range(n_steps):
#                         self.action_queue[env_idx].append(action_chunk[env_idx, step_idx])

#         # 从每个环境的队列中取出第一个 action
#         action_numpy = np.array([queue.pop(0) for queue in self.action_queue])

#         assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"
#         return action_numpy

#     def reset(self):
#         """重置策略状态（包括 action queue）"""
#         if hasattr(self.policy, 'reset'):
#             self.policy.reset()
#         if self.action_queue is not None:
#             self.action_queue = [[] for _ in range(self.batch_size)]
import torch
from torch import nn
import torch.nn.functional as F  # [新增] 用于 Resize
import numpy as np
from vlaresidual.utils.train_utils import fold_time_dim, unfold_time_dim, decode_task_description_from_obs
from lerobot.policies.factory import make_policy
from vlaresidual.utils.print_utils import print_green
# =============================================================================
# 1. 统计数据 (保持不变)
# =============================================================================


NORM_STATS = {
    "state": {
        "mean": [
            -0.3303123116493225, 0.1998661607503891, 0.3552396595478058, -2.3032703399658203,
            -0.1356499046087265, 2.4885408878326416, 0.8061940670013428, 0.03619449958205223
        ],
        "std": [
            0.05331510677933693, 0.2647896409034729, 0.2470577359199524, 0.27789509296417236,
            0.21110141277313232, 0.23023897409439087, 0.39245960116386414, 0.03628121316432953
        ]
    },
    "action": {
        "mean": [
            0.0030841389670968056, 0.006400584243237972, -0.006992667447775602, -0.009621165692806244,
            0.0033867715392261744, -0.02155510149896145, 0.003844755468890071
        ],
        "std": [
            0.1866438388824463, 0.13990944623947144, 0.2263418585062027, 0.18003825843334198,
            0.24267759919166565, 0.11923189461231232, 0.11489319056272507
        ]
    }
}

class OpenPi(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.policy = make_policy(
            cfg=cfg.policy,
            env_cfg=cfg.env,
            rename_map=cfg.rename_map,
        )
        
        self.device = self.policy.config.device
        self.postprocessor = nn.Identity()
        
        # 归一化参数
        self.state_mean = torch.tensor(NORM_STATS["state"]["mean"], device=self.device, dtype=torch.float32)
        self.state_std  = torch.tensor(NORM_STATS["state"]["std"],  device=self.device, dtype=torch.float32)
        self.action_mean = torch.tensor(NORM_STATS["action"]["mean"], device=self.device, dtype=torch.float32)
        self.action_std  = torch.tensor(NORM_STATS["action"]["std"],  device=self.device, dtype=torch.float32)

        # PaliGemma 图像归一化参数 (均值0.5, 方差0.5 -> 映射到 [-1, 1])
        # SigLIP 预处理标准
        self.img_mean = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(3, 1, 1)
        self.img_std  = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(3, 1, 1)

        # Tokenizer
        from transformers import AutoTokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "google/paligemma-3b-pt-224", padding_side="right"
            )
        except OSError:
            print("⚠️ 无法联网加载 Tokenizer，请确保本地缓存存在。")
            raise

        # [FIX] 与 preprocessor.json 保持一致
        self.max_len = 200 

        self.action_queue = None
        self.batch_size = None

        print(f"✓ PI05Policy (Manual Pipeline Fixed) 初始化完成")

    def preprocess_inputs(self, observation, tasks):
        
        tasks = "Put the red chili peppers into the basket"
        

        batch = {}
        
        for key in ["image", "image2"]:
            # 兼容各种嵌套结构
            if 'pixels' in observation and key in observation['pixels']:
                img = observation['pixels'][key]
            else:
                img = observation.get(key, None)
                
            if img is not None:
                if not isinstance(img, torch.Tensor):
                    img = torch.tensor(img, device=self.device)
                else:
                    img = img.to(self.device)
                
                # 调整维度 (B, H, W, C) -> (B, C, H, W)
                if img.ndim == 4 and img.shape[-1] == 3:
                    img = img.permute(0, 3, 1, 2)
                
                # 归一化到 [0, 1]
                if img.dtype == torch.uint8:
                    img = img.float() / 255.0
                elif img.max() > 1.0:
                    img = img / 255.0
                
                # Resize 到 224x224
                if img.shape[-2:] != (224, 224):
                    img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)

                # 归一化到 [-1, 1]
                img = (img - self.img_mean) / self.img_std
                
                # [DEBUG] 保存图片检查摄像头是否接反
                # 取 Batch 0, 还原到 [0, 255], 调整为 (H, W, C)
                # -----------------------------------------------------------
                try:
                    import cv2
                    debug_single = img[0].detach()  # 取第一张图 (C, H, W)
                    # 反归一化: [-1, 1] -> [0, 1] -> [0, 255]
                    debug_np = (debug_single.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255
                    # RGB -> BGR (OpenCV 格式)
                    debug_np = cv2.cvtColor(debug_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"debug_camera_{key}.jpg", debug_np)
                    # print(f"已保存调试图片: debug_camera_{key}.jpg")
                except Exception as e:
                    print(f"保存调试图片失败: {e}")
                # -----------------------------------------------------------
                
                batch[f"observation.images.{key}"] = img
                

       

        # --- B. 处理状态 (State) ---
        state = observation.get('agent_pos')
        if state is None:
            state = observation.get('observation.state')

        if state is not None:
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, device=self.device, dtype=torch.float32)
            else:
                state = state.to(self.device, dtype=torch.float32)
            
            # 状态归一化
            state_norm = (state - self.state_mean) / self.state_std
            batch["observation.state"] = state_norm

        # --- C. 处理文本 ---
        if tasks is None:
            raw_task = observation.get('task', None)
            if raw_task is not None:
                if isinstance(raw_task, np.ndarray) or isinstance(raw_task, list):
                    tasks = raw_task
                else:
                    tasks = [raw_task]
            else:
                batch_size = state.shape[0] if state is not None else 1
                tasks = [""] * batch_size

        if isinstance(tasks, np.ndarray):
             if tasks.dtype.kind in {'S', 'U'}:
                 tasks = tasks.tolist()
        tokens = self.tokenizer(
            tasks,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt"
        )
        
        batch["observation.language.tokens"] = tokens["input_ids"].to(self.device)
        batch["observation.language.attention_mask"] = tokens["attention_mask"].to(self.device).bool()
        
        return batch

    def postprocess_action(self, action_norm):
        """反归一化"""
        action = action_norm * self.action_std + self.action_mean
        
        # action[..., 1] = -action[..., 1]
        
        # 反转 Z 轴 (修复上下漂移 - 让它向下走)
        # 模型输出 -0.05 (想向下)，翻转成 +0.05，在 EE Frame 里就是向下伸
        # action[..., 2] = -action[..., 2]
        return action

    def get_action_chunk(self, observation, tasks=None):
        batch = self.preprocess_inputs(observation, tasks)
        
        with torch.inference_mode():
            action_chunk_norm = self.policy.predict_action_chunk(batch)
            print_green(action_chunk_norm)
            
        # 截取维度
        env_action_dim = self.action_mean.shape[0]
        if action_chunk_norm.shape[-1] > env_action_dim:
            action_chunk_norm = action_chunk_norm[..., :env_action_dim]
            
        # 反归一化
        action_chunk = self.postprocess_action(action_chunk_norm)
        
        # 截取步数
        n_steps = self.policy.config.n_action_steps
        action_chunk = action_chunk[:, :n_steps, :]
        
        return action_chunk

    def __call__(self, observation, tasks: list[str] = None, return_chunk=False):
        if return_chunk:
            return self.get_action_chunk(observation, tasks=tasks)

        if self.batch_size is None:
            if 'agent_pos' in observation:
                self.batch_size = len(observation['agent_pos'])
            elif tasks is not None:
                self.batch_size = len(tasks)
            else:
                self.batch_size = 1
            self.action_queue = [[] for _ in range(self.batch_size)]

        need_prediction = [len(queue) == 0 for queue in self.action_queue]

        if any(need_prediction):
            action_chunk = self.get_action_chunk(observation, tasks=tasks)
            for env_idx in range(self.batch_size):
                if need_prediction[env_idx]:
                    n_steps = action_chunk.shape[1]
                    for step_idx in range(n_steps):
                        self.action_queue[env_idx].append(action_chunk[env_idx, step_idx])

        action_values = [queue.pop(0) for queue in self.action_queue]
        action_tensor = torch.stack(action_values)
        
        return action_tensor

    def _recursive_to_numpy(self, x):
        pass # 保留空函数防止报错

    def _ensure_image_uint8(self, x):
        pass 

    def reset(self):
        if hasattr(self.policy, 'reset'):
            self.policy.reset()
        if self.action_queue is not None:
            self.action_queue = [[] for _ in range(self.batch_size)]