from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_policy
from lerobot.policies.factory import make_pre_post_processors
import numpy as np
import torch
from torch import nn
from vlaresidual.utils.train_utils import decode_task_description_from_obs
from vlaresidual.utils.train_utils import fold_time_dim
from vlaresidual.utils.train_utils import unfold_time_dim


class OpenPi(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.policy = make_policy(
            cfg=cfg.policy,
            env_cfg=cfg.env,
            rename_map=cfg.rename_map,
        )

        # 创建输入/输出预处理器
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            preprocessor_overrides={
                "device_processor": {"device": str(self.policy.config.device)},
                "rename_observations_processor": {"rename_map": cfg.rename_map},
            },
        )

        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

        # Action queue 管理
        self.action_queue = None
        self.batch_size = None

        # 获取设备
        self.device = self.policy.config.device

        print("✓ PI05Policy 初始化完成")
        print(f"  - n_action_steps: {self.policy.config.n_action_steps}")
        print(f"  - chunk_size: {self.policy.config.chunk_size}")

    def _recursive_to_numpy(self, x):
        """
        安全地将 Tensor (GPU/CPU) 转为 NumPy (CPU)。
        如果是 Dict，则递归转换。
        """
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if isinstance(x, dict):
            return {k: self._recursive_to_numpy(v) for k, v in x.items()}
        if isinstance(x, list):
            return [self._recursive_to_numpy(v) for v in x]
        return x

    def _ensure_image_uint8(self, x):
        """
        递归地将图像数据强制转换为 uint8。
        LeRobot 的 preprocess_observation 严格要求 uint8 输入。
        """
        if isinstance(x, dict):
            return {k: self._ensure_image_uint8(v) for k, v in x.items()}

        if isinstance(x, np.ndarray):
            # 判定是否为图像：维度 >= 3 且 最后一维是 3
            # 注意：这里的 x 可能是已经被折叠过的 (B*T, H, W, C)
            if x.ndim >= 3 and x.shape[-1] == 3:
                # 如果是浮点数
                if x.dtype in [np.float32, np.float64]:
                    # 检查是否被归一化到了 [0, 1]
                    if x.max() <= 1.05:  # 留点误差空间
                        x = x * 255.0

                    # 强制转为 uint8
                    return x.astype(np.uint8)

            # 也可以针对特定 key (如 'image') 做判断，但维度判断通常足够通用

        return x

    def _add_envs_task(self, observation, task):
        """
        将任务描述添加到观测中
        参考lerobot中对pi调用的示例
        """
        if task is not None:
            if isinstance(observation, dict):
                observation["task"] = task
            else:
                raise ValueError("Observation should be a dict to add task description.")
        return observation

    def get_action_chunk(self, observation):
        """
        获取 action chunk，只返回前 n_action_steps 个
        参考lerobot中对pi调用的示例
        """
        # 预处理观测

        # 获取task
        task = decode_task_description_from_obs(observation)

        # 处理后的observation{'agent_pos', 'pixels'{'image', 'image2'}, 'task_description'}
        # agent_pos: numpy.ndarray
        # pixels: dict{ numpy.ndarray, dtype=uint8 }, [B, H, W, C]
        # task_description: numpy.ndarray, dtype='<U82'
        observation = self._recursive_to_numpy(observation)

        # folded_obs{'agent_pos', 'pixels'{'image', 'image2'}, 'task_description'}, 只折叠了维度
        folded_obs, original_shapes = fold_time_dim(observation)
        b, t = original_shapes

        # 确保图像是 uint8 格式
        folded_obs = self._ensure_image_uint8(folded_obs)

        # 此处调用lerobot的接口，处理后的observation不含task，后续需把task加进来
        # processed_folded_obs{'observation.images.image', 'observation.images.image2', 'observation.state'}
        # agent_pos -> observation.state, torch.Tensor
        # image -> observation.images.image, torch.Tensor, 除255做归一化
        # image2 -> observation.images.image2, torch.Tensor, 除255做归一化
        processed_folded_obs = preprocess_observation(folded_obs)

        observation = unfold_time_dim(processed_folded_obs, original_shapes)

        # 处理后的observation{'observation.images.image', 'observation.images.image2', 'observation.state', 'task'}
        observation = self._add_envs_task(observation, task)

        batch = self.preprocessor(observation)

        # 使用 predict_action_chunk 方法获取完整的 action chunk
        with torch.inference_mode():
            batch, _ = fold_time_dim(batch)  # 折叠时间维度
            action_chunk = self.policy.predict_action_chunk(batch)  # List(torch.Tensor)

        # 只取前 n_action_steps 个 actions
        n_steps = self.policy.config.n_action_steps
        action_chunk = action_chunk[:, :n_steps, :]

        return action_chunk

    def __call__(self, observation, tasks: list[str] = None, return_chunk=False):
        """
        运行策略推理

        Args:
            observation: 环境观测
            env: 环境实例
            return_chunk: 如果为True，返回完整的action chunk；否则返回单个action

        Returns:
            如果 return_chunk=True: [batch_size, n_action_steps, action_dim]
            如果 return_chunk=False: [batch_size, action_dim]
        """
        # 如果请求返回完整 chunk，直接调用 get_action_chunk
        if return_chunk:
            return self.get_action_chunk(observation, tasks=tasks)

        # 初始化 batch size 和 action queue
        if self.batch_size is None:
            if tasks is not None:
                self.batch_size = len(tasks)
            else:
                # Fallback logic
                self.batch_size = 1  # TODO： 确认batch_size是否正确
            self.action_queue = [[] for _ in range(self.batch_size)]

        # 检查是否需要预测新的 action chunk
        need_prediction = [len(queue) == 0 for queue in self.action_queue]

        if any(need_prediction):
            # 获取新的 action chunk
            action_chunk = self.get_action_chunk(observation, tasks=tasks)

            # 将 action chunk 分配到各个环境的队列中
            for env_idx in range(self.batch_size):
                if need_prediction[env_idx]:
                    # 将该环境的所有 actions 加入队列
                    n_steps = action_chunk.shape[1]
                    for step_idx in range(n_steps):
                        self.action_queue[env_idx].append(action_chunk[env_idx, step_idx])

        # 从每个环境的队列中取出第一个 action
        action_numpy = np.array([queue.pop(0) for queue in self.action_queue])

        assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"
        return action_numpy

    def reset(self):
        """重置策略状态（包括 action queue）"""
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        if self.action_queue is not None:
            self.action_queue = [[] for _ in range(self.batch_size)]
