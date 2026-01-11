import multiprocessing
import os
from pathlib import Path
import pickle
import random

import cv2
import numpy as np
import torch
from tqdm import tqdm

# ================= 📝 配置区域 =================
CONFIG_NAME = "pi0_franka_low_mem_finetune"
JAX_CHECKPOINT_DIR = "/work/wmx/openpi/checkpoints/pi0_franka_low_mem_finetune/pi0_clean_single_grasp/40000"
TORCH_CHECKPOINT_DIR = "/work/wmx/openpi/ckpt_torch/after_clean_bs32_4w"
DATA_DIR = Path("/work/wmx/openpi/data_clean/single_grasp")

# 📊 统计评估配置
EVAL_NUM_EPISODES = 20  # 用多少个 Episode 做数值统计
EVAL_FRAMES_PER_EP = 10  # 每个 Episode 抽多少帧

# 🎥 视频可视化配置
VIS_EPISODE_IDX = 0  # 指定要把第几个文件做成视频 (0 表示随机列表中的第一个)
OUTPUT_DIR = "vis_results"
# ===============================================


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_frame_to_example(frame):
    """将一帧数据解析为 OpenPI 模型输入格式"""
    obs = frame["observations"]
    # 优先读取 language_instruction，如果没有则用默认
    task_desc = frame.get("language_instruction", "Put the red chili peppers into the basket")

    example = {
        "observation/image": np.array(obs["pixels"]["image"], dtype=np.uint8),
        "observation/wrist_image": np.array(obs["pixels"]["image2"], dtype=np.uint8),
        "observation/state": np.array(obs.get("observation.state", obs.get("agent_pos")), dtype=np.float32),
        "prompt": task_desc,
    }
    gt_action = np.array(frame["action"], dtype=np.float32)
    return example, gt_action


def prepare_data():
    """
    准备两份数据：
    1. eval_data: 用于计算 MSE 的散乱帧
    2. vis_data: 用于生成视频的完整 Episode 序列
    """
    all_files = sorted(list(DATA_DIR.glob("*.pkl")))
    if not all_files:
        raise FileNotFoundError(f"❌ 没在 {DATA_DIR} 找到 .pkl 文件")

    # 1. 准备统计数据 (随机抽样)
    eval_files = random.sample(all_files, min(len(all_files), EVAL_NUM_EPISODES))
    eval_batch = []

    print(f"📊 正在加载统计数据 ({len(eval_files)} episodes)...")
    for pkl_path in eval_files:
        data = load_pkl(pkl_path)
        indices = random.sample(range(len(data)), min(len(data), EVAL_FRAMES_PER_EP))
        for idx in indices:
            ex, gt = parse_frame_to_example(data[idx])
            eval_batch.append(({"id": f"{pkl_path.name}_{idx}"}, ex, gt))

    # 2. 准备视频数据 (取一个完整文件)
    vis_file = all_files[VIS_EPISODE_IDX % len(all_files)]
    print(f"🎥 正在加载可视化数据 (完整 Episode): {vis_file.name} ...")
    vis_data_raw = load_pkl(vis_file)
    vis_batch = []
    for idx, frame in enumerate(vis_data_raw):
        ex, gt = parse_frame_to_example(frame)
        vis_batch.append(({"id": f"vis_{idx}"}, ex, gt))

    return eval_batch, vis_batch, vis_file.name


# ==============================================================================
#  推理进程 (Multiprocessing Worker)
# ==============================================================================
def _worker_process(config_name, ckpt_dir, eval_data, vis_data, backend, queue):
    try:
        print(f"\n🚀 [{backend}] 进程启动...")
        if backend == "JAX":
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

        from openpi.policies import policy_config
        from openpi.training import config as _config

        config = _config.get_config(config_name)
        # 加载 Policy
        policy = policy_config.create_trained_policy(config, ckpt_dir)

        # 定义推理函数
        def infer_list(data_list, desc):
            res = []
            for _, example, _ in tqdm(data_list, desc=desc):
                if backend == "Torch":
                    with torch.inference_mode():
                        out = policy.infer(example)
                        action = out["actions"][0]
                        if isinstance(action, torch.Tensor):
                            action = action.cpu().numpy()
                else:
                    out = policy.infer(example)
                    action = np.array(out["actions"])[0]
                res.append(action)
            return res

        # 1. 跑统计数据
        eval_res = infer_list(eval_data, f"{backend} Eval")
        # 2. 跑视频数据
        vis_res = infer_list(vis_data, f"{backend} Vis")

        queue.put((True, eval_res, vis_res))

    except Exception as e:
        import traceback

        traceback.print_exc()
        queue.put((False, str(e), None))


def run_inference(backend, ckpt_dir, eval_data, vis_data):
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    p = ctx.Process(target=_worker_process, args=(CONFIG_NAME, ckpt_dir, eval_data, vis_data, backend, queue))
    p.start()

    try:
        success, eval_res, vis_res = queue.get()
    except Exception as e:
        p.terminate()
        raise RuntimeError(f"{backend} 数据获取失败: {e}")

    p.join()
    if not success:
        raise RuntimeError(f"{backend} 运行报错: {eval_res}")
    return eval_res, vis_res


# ==============================================================================
#  可视化绘制工具
# ==============================================================================
def draw_text_with_bg(img, text, pos, font_scale=0.4, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    x, y = pos
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(img, (x, y - h - 4), (x + w, y + baseline), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def generate_video(vis_batch, jax_res, torch_res, filename):
    print(f"\n🎬 正在生成对比视频: {filename}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, filename)

    # 获取第一帧尺寸
    img1 = vis_batch[0][1]["observation/image"]
    img2 = vis_batch[0][1]["observation/wrist_image"]
    h, w, _ = img1.shape

    # 画布布局: 上方留 120px 写字，下方左右拼接图像
    header_h = 130
    canvas_w = w * 2
    canvas_h = h + header_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, 10, (canvas_w, canvas_h))

    for i in tqdm(range(len(vis_batch)), desc="Rendering"):
        _, ex, gt = vis_batch[i]
        j_act = jax_res[i]
        t_act = torch_res[i]

        # 1. 图像部分
        im1_bgr = cv2.cvtColor(ex["observation/image"], cv2.COLOR_RGB2BGR)
        im2_bgr = cv2.cvtColor(ex["observation/wrist_image"], cv2.COLOR_RGB2BGR)
        imgs_combined = np.hstack([im1_bgr, im2_bgr])

        # 2. 背景部分
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[header_h:, :, :] = imgs_combined

        # 3. 绘制文字 (三行 Action)
        # 格式化函数
        def fmt_act(name, act, color):
            # 分成两段显示: J1-4 | J5-G
            s1 = ", ".join([f"{x:6.3f}" for x in act[:4]])
            s2 = ", ".join([f"{x:6.3f}" for x in act[4:]])
            return f"{name}: [{s1} | {s2}]"

        # 绘制
        draw_text_with_bg(canvas, f"Frame: {i:03d} | Task: {ex['prompt']}", (10, 20), font_scale=0.5)

        # GT (Green)
        draw_text_with_bg(canvas, fmt_act("GT   ", gt, None), (10, 50), text_color=(0, 255, 0))
        # JAX (Cyan)
        draw_text_with_bg(canvas, fmt_act("JAX  ", j_act, None), (10, 80), text_color=(255, 255, 0))
        # Torch (Orange/Blue in BGR)
        draw_text_with_bg(canvas, fmt_act("TORCH", t_act, None), (10, 110), text_color=(0, 165, 255))

        out.write(canvas)

    out.release()
    print(f"✅ 视频已保存: {save_path}")


# ==============================================================================
#  统计计算
# ==============================================================================
def compute_stats(eval_batch, jax_res, torch_res):
    print(f"\n📊 计算统计指标 (共 {len(eval_batch)} 帧)...")

    jax_mses = []
    torch_mses = []

    for i in range(len(eval_batch)):
        gt = eval_batch[i][2]
        j_a = jax_res[i]
        t_a = torch_res[i]

        # 简单的 MSE
        jax_mses.append(np.mean((j_a - gt) ** 2))
        torch_mses.append(np.mean((t_a - gt) ** 2))

    avg_j = np.mean(jax_mses)
    avg_t = np.mean(torch_mses)

    print("-" * 40)
    print(f"JAX Mean MSE   : {avg_j:.6f}")
    print(f"Torch Mean MSE : {avg_t:.6f}")
    print(f"Diff (J - T)   : {avg_j - avg_t:.6f}")
    print("-" * 40)

    if abs(avg_j - avg_t) < 1e-5:
        print("✅ 两个框架推理结果基本一致")
    else:
        print("⚠️ 存在精度差异，请检查")


# ==============================================================================
#  主函数
# ==============================================================================
def main():
    # 1. 准备数据
    eval_data, vis_data, vis_filename = prepare_data()

    # 2. JAX 推理
    print("\n>>> 开始 JAX 推理...")
    j_eval, j_vis = run_inference("JAX", JAX_CHECKPOINT_DIR, eval_data, vis_data)

    # 3. Torch 推理
    print("\n>>> 开始 Torch 推理...")
    t_eval, t_vis = run_inference("Torch", TORCH_CHECKPOINT_DIR, eval_data, vis_data)

    # 4. 生成视频
    video_name = f"compare_{vis_filename.replace('.pkl', '')}.mp4"
    generate_video(vis_data, j_vis, t_vis, video_name)

    # 5. 输出统计
    compute_stats(eval_data, j_eval, t_eval)


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
