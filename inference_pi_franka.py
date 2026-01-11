import cv2
import pyrealsense2 as rs
import numpy as np
from PIL import Image
import torch
import time
import socket
import struct
import argparse
import threading

# 引入转换库 (确保 transformations.py 在同级目录)
# 该库期望四元数顺序为 [w, x, y, z]
import transformations 

# OpenPi Imports
from openpi.training import config
from openpi.policies import policy_config

# ==========================================
# ⚙️ 配置区域 (Configuration)
# ==========================================

# [执行策略配置]
EXECUTION_STEPS = 30
ACTION_DT = 0.04  # 25Hz 控制频率

# [网络配置]
PC2_IP_TARGET = '192.168.2.222'  # PC2 IP (Robot Side)
PC2_PORT_TARGET = 9090           
PC1_BIND_IP = '0.0.0.0'          # 本机监听 IP
PC1_BIND_PORT = 9091             

# [模型配置]
CHECKPOINT_PATH = "/mnt/satadisk2/ckpt/1222_dataset_trained_ckpt/4000"
MODEL_NAME = "pi0_franka_low_mem_finetune"
TASK_INSTRUCTION = "pick up the red chilli pepper into the basket."

# ==========================================
# 🛠️ 辅助函数
# ==========================================

def preprocess_image(image_pil: Image.Image, out_size=(224, 224)) -> Image.Image:
    img = np.array(image_pil)
    img_r = cv2.resize(img, out_size, interpolation=cv2.INTER_AREA)
    return Image.fromarray(img_r, mode="RGB")

def start_pipeline(serial: str):
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(cfg)
    return pipeline

# ==========================================
# 📡 网络通信 Setup
# ==========================================

sock_sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# Action 发送格式: [x, y, z, qw, qx, qy, qz, gripper] (8 doubles)
ACTION_FMT = "<8d"  

state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
state_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
state_sock.bind((PC1_BIND_IP, PC1_BIND_PORT))
state_sock.settimeout(0.0)

# 接收格式: [x, y, z, qw, qx, qy, qz, gripper, q1...q7] (15 doubles)
STATE_FMT = "<15d" 
last_robot_state = np.zeros(15, dtype=np.float64) 

def poll_robot_state_nonblocking():
    """读取最新的机器人状态，清空缓冲区"""
    global last_robot_state
    while True:
        try:
            data, _ = state_sock.recvfrom(2048) 
            if len(data) == struct.calcsize(STATE_FMT):
                last_robot_state = np.array(struct.unpack(STATE_FMT, data), dtype=np.float64)
        except BlockingIOError:
            break
        except Exception:
            break
    return last_robot_state

# ==========================================
# 🚀 主程序
# ==========================================

def main():
    print(f"[Init] Loading Model: {MODEL_NAME}...")
    cfg_model = config.get_config(MODEL_NAME)
    policy = policy_config.create_trained_policy(cfg_model, CHECKPOINT_PATH)
    print("[Init] Model Loaded.")
    print(f"[Config] Execution Steps: {EXECUTION_STEPS} | Interval: {ACTION_DT}s")

    # --- RealSense Setup ---
    ctx = rs.context()
    devices = list(ctx.query_devices())
    if len(devices) < 2:
        print(f"⚠️ Warning: Need 2 cameras. Found {len(devices)}.")
    
    serials = [d.get_info(rs.camera_info.serial_number) for d in devices[:2]]
    print(f"[Camera] Detected: {serials}")
    
    pipe_wrist = start_pipeline(serials[1])
    pipe_front = start_pipeline(serials[0])
    print("[Camera] Pipelines started.")

    # 计时器
    t_last_inference_loop = time.time()
    t_last_control_step = time.time()

    try:
        while True:
            # --- 1. 计算循环频率 ---
            t_now = time.time()
            infer_dt = t_now - t_last_inference_loop
            t_last_inference_loop = t_now
            print(f"\n[Main Loop] Inference Freq: {1.0/infer_dt:.2f} Hz | Time: {infer_dt:.3f}s")

            # --- 2. 采集图像 ---
            try:
                frames_w = pipe_wrist.wait_for_frames(timeout_ms=1000)
                frames_f = pipe_front.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                print("⚠️ Camera Timeout! Retrying...")
                continue
            
            color_w = frames_w.get_color_frame()
            color_f = frames_f.get_color_frame()
            if not color_w or not color_f: continue

            # 转换
            pil_w = Image.fromarray(cv2.cvtColor(np.asanyarray(color_w.get_data()), cv2.COLOR_BGR2RGB))
            pil_f = Image.fromarray(cv2.cvtColor(np.asanyarray(color_f.get_data()), cv2.COLOR_BGR2RGB))

            input_wrist = preprocess_image(pil_w)
            input_front = preprocess_image(pil_f)
            
            # --- 3. 获取机器人状态 ---
            raw_state = poll_robot_state_nonblocking().astype(np.float32)
            
            # 解析: [x, y, z, qw, qx, qy, qz, gripper, ...]
            # 此时 PC2 已经修复，发来的是 [w, x, y, z] 标准顺序
            curr_xyz = raw_state[0:3]
            curr_quat = raw_state[3:7] # [w, x, y, z]
            curr_gripper = raw_state[7]
            
            # 转换 Quat -> 6D Rotation 
            # transformations.py 期望输入 [w, x, y, z]，此处匹配
            curr_rot6d = transformations.quaternion_to_rotation6d(curr_quat)
            
            # --- 遵照你的指令：只传 10 维，不填充 ---
            state_basic = np.concatenate([curr_xyz, curr_rot6d, [curr_gripper]]) # Shape: (10,)
            
            # 组装输入字典
            example = {
                "observation/image": input_front,
                "observation/wrist_image": input_wrist,
                "observation/state": state_basic, # 直接传 10 维
                "prompt": TASK_INSTRUCTION,
            }
            print("state_basic",state_basic)

            #--- 4. 模型推理 ---
            try:
                t_infer_start = time.time()
                result = policy.infer(example)
                t_infer_end =  time.time()
                print("推理时间",t_infer_end-t_infer_start)
                action_chunk = result["actions"] # [Time, Dim]

                # --- 5. 动作执行循环 ---
                steps_to_run = min(EXECUTION_STEPS, len(action_chunk))
                print(f"  -> Executing {steps_to_run} steps...")
                
                for i in range(steps_to_run):
                    # 计算控制频率
                    t_step_now = time.time()
                    step_dt = t_step_now - t_last_control_step
                    t_last_control_step = t_step_now
                    
                    if i % 10 == 0 and i > 0:
                        print(f"     [Control] Freq: {1.0/step_dt:.2f} Hz")

                    # 解析动作
                    action_pred = action_chunk[i]
                    pred_xyz = action_pred[0:3]
                    pred_rot6d = action_pred[3:9]
                    pred_gripper = action_pred[9]

                    # 6D -> Quat
                    # transformations.py 返回的是 [w, x, y, z]
                    pred_quat = transformations.rotation6d_to_quaternion(pred_rot6d)
                    
                    # 组装发送包 [8维]
                    # 发送顺序: [x, y, z, w, x, y, z, gripper]
                    udp_packet = np.concatenate([pred_xyz, pred_quat, [pred_gripper]])
                    
                    if len(udp_packet) == 8:
                        msg = struct.pack(ACTION_FMT, *udp_packet)
                        sock_sender.sendto(msg, (PC2_IP_TARGET, PC2_PORT_TARGET))
                    
                    time.sleep(ACTION_DT)

            except Exception as e:
                print(f"Inference/Execution Error: {e}")
                time.sleep(0.1)
            
            
            
            # try:
            #     t_infer_start = time.time()
            #     result = policy.infer(example)
                
            #     # 计算推理花了多久
            #     infer_duration = time.time() - t_infer_start
            #     print(f"推理时间: {infer_duration:.4f}s")
                
            #     action_chunk = result["actions"] 

            #     # ==========================================
            #     # ✅ 恢复这段逻辑来解决“一进一退”
            #     # ==========================================
                
            #     # 1. 计算因为推理卡顿，导致有多少步动作已经“过期”了
            #     # 例如：推理 0.15s / 控制 0.04s = 3.75 -> 跳过 4 步
            #     steps_to_skip = int(infer_duration / ACTION_DT) + 1
                
            #     # 2. 限制一下，别跳太多 (比如最多跳 10 步)
            #     steps_to_skip = min(steps_to_skip, 10)
                
            #     # 3. 计算实际要执行的步数
            #     steps_total = min(EXECUTION_STEPS, len(action_chunk))
                
            #     print(f"  -> 延迟补偿: 跳过前 {steps_to_skip} 步 (过期), 执行 {steps_to_skip} -> {steps_total}")

            #     # 4. 【关键】循环从 steps_to_skip 开始，而不是从 0 开始
            #     for i in range(steps_to_skip, steps_total):
                    
            #         # --- 下面的发送逻辑保持不变 ---
            #         t_step_now = time.time()
            #         step_dt = t_step_now - t_last_control_step
            #         t_last_control_step = t_step_now
                    
            #         # 解析动作
            #         action_pred = action_chunk[i]
            #         pred_xyz = action_pred[0:3]
            #         pred_rot6d = action_pred[3:9]
            #         pred_gripper = action_pred[9]

            #         # 6D -> Quat
            #         pred_quat = transformations.rotation6d_to_quaternion(pred_rot6d)
                    
            #         # 组装发送
            #         udp_packet = np.concatenate([pred_xyz, pred_quat, [pred_gripper]])
                    
            #         if len(udp_packet) == 8:
            #             msg = struct.pack(ACTION_FMT, *udp_packet)
            #             sock_sender.sendto(msg, (PC2_IP_TARGET, PC2_PORT_TARGET))
                    
            #         # 严格控制频率
            #         time.sleep(ACTION_DT)

            # except Exception as e:
            #     print(f"Inference/Execution Error: {e}")
            #     time.sleep(0.1)
    finally:
        pipe_wrist.stop()
        pipe_front.stop()
        sock_sender.close()
        state_sock.close()

if __name__ == "__main__":
    main()