#!/usr/bin/env python3
import socket
import struct
import time

from franky import Affine
from franky import CartesianMotion
from franky import Gripper
from franky import ReferenceType
from franky import Robot
import numpy as np

# ==========================================
# ⚙️ 配置区域 (Configuration)
# ==========================================
# [网络配置]
PC1_IP = "192.168.2.223"  # ⚠️ 请确保此 IP 正确
PC1_PORT_TARGET = 9091

BIND_IP = "0.0.0.0"
BIND_PORT = 9090

# [机器人配置]
ROBOT_IP = "172.16.0.2"
DYNAMICS_FACTOR = 0.05
GRIPPER_SPEED = 0.05
GRIPPER_FORCE = 20.0
STATE_FREQ = 50
STATE_INTERVAL = 1.0 / STATE_FREQ

# [通信协议]
ACTION_FMT = "<8d"
ACTION_BYTES = struct.calcsize(ACTION_FMT)
STATE_FMT = "<15d"

# ==========================================
# 🤖 机器人初始化
# ==========================================
print(f"[Init] Connecting to Robot at {ROBOT_IP}...")
try:
    robot = Robot(ROBOT_IP)
    gripper = Gripper(ROBOT_IP)
    robot.relative_dynamics_factor = DYNAMICS_FACTOR
    try:
        robot.recover_from_errors()
    except:
        pass
    print("[Init] Robot Connected.")
except Exception as e:
    print(f"❌ Connection Failed: {e}")
    exit(1)

# ==========================================
# 📡 网络 Setup
# ==========================================
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock_recv.bind((BIND_IP, BIND_PORT))
sock_recv.settimeout(0.001)

sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


# ==========================================
# 🛠️ 辅助函数
# ==========================================
def normalize_quaternion(q):
    """确保四元数模长为1"""
    norm = np.linalg.norm(q)
    if norm < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def read_gripper_width(grp):
    try:
        return float(grp.width)
    except:
        return 0.0


def send_state_once():
    """读取机器人状态并发送给 PC1"""
    try:
        state = robot.current_cartesian_state
        pose = state.pose.end_effector_pose

        tx, ty, tz = pose.translation
        q_raw = pose.quaternion

        # 🚨 [关键修复] 假设 Franky 返回的是 [w, x, y, z] (Scalar First)
        qw = q_raw[0]
        qx = q_raw[1]
        qy = q_raw[2]
        qz = q_raw[3]

        # 获取夹爪和关节
        g_width = read_gripper_width(gripper)
        q_joints = list(robot.current_joint_state.position)

        # 打包发送 [w, x, y, z] 标准顺序
        data = struct.pack(STATE_FMT, tx, ty, tz, qw, qx, qy, qz, g_width, *q_joints)

        sock_send.sendto(data, (PC1_IP, PC1_PORT_TARGET))

    except Exception:
        pass


# ==========================================
# 🚀 主循环
# ==========================================
def main():
    print(f"[Network] Listening for Actions on {BIND_IP}:{BIND_PORT}")
    print(f"[Network] Sending States to {PC1_IP}:{PC1_PORT_TARGET}")

    # 初始打开夹爪
    gripper.open(GRIPPER_SPEED)
    last_state_time = 0.0

    # 打印一次当前姿态，用于确认四元数顺序
    init_pose = robot.current_cartesian_state.pose.end_effector_pose
    print(f"\n📢 [DEBUG CHECK] Current Quaternion: {init_pose.quaternion}\n")

    while True:
        # --- 1. 接收 Action (清空缓冲区) ---
        data = None
        while True:
            try:
                chunk, _ = sock_recv.recvfrom(1024)
                data = chunk
            except (TimeoutError, BlockingIOError):
                break

        if data and len(data) == ACTION_BYTES:
            try:
                # 解包: [x, y, z, qw, qx, qy, qz, gripper]
                act = struct.unpack(ACTION_FMT, data)

                target_pos = list(act[0:3])

                # VLA 发来的是 [w, x, y, z]
                target_quat_wxyz = np.array(act[3:7])
                target_quat_wxyz = normalize_quaternion(target_quat_wxyz)

                # 直接透传 [w, x, y, z] 给 Franky
                target_quat_final = target_quat_wxyz

                target_grip_cmd = act[7]

                # --- 运动控制 ---
                target_affine = Affine(target_pos, target_quat_final)
                motion = CartesianMotion(target_affine, ReferenceType.Absolute)
                robot.move(motion)

                # --- 夹爪控制 (修改版) ---
                try:
                    # 1. 限制目标范围在 Franka 物理极限内 [0.0, 0.08]米
                    # 如果你的模型输出是 0-1 的归一化数值，请确认是否需要乘以 0.08
                    target_width = np.clip(target_grip_cmd, 0.0, 0.08)

                    # 2. 读取当前宽度
                    current_width = read_gripper_width(gripper)

                    # 3. 只有当目标宽度与当前宽度差异超过 1mm 时才发送指令
                    # 这是为了防止在同一位置反复调用阻塞的 move 函数，导致主循环卡顿
                    if abs(target_width - current_width) > 0.001:
                        gripper.move(target_width, GRIPPER_SPEED)

                except Exception:
                    # 忽略偶尔的夹爪通信错误
                    pass

            except Exception:
                pass

        # --- 2. 发送 State ---
        t_now = time.time()
        if t_now - last_state_time > STATE_INTERVAL:
            send_state_once()
            last_state_time = t_now


if __name__ == "__main__":
    main()
