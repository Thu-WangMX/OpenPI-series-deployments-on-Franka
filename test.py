# import time
# from franky import Robot

# # 连机器人
# robot = Robot("172.16.0.2")

# print("请手动上下移动机械臂...")
# while True:
#     pose = robot.current_cartesian_state.pose.end_effector_pose
#     tx, ty, tz = pose.translation
    
#     # 实时打印，观察 Z 值变化
#     print(f"X: {tx:.4f} | Y: {ty:.4f} | Z: {tz:.4f} <--- 看这里")
#     time.sleep(0.1)
import numpy as np
from franky import Robot

# =================配置=================
ROBOT_IP = "172.16.0.2"
# ======================================

def quaternion_to_z_axis(q, order='wxyz'):
    """
    根据四元数计算末端 Z 轴在基座标系下的向量。
    q: 输入的四元数 [a, b, c, d]
    order: 'wxyz' 或 'xyzw'
    """
    # 归一化，防止计算误差
    q = q / np.linalg.norm(q)
    
    if order == 'wxyz':
        w, x, y, z = q[0], q[1], q[2], q[3]
    else: # xyzw
        x, y, z, w = q[0], q[1], q[2], q[3]

    # 根据旋转矩阵公式，第三列 (Z轴方向) 的计算公式：
    # R[0, 2] = 2(xz + wy)
    # R[1, 2] = 2(yz - wx)
    # R[2, 2] = 1 - 2(x^2 + y^2)
    
    z_axis_x = 2 * (x * z + w * y)
    z_axis_y = 2 * (y * z - w * x)
    z_axis_z = 1 - 2 * (x**2 + y**2)
    
    return np.array([z_axis_x, z_axis_y, z_axis_z])

def main():
    print(f"正在连接机器人 {ROBOT_IP}...")
    try:
        robot = Robot(ROBOT_IP)
        print("✅ 连接成功！")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    print("\n⚠️ 请确保机械臂当前的夹爪是【大致朝下】的 (指向地面)")
    input("准备好后按回车键开始检测...")

    # 读取一次状态
    state = robot.current_cartesian_state
    pose = state.pose.end_effector_pose
    
    # 获取原始四元数数据
    # franky/libfranka 通常返回一个 list 或 array
    q_raw = np.array(pose.quaternion) 
    
    print("-" * 40)
    print(f"原始四元数数据: {q_raw}")
    print("-" * 40)

    # --- 假设 1: 格式是 WXYZ ---
    z_vec_1 = quaternion_to_z_axis(q_raw, order='wxyz')
    
    # --- 假设 2: 格式是 XYZW ---
    z_vec_2 = quaternion_to_z_axis(q_raw, order='xyzw')

    print(f"假设 WXYZ 时，算出的 Z 轴朝向: {np.round(z_vec_1, 3)}")
    print(f"假设 XYZW 时，算出的 Z 轴朝向: {np.round(z_vec_2, 3)}")
    print("-" * 40)

    # --- 自动判定 ---
    # 既然夹爪朝下，Z 轴的 Z 分量应该是负数 (接近 -1)
    score_1 = z_vec_1[2] # 取 Z 分量
    score_2 = z_vec_2[2]

    if score_1 < -0.8 and score_2 > -0.8:
        print("🎉 结论: Franky 返回的是 【W, X, Y, Z】 (实部在前)")
    elif score_2 < -0.8 and score_1 > -0.8:
        print("🎉 结论: Franky 返回的是 【X, Y, Z, W】 (实部在后)")
    else:
        print("❓ 无法自动判断。")
        print("原因可能是：")
        print("1. 机械臂没有垂直朝下。")
        print("2. 两个假设算出来的 Z 分量都很小（比如机械臂是平放的）。")
        print("请把机械臂摆正（夹爪垂直指向桌面）再试一次。")

if __name__ == "__main__":
    main()