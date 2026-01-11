#验证quat和rotation6d的转换是否无损
import numpy as np
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. 你提供的核心转换代码 (保持不变)
# ==========================================

def normalize_vector(v: np.ndarray) -> np.ndarray:
    v_mag = np.linalg.norm(v, axis=-1, keepdims=True)
    v_mag = np.maximum(v_mag, 1e-8)
    return v / v_mag

def rotation6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    d6 = np.atleast_2d(d6)
    x_raw = d6[:, 0:3]
    y_raw = d6[:, 3:6]
    x = normalize_vector(x_raw)
    z = np.cross(x, y_raw)
    z = normalize_vector(z)
    y = np.cross(z, x)
    x = x[..., np.newaxis]
    y = y[..., np.newaxis]
    z = z[..., np.newaxis]
    matrix = np.concatenate((x, y, z), axis=2)
    return matrix.squeeze()

def matrix_to_rotation6d(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim == 2:
        matrix = matrix[np.newaxis, ...]
    batch_dim = matrix.shape[0]
    rot_6d = matrix[..., :2]
    rot_6d = rot_6d.swapaxes(1, 2).reshape(batch_dim, 6)
    return rot_6d.squeeze()

def quaternion_to_rotation6d(quat: np.ndarray) -> np.ndarray:
    # 输入: [w, x, y, z]
    quat = np.atleast_2d(quat)
    # Scipy 需要 [x, y, z, w]
    scipy_quat = np.concatenate([quat[:, 1:], quat[:, 0:1]], axis=1)
    r = R.from_quat(scipy_quat)
    matrix = r.as_matrix()
    rot6d = matrix_to_rotation6d(matrix)
    return rot6d.squeeze()

def rotation6d_to_quaternion(d6: np.ndarray) -> np.ndarray:
    # 返回: [w, x, y, z]
    matrix = rotation6d_to_matrix(d6)
    r = R.from_matrix(matrix)
    scipy_quat = r.as_quat() # 返回 [x, y, z, w]
    if scipy_quat.ndim == 1:
        wxyz_quat = np.concatenate([scipy_quat[-1:], scipy_quat[:-1]])
    else:
        wxyz_quat = np.concatenate([scipy_quat[:, -1:], scipy_quat[:, :-1]], axis=1)
    return wxyz_quat

# ==========================================
# 2. 验证代码
# ==========================================

def verify_conversion():
    print("🚀 开始验证 Quaternion <-> Rot6D 的无损转换...")
    
    # --- A. 生成随机测试数据 ---
    N = 1000  # 测试 1000 个随机旋转
    print(f"Generating {N} random rotations...")
    
    # 使用 scipy 生成合法的随机旋转，确保输入是完美的单位四元数
    random_rots = R.random(N)
    scipy_quat = random_rots.as_quat() # [x, y, z, w]
    
    # 转换为你的格式 [w, x, y, z]
    q_input = np.concatenate([scipy_quat[:, 3:4], scipy_quat[:, :3]], axis=1)
    
    # --- B. 执行 Round-Trip (一来一回) ---
    # 1. Quat -> 6D
    r6d = quaternion_to_rotation6d(q_input)
    
    # 2. 6D -> Quat
    q_recovered = rotation6d_to_quaternion(r6d)
    
    # --- C. 计算误差 (关键！) ---
    # 注意：四元数 q 和 -q 代表同一个旋转 (Double Cover)
    # 我们不能简单计算 norm(q1 - q2)，而要看 min(norm(q1 - q2), norm(q1 + q2))
    # 或者检查点积的绝对值是否接近 1
    
    # 方法1：计算欧氏距离（考虑符号翻转）
    diff_plus = np.linalg.norm(q_input - q_recovered, axis=1)
    diff_minus = np.linalg.norm(q_input + q_recovered, axis=1)
    min_errors = np.minimum(diff_plus, diff_minus)
    
    # 方法2：计算角度误差 (Geodesic Distance)
    # 点积绝对值，夹紧到 [0, 1] 防止数值溢出
    dot_products = np.abs(np.sum(q_input * q_recovered, axis=1))
    dot_products = np.clip(dot_products, -1.0, 1.0)
    # 角度差 = 2 * arccos(|q1 . q2|)
    angle_errors_rad = 2 * np.arccos(dot_products)
    angle_errors_deg = np.degrees(angle_errors_rad)

    # --- D. 输出结果 ---
    max_error = np.max(min_errors)
    max_angle_error = np.max(angle_errors_deg)
    
    print("-" * 30)
    print(f"最大数值误差 (Euclidean): {max_error:.2e}")
    print(f"最大角度误差 (Degree):    {max_angle_error:.2e} 度")
    print("-" * 30)
    
    # 设定通过标准 (通常浮点数精度在 1e-7 左右)
    if max_error < 1e-6:
        print("✅ 验证成功！转换是无损的（在浮点误差范围内）。")
        
        # 展示前 3 个样本的对比
        print("\n👇 样本展示 (前3个):")
        for i in range(3):
            print(f"样本 {i}:")
            print(f"  原始 Quat: {q_input[i]}")
            print(f"  恢复 Quat: {q_recovered[i]}")
            
            # 检查符号是否翻转
            sign_flipped = np.dot(q_input[i], q_recovered[i]) < 0
            if sign_flipped:
                print("  (注意: 符号发生了翻转，但这代表相同的物理旋转)")
            print(f"  误差: {min_errors[i]:.2e}")
    else:
        print("❌ 验证失败！误差过大，请检查代码逻辑。")

if __name__ == "__main__":
    verify_conversion()