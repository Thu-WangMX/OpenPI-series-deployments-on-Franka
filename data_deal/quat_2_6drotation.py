import numpy as np
from scipy.spatial.transform import Rotation as R

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    归一化向量
    Args:
        v: (N, 3) array
    Returns:
        (N, 3) normalized array
    """
    v_mag = np.linalg.norm(v, axis=-1, keepdims=True)
    v_mag = np.maximum(v_mag, 1e-8)
    return v / v_mag

def rotation6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """
    将 6D 旋转表示 (Continuous Rotation 6D) 转换为 3x3 旋转矩阵。
    使用 Gram-Schmidt 正交化。
    
    Args:
        d6: (N, 6) array, [r1x, r1y, r1z, r2x, r2y, r2z] (前两列)
    Returns:
        (N, 3, 3) rotation matrices
    """
    d6 = np.atleast_2d(d6)
    
    x_raw = d6[:, 0:3]
    y_raw = d6[:, 3:6]
    
    # 1. 归一化第一列 (x)
    x = normalize_vector(x_raw)
    
    # 2. 计算第三列 (z) = x cross y
    z = np.cross(x, y_raw)
    z = normalize_vector(z)
    
    # 3. 重新计算第二列 (y) = z cross x (保证正交)
    y = np.cross(z, x)
    
    # 4. 堆叠成矩阵 (按列)
    # x, y, z shape: (N, 3) -> (N, 3, 1)
    x = x[..., np.newaxis]
    y = y[..., np.newaxis]
    z = z[..., np.newaxis]
    
    matrix = np.concatenate((x, y, z), axis=2)
    return matrix.squeeze() # 如果输入是 (6,) 则返回 (3,3)

def matrix_to_rotation6d(matrix: np.ndarray) -> np.ndarray:
    """
    将 3x3 旋转矩阵转换为 6D 旋转表示。
    取矩阵的前两列并展平。
    
    Args:
        matrix: (N, 3, 3) or (3, 3) rotation matrices
    Returns:
        (N, 6) or (6,) array
    """
    # 确保输入是 (N, 3, 3)
    if matrix.ndim == 2:
        matrix = matrix[np.newaxis, ...]
        
    # 取前两列: matrix[:, :, 0] 和 matrix[:, :, 1]
    # 使用切片 matrix[:, :, :2] 得到 (N, 3, 2)
    # 交换轴使得变成 (N, 2, 3) 以便按顺序 flatten: [c1_x, c1_y, c1_z, c2_x, ...]
    # 注意：你的参考代码是 swapaxes(1, 2) 然后 reshape，这意味着它先存第一列，再存第二列
    batch_dim = matrix.shape[0]
    
    # 提取前两列: [N, 3, 2]
    rot_6d = matrix[..., :2]
    
    # 转置成 [N, 2, 3] -> Flatten -> [N, 6]
    # 这样顺序是 [col1_x, col1_y, col1_z, col2_x, col2_y, col2_z]
    rot_6d = rot_6d.swapaxes(1, 2).reshape(batch_dim, 6)
    
    return rot_6d.squeeze()

def quaternion_to_rotation6d(quat: np.ndarray) -> np.ndarray:
    """
    四元数 -> 6D 旋转
    数据格式: [x, y, z, w] (Scalar-Last)
    
    Args:
        quat: (N, 4) or (4,) array, order [x, y, z, w]
    Returns:
        (N, 6) or (6,) array
    """
    quat = np.atleast_2d(quat)
    
    # Scipy 默认就是 [x, y, z, w]，无需调整顺序
    r = R.from_quat(quat)
    matrix = r.as_matrix() # (N, 3, 3)
    
    rot6d = matrix_to_rotation6d(matrix)
    return rot6d.squeeze()

def rotation6d_to_quaternion(d6: np.ndarray) -> np.ndarray:
    """
    6D 旋转 -> 四元数
    返回格式: [x, y, z, w] (Scalar-Last)
    
    Args:
        d6: (N, 6) or (6,) array
    Returns:
        (N, 4) or (4,) array
    """
    matrix = rotation6d_to_matrix(d6)
    
    r = R.from_matrix(matrix)
    
    # Scipy 返回的就是 [x, y, z, w]，无需调整顺序
    return r.as_quat()

# === 测试代码 ===
if __name__ == "__main__":
    # 测试数据: 单位四元数 (xyzw格式) -> [0, 0, 0, 1]
    # 应该是单位矩阵 -> [1,0,0, 0,1,0]
    q_identity = np.array([0.0, 0.0, 0.0, 1.0]) 
    r6d = quaternion_to_rotation6d(q_identity)
    print(f"Quat {q_identity} (xyzw) -> Rot6D: {r6d}")
    # 预期输出: [1. 0. 0. 0. 1. 0.]
    
    q_back = rotation6d_to_quaternion(r6d)
    print(f"Rot6D -> Quat (xyzw): {q_back}")
    # 预期输出: [0. 0. 0. 1.] (或者 [0. 0. 0. -1.]，因为 q 和 -q 代表相同旋转)
    
    # 测试批量
    # [0, 0, 0, 1] -> Identity
    # [1, 0, 0, 0] -> 绕 X 轴旋转 180 度 (x=1, w=0)
    qs = np.array([[0., 0., 0., 1.], [1., 0., 0., 0.]]) 
    r6ds = quaternion_to_rotation6d(qs)
    print(f"Batch Input Shape: {qs.shape} -> Output Shape: {r6ds.shape}")