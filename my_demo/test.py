import numpy as np
from scipy.spatial.transform import Rotation

def pose_7d_to_matrices(pose_7d):
    """
    将 7D 位姿 [x, y, z, qx, qy, qz, qw] 转换为 4x4 变换矩阵。
    """
    # 1. 拆分平移和四元数
    # 注意：需确认你的 7D 向量中，实部 qw 是在最后还是最前。
    # 这里假设是常见的 TUM 格式: [x, y, z, qx, qy, qz, qw]
    t = np.array(pose_7d[0:3])
    quat = pose_7d[3:7] 
    
    # 2. 四元数转 3x3 旋转矩阵 (scipy 默认四元数顺序为 x, y, z, w)
    R_wc = Rotation.from_quat(quat).as_matrix()
    
    # 3. 组装 T_wc (Camera to World 矩阵)
    T_wc = np.eye(4)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = t
    
    # 4. 求逆得到 T_cw (World to Camera 矩阵)
    # 方法一：直接用 numpy 求逆
    # T_cw = np.linalg.inv(T_wc)
    
    # 方法二：利用正交矩阵特性手动求逆（计算速度更快，且不易丢失精度）
    T_cw = np.eye(4)
    R_cw = R_wc.T
    t_cw = -R_cw @ t
    
    T_cw[:3, :3] = R_cw
    T_cw[:3, 3] = t_cw
    
    return T_cw, T_wc

# === 测试一下 ===
# 假设 Mega-SAM 输出的 7 维向量如下：
sample_pose = [-0.2212127447128296, 0.0630766749382019, 0.07313699275255203, -0.05893091484904289, 0.28921180963516235, 0.18304342031478882, 0.9377524852752686]
world2cam_matrix, cam2world_matrix = pose_7d_to_matrices(sample_pose)

print("World-to-Camera (T_cw) 4x4 矩阵:\n", world2cam_matrix)