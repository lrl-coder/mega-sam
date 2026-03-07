import torch
import numpy as np
import argparse


def compute_camera_errors(extr_est, extr_gt):
    """
    计算相机的旋转误差和平移误差
    :param extr_est: 估计的 w2c 矩阵, shape (T, 4, 4)
    :param extr_gt: 真实的 w2c 矩阵, shape (T, 4, 4)
    :return: rot_errors (度), trans_errors (距离单位)
    """
    # 1. 提取 R (T, 3, 3) 和 t (T, 3, 1)
    R_est = extr_est[:, :3, :3]
    t_est = extr_est[:, :3, 3:4]

    R_gt = extr_gt[:, :3, :3]
    t_gt = extr_gt[:, :3, 3:4]

    # ====================
    # 计算旋转误差 (Rotation Error)
    # ====================
    # 计算相对旋转 R_rel = R_est @ R_gt^T
    R_rel = torch.bmm(R_est, R_gt.transpose(1, 2))

    # 计算 Trace
    trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)

    # 限制在 [-1, 1] 范围内，防止浮点误差导致 arccos 产生 NaN
    cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)

    # 计算角度并转为度数
    rot_errors_rad = torch.acos(cos_theta)
    rot_errors_deg = torch.rad2deg(rot_errors_rad)

    # ====================
    # 计算平移误差 (Translation Error)
    # ====================
    # 计算相机光心在世界坐标系下的位置: C = -R^T @ t
    C_est = -torch.bmm(R_est.transpose(1, 2), t_est).squeeze(-1)  # (T, 3)
    C_gt = -torch.bmm(R_gt.transpose(1, 2), t_gt).squeeze(-1)     # (T, 3)

    # 计算 L2 距离
    trans_errors = torch.norm(C_est - C_gt, dim=-1)  # (T,)

    return rot_errors_deg, trans_errors


def load_and_evaluate(pred_npy_path, gt_npy_path):
    """
    从 npy 文件加载外参数据并计算误差。

    :param pred_npy_path: 预测外参 npy 路径，数组 shape 为 (M, 4, 4)
    :param gt_npy_path:   GT 外参 npy 路径，字典包含：
                            - 'extrinsics'       : shape (T, 4, 4) 的 GT 外参矩阵
                            - 'video_decode_frame': GT 对应的真实帧索引，如 [2, 3, 4, 5]
    """
    # ---------- 加载数据 ----------
    # 预测: shape (M, 4, 4)
    pred_arr = np.load(pred_npy_path)  # ndarray (M, 4, 4)

    # GT: 字典
    gt_dict = np.load(gt_npy_path, allow_pickle=True).item()
    extr_gt_arr = gt_dict['extrinsics']           # ndarray (T, 4, 4)
    frame_indices = gt_dict['video_decode_frame']  # ndarray 或 list，长度为 T

    frame_indices = np.array(frame_indices, dtype=np.int64)
    T = len(frame_indices)
    M = pred_arr.shape[0]

    print(f"预测外参帧数 M = {M}")
    print(f"GT  外参帧数 T = {T}")
    print(f"GT 对应的真实帧索引: {frame_indices}")

    assert M >= T, f"预测帧数 M={M} 必须大于等于 GT 帧数 T={T}"
    assert frame_indices.max() < M, (
        f"帧索引最大值 {frame_indices.max()} 超出预测数组范围 [0, {M-1}]"
    )

    # ---------- 按帧索引从预测中截取 ----------
    # pred_arr[frame_indices] → shape (T, 4, 4)
    pred_selected = pred_arr[frame_indices]

    # ---------- 转为 Tensor ----------
    extr_est_tensor = torch.tensor(pred_selected, dtype=torch.float64)
    extr_gt_tensor  = torch.tensor(extr_gt_arr,   dtype=torch.float64)

    # ---------- 计算误差 ----------
    rot_err, trans_err = compute_camera_errors(extr_est_tensor, extr_gt_tensor)

    print("\n===== 逐帧误差 =====")
    for i, (fi, re, te) in enumerate(zip(frame_indices, rot_err, trans_err)):
        print(f"  GT帧 {i:3d} (预测帧索引 {fi:4d}) | 旋转误差: {re.item():.4f} 度 | 平移误差: {te.item():.6f}")

    print("\n===== 汇总 =====")
    print(f"平均旋转误差 : {rot_err.mean().item():.4f} 度")
    print(f"中位旋转误差 : {rot_err.median().item():.4f} 度")
    print(f"平均平移误差 : {trans_err.mean().item():.6f}")
    print(f"中位平移误差 : {trans_err.median().item():.6f}")

    return rot_err, trans_err


# --------- 入口 ---------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算预测外参与GT外参的旋转/平移误差")
    parser.add_argument(
        "--pred",
        type=str,
        required=True,
        help="预测外参 npy 文件路径，shape (M, 4, 4)"
    )
    parser.add_argument(
        "--gt",
        type=str,
        required=True,
        help="GT 外参 npy 文件路径（字典），包含 'extrinsics'(T,4,4) 和 'video_decode_frame'(T,)"
    )
    args = parser.parse_args()

    load_and_evaluate(args.pred, args.gt)