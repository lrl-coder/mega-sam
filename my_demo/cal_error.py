import os
import re
import csv
import glob
import torch
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')  # 非交互后端，避免无 GUI 环境报错
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def compute_camera_errors(extr_est, extr_gt):
    """
    计算相机的旋转误差和平移误差
    :param extr_est: 估计的 w2c 矩阵, shape (T, 4, 4)
    :param extr_gt: 真实的 w2c 矩阵, shape (T, 4, 4)
    :return: rot_errors (度), trans_errors (距离单位)
    """
    R_est = extr_est[:, :3, :3]
    t_est = extr_est[:, :3, 3:4]
    R_gt  = extr_gt[:, :3, :3]
    t_gt  = extr_gt[:, :3, 3:4]

    # 旋转误差
    R_rel = torch.bmm(R_est, R_gt.transpose(1, 2))
    trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    rot_errors_deg = torch.rad2deg(torch.acos(cos_theta))

    # 平移误差（相机光心 L2 距离）
    C_est = -torch.bmm(R_est.transpose(1, 2), t_est).squeeze(-1)
    C_gt  = -torch.bmm(R_gt.transpose(1, 2),  t_gt).squeeze(-1)
    trans_errors = torch.norm(C_est - C_gt, dim=-1)

    return rot_errors_deg, trans_errors


def load_and_evaluate(pred_npy_path, gt_npy_path, verbose=True):
    """
    从 npy 文件加载外参数据并计算误差。

    :param pred_npy_path: 预测外参 npy 路径，数组 shape 为 (M, 4, 4)
    :param gt_npy_path:   GT 外参 npy 路径，字典包含：
                            - 'extrinsics'       : shape (T, 4, 4)
                            - 'video_decode_frame': 真实帧索引，长度 T
    :param verbose:       是否打印逐帧误差详情
    :return: (rot_err, trans_err, frame_indices)  均为长度 T 的 Tensor/ndarray
    """
    pred_arr    = np.load(pred_npy_path)
    gt_dict     = np.load(gt_npy_path, allow_pickle=True).item()
    extr_gt_arr = gt_dict['extrinsics']
    frame_indices = np.array(gt_dict['video_decode_frame'], dtype=np.int64)

    T = len(frame_indices)
    M = pred_arr.shape[0]

    if verbose:
        print(f"  预测外参帧数 M = {M}")
        print(f"  GT  外参帧数 T = {T}")
        print(f"  GT 对应的真实帧索引: {frame_indices}")

    assert M >= T, f"预测帧数 M={M} 必须大于等于 GT 帧数 T={T}"
    assert frame_indices.max() < M, (
        f"帧索引最大值 {frame_indices.max()} 超出预测数组范围 [0, {M-1}]"
    )

    pred_selected   = pred_arr[frame_indices]
    extr_est_tensor = torch.tensor(pred_selected, dtype=torch.float64)
    extr_gt_tensor  = torch.tensor(extr_gt_arr,   dtype=torch.float64)

    rot_err, trans_err = compute_camera_errors(extr_est_tensor, extr_gt_tensor)

    if verbose:
        print("\n  ===== 逐帧误差 =====")
        for i, (fi, re, te) in enumerate(zip(frame_indices, rot_err, trans_err)):
            print(f"    GT帧 {i:3d} (预测帧索引 {fi:4d}) | "
                  f"旋转误差: {re.item():.4f} 度 | 平移误差: {te.item():.6f}")
        print(f"\n  平均旋转误差 : {rot_err.mean().item():.4f} 度")
        print(f"  中位旋转误差 : {rot_err.median().item():.4f} 度")
        print(f"  标准差旋转误差: {rot_err.std().item():.4f} 度")
        print(f"  平均平移误差 : {trans_err.mean().item():.6f}")
        print(f"  中位平移误差 : {trans_err.median().item():.6f}")
        print(f"  标准差平移误差: {trans_err.std().item():.6f}")

    return rot_err, trans_err, frame_indices


def build_gt_index(gt_dir):
    """
    扫描 GT 目录，构建 video_id -> gt_npy_path 的映射。
    GT 文件名格式: <prefix>_<video_id>_ep_<suffix>.npy
    例如: somethingsomethingv2_105211_ep_000000.npy  ->  video_id = '105211'
    """
    gt_index = {}
    for fpath in glob.glob(os.path.join(gt_dir, "*.npy")):
        fname = os.path.basename(fpath)
        m = re.search(r'_(\d+)_ep_', fname)
        if m:
            vid = m.group(1)
            if vid in gt_index:
                print(f"  [警告] video_id={vid} 存在多个GT文件，将使用: {fpath}")
            gt_index[vid] = fpath
    return gt_index


def plot_error_histograms(rot_errors, trans_errors,
                          title_prefix="", save_path=None,
                          xlabel_rot="Rotation Error (deg)",
                          xlabel_trans="Translation Error",
                          clip_percentile=None):
    """
    绘制旋转误差和平移误差的直方图

    :param rot_errors:      旋转误差数组（度），numpy array 或 torch.Tensor
    :param trans_errors:    平移误差数组，numpy array 或 torch.Tensor
    :param title_prefix:    图标题前缀（例如视频 ID 或 "ALL"）
    :param save_path:       若不为 None，则将图片保存到该路径，否则弹窗显示
    :param xlabel_rot:      左子图 x 轴标签
    :param xlabel_trans:    右子图 x 轴标签
    :param clip_percentile: 若指定（例如 95），则仅显示两个数组中均 <= 该百分位数的样本，用于去除离群点
    """
    if isinstance(rot_errors, torch.Tensor):
        rot_errors = rot_errors.cpu().numpy()
    if isinstance(trans_errors, torch.Tensor):
        trans_errors = trans_errors.cpu().numpy()

    rot_errors   = rot_errors.astype(float)
    trans_errors = trans_errors.astype(float)

    # --- 离群点裁剪 ---
    n_total = len(rot_errors)
    clip_info = ""   # 裁剪信息，写入图标题
    if clip_percentile is not None:
        thr_rot   = np.percentile(rot_errors,   clip_percentile)
        thr_trans = np.percentile(trans_errors, clip_percentile)
        mask = (rot_errors <= thr_rot) & (trans_errors <= thr_trans)
        n_removed = int((~mask).sum())
        rot_errors   = rot_errors[mask]
        trans_errors = trans_errors[mask]
        clip_info = (f" [clipped @P{clip_percentile}: "
                     f"kept {len(rot_errors)}/{n_total}, "
                     f"removed {n_removed}]")
        print(f"[Plot] Clip @P{clip_percentile}: "
              f"rot_thr={thr_rot:.4f}, trans_thr={thr_trans:.6f}, "
              f"removed {n_removed}/{n_total} samples")

    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor':   'white',
        'axes.edgecolor':   '#333333',
        'axes.linewidth':   0.8,
        'axes.grid':        True,
        'grid.color':       '#dddddd',
        'grid.linestyle':   '--',
        'grid.linewidth':   0.6,
        'xtick.direction':  'in',
        'ytick.direction':  'in',
        'xtick.color':      '#333333',
        'ytick.color':      '#333333',
        'font.size':        10,
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('white')

    def _draw_hist(ax, data, xlabel, title, bar_color, mean_color, median_color, unit=""):
        n_bins = max(50, min(60, len(data) // 3))
        ax.hist(data, bins=n_bins,
                color=bar_color, edgecolor='white', linewidth=0.5, alpha=0.82)

        mean_val   = data.mean()
        median_val = float(np.median(data))
        std_val    = data.std()

        ax.axvline(mean_val,   color=mean_color,   linewidth=1.8, linestyle='--',
                   label=f'Mean:   {mean_val:.4f}{unit}')
        ax.axvline(median_val, color=median_color, linewidth=1.8, linestyle='-.',
                   label=f'Median: {median_val:.4f}{unit}')

        stats_text = (f"N      = {len(data)}\n"
                      f"Mean   = {mean_val:.4f}{unit}\n"
                      f"Median = {median_val:.4f}{unit}\n"
                      f"Std    = {std_val:.4f}{unit}\n"
                      f"Min    = {data.min():.4f}{unit}\n"
                      f"Max    = {data.max():.4f}{unit}")
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                fontsize=8.5, verticalalignment='top', horizontalalignment='right',
                color='#222222', family='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5',
                          edgecolor='#aaaaaa', alpha=0.9))

        ax.set_title(title, fontsize=12, pad=8, fontweight='bold', color='#222222')
        ax.set_xlabel(xlabel, fontsize=10, color='#333333')
        ax.set_ylabel('Count', fontsize=10, color='#333333')
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which='minor', linestyle=':', linewidth=0.4, color='#eeeeee')
        ax.legend(fontsize=9, framealpha=0.9, edgecolor='#aaaaaa')

        # x 轴完全由数据自适应，不手动设置范围
        ax.autoscale(axis='x', tight=False)

    _draw_hist(axes[0], rot_errors,
               xlabel=xlabel_rot, title='Rotation Error Distribution',
               bar_color='#4C72B0', mean_color='#C44E52', median_color='#2CA02C',
               unit='°')
    _draw_hist(axes[1], trans_errors,
               xlabel=xlabel_trans, title='Translation Error Distribution',
               bar_color='#DD8452', mean_color='#C44E52', median_color='#2CA02C')

    prefix_str = f" — {title_prefix}" if title_prefix else ""
    fig.suptitle(f'Camera Pose Error Histograms{prefix_str}{clip_info}',
                 fontsize=13, fontweight='bold', y=1.01, color='#222222')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[图表] 直方图已保存: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def save_csv(output_dir, details_rows, summary_rows):
    """
    将逐帧明细和汇总结果分别保存为 CSV 文件。

    details.csv 列:
        video_id, gt_frame_idx, pred_frame_idx, rot_err_deg, trans_err

    summary.csv 列:
        video_id, num_frames,
        mean_rot_err_deg, median_rot_err_deg, std_rot_err_deg,
        mean_trans_err,   median_trans_err,   std_trans_err
    最后一行为全局汇总（video_id = "ALL"）。
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---------- 逐帧明细 ----------
    details_path = os.path.join(output_dir, "details.csv")
    with open(details_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "gt_frame_idx", "pred_frame_idx",
                         "rot_err_deg", "trans_err"])
        writer.writerows(details_rows)
    print(f"\n[CSV] 逐帧明细已保存: {details_path}")

    # ---------- 汇总 ----------
    summary_path = os.path.join(output_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "num_frames",
                         "mean_rot_err_deg", "median_rot_err_deg", "std_rot_err_deg",
                         "mean_trans_err",   "median_trans_err",   "std_trans_err"])
        writer.writerows(summary_rows)
    print(f"[CSV] 汇总结果已保存: {summary_path}")


def evaluate_folder(pred_dir, gt_dir, output_dir=None, plot=True):
    global _should_plot
    _should_plot = plot
    """
    批量评测：遍历预测目录中所有 *_poses.npy 文件，
    按 video_id 匹配 GT 目录中的对应文件并计算误差。
    若指定 output_dir，则将结果保存为 CSV。

    预测文件名格式: <video_id>_poses.npy
    GT    文件名格式: <prefix>_<video_id>_ep_<suffix>.npy
    """
    gt_index = build_gt_index(gt_dir)
    print(f"GT 目录共找到 {len(gt_index)} 个文件: {gt_dir}\n")

    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*_poses.npy")))
    if not pred_files:
        print(f"[错误] 预测目录中未找到 *_poses.npy 文件: {pred_dir}")
        return

    all_rot_errs    = []
    all_trans_errs  = []
    per_video_rot   = []   # 每视频均值旋转误差（直方图用）
    per_video_trans = []   # 每视频均值平移误差（直方图用）
    details_rows   = []   # 逐帧明细
    summary_rows   = []   # 每视频汇总
    matched = 0
    skipped = 0

    for pred_path in pred_files:
        fname = os.path.basename(pred_path)
        m = re.match(r'^(\d+)_poses\.npy$', fname)
        if not m:
            print(f"  [跳过] 文件名不符合格式 '<video_id>_poses.npy': {fname}")
            skipped += 1
            continue

        video_id = m.group(1)
        if video_id not in gt_index:
            print(f"  [跳过] video_id={video_id} 在GT目录中未找到对应文件")
            skipped += 1
            continue

        gt_path = gt_index[video_id]
        print(f"{'='*60}")
        print(f"video_id : {video_id}")
        print(f"预测文件 : {pred_path}")
        print(f"GT  文件 : {gt_path}")

        try:
            rot_err, trans_err, frame_indices = load_and_evaluate(
                pred_path, gt_path, verbose=True
            )

            # --- 逐帧明细行 ---
            for i, (fi, rot_e, te) in enumerate(
                    zip(frame_indices, rot_err, trans_err)):
                details_rows.append([
                    video_id,
                    i,                      # GT 帧序号（0-based）
                    int(fi),                # 对应预测帧索引
                    f"{rot_e.item():.6f}",
                    f"{te.item():.8f}",
                ])

            # --- 本视频汇总行 ---
            summary_rows.append([
                video_id,
                len(frame_indices),
                f"{rot_err.mean().item():.6f}",
                f"{rot_err.median().item():.6f}",
                f"{rot_err.std().item():.6f}",
                f"{trans_err.mean().item():.8f}",
                f"{trans_err.median().item():.8f}",
                f"{trans_err.std().item():.8f}",
            ])

            all_rot_errs.append(rot_err)
            all_trans_errs.append(trans_err)
            per_video_rot.append(rot_err.mean().item())
            per_video_trans.append(trans_err.mean().item())
            matched += 1

        except Exception as e:
            print(f"  [错误] 处理 video_id={video_id} 时发生异常: {e}")
            skipped += 1

    print(f"\n{'='*60}")
    print(f"批量评测完成: 成功 {matched} 个，跳过/失败 {skipped} 个")

    if all_rot_errs:
        all_rot_cat   = torch.cat(all_rot_errs)
        all_trans_cat = torch.cat(all_trans_errs)

        print("\n===== 全局汇总（所有视频所有帧）=====")
        print(f"  平均旋转误差 : {all_rot_cat.mean().item():.4f} 度")
        print(f"  中位旋转误差 : {all_rot_cat.median().item():.4f} 度")
        print(f"  标准差旋转误差: {all_rot_cat.std().item():.4f} 度")
        print(f"  平均平移误差 : {all_trans_cat.mean().item():.6f}")
        print(f"  中位平移误差 : {all_trans_cat.median().item():.6f}")
        print(f"  标准差平移误差: {all_trans_cat.std().item():.6f}")

        # 全局汇总追加到 summary_rows 末尾
        summary_rows.append([
            "ALL",
            len(all_rot_cat),
            f"{all_rot_cat.mean().item():.6f}",
            f"{all_rot_cat.median().item():.6f}",
            f"{all_rot_cat.std().item():.6f}",
            f"{all_trans_cat.mean().item():.8f}",
            f"{all_trans_cat.median().item():.8f}",
            f"{all_trans_cat.std().item():.8f}",
        ])

        if output_dir:
            save_csv(output_dir, details_rows, summary_rows)

        if _should_plot:
            rot_arr   = np.array(per_video_rot)
            trans_arr = np.array(per_video_trans)

            # --- 图1：全量数据 ---
            save_path_full = os.path.join(output_dir, "error_histograms.png") \
                             if output_dir else None
            plot_error_histograms(
                rot_arr, trans_arr,
                title_prefix="ALL (per-video mean)",
                save_path=save_path_full,
                xlabel_rot="Mean Rotation Error per Video (deg)",
                xlabel_trans="Mean Translation Error per Video",
            )

            # --- 图2：去除离群点（P95）---
            save_path_filt = os.path.join(output_dir, "error_histograms_filtered.png") \
                             if output_dir else None
            plot_error_histograms(
                rot_arr, trans_arr,
                title_prefix="ALL (per-video mean)",
                save_path=save_path_filt,
                xlabel_rot="Mean Rotation Error per Video (deg)",
                xlabel_trans="Mean Translation Error per Video",
                clip_percentile=95,
            )


def evaluate_single_file(pred_path, gt_path, output_dir=None, plot=True):
    """
    单文件模式：计算误差并（可选）保存 CSV。
    """
    video_id = os.path.splitext(os.path.basename(pred_path))[0]
    print(f"预测文件 : {pred_path}")
    print(f"GT  文件 : {gt_path}\n")

    rot_err, trans_err, frame_indices = load_and_evaluate(
        pred_path, gt_path, verbose=True
    )

    if output_dir:
        details_rows = []
        for i, (fi, rot_e, te) in enumerate(zip(frame_indices, rot_err, trans_err)):
            details_rows.append([
                video_id,
                i,
                int(fi),
                f"{rot_e.item():.6f}",
                f"{te.item():.8f}",
            ])

        summary_rows = [
            [
                video_id,
                len(frame_indices),
                f"{rot_err.mean().item():.6f}",
                f"{rot_err.median().item():.6f}",
                f"{rot_err.std().item():.6f}",
                f"{trans_err.mean().item():.8f}",
                f"{trans_err.median().item():.8f}",
                f"{trans_err.std().item():.8f}",
            ],
            # 单文件时全局汇总与本视频相同，单独标注
            [
                "ALL",
                len(frame_indices),
                f"{rot_err.mean().item():.6f}",
                f"{rot_err.median().item():.6f}",
                f"{rot_err.std().item():.6f}",
                f"{trans_err.mean().item():.8f}",
                f"{trans_err.median().item():.8f}",
                f"{trans_err.std().item():.8f}",
            ],
        ]
        save_csv(output_dir, details_rows, summary_rows)

    if plot:
        save_path = os.path.join(output_dir, "error_histograms.png") \
                    if output_dir else None
        plot_error_histograms(rot_err, trans_err,
                              title_prefix=video_id, save_path=save_path)

    return rot_err, trans_err


# --------- 入口 ---------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="计算预测外参与GT外参的旋转/平移误差（支持单文件或文件夹批量模式）"
    )
    parser.add_argument(
        "--pred",
        type=str, required=True,
        help="预测外参路径：单个 npy 文件（shape M×4×4）或包含 *_poses.npy 的文件夹"
    )
    parser.add_argument(
        "--gt",
        type=str, required=True,
        help="GT 外参路径：单个 npy 文件（字典）或包含 GT npy 的文件夹"
    )
    parser.add_argument(
        "--output",
        type=str, default=None,
        help="（可选）CSV 输出目录。会生成 details.csv（逐帧）和 summary.csv（汇总）。"
             "若不指定则仅打印结果。"
    )
    parser.add_argument(
        "--plot", dest="plot", action="store_true", default=True,
        help="（默认启用）绘制误差直方图。若指定 --output 则保存图片，否则弹窗显示。"
    )
    parser.add_argument(
        "--no-plot", dest="plot", action="store_false",
        help="禁用误差直方图输出。"
    )
    args = parser.parse_args()

    pred_is_dir = os.path.isdir(args.pred)
    gt_is_dir   = os.path.isdir(args.gt)

    if pred_is_dir and gt_is_dir:
        evaluate_folder(args.pred, args.gt, output_dir=args.output, plot=args.plot)
    elif (not pred_is_dir) and (not gt_is_dir):
        evaluate_single_file(args.pred, args.gt, output_dir=args.output, plot=args.plot)
    else:
        raise ValueError(
            f"--pred 和 --gt 必须同时为文件或同时为文件夹。\n"
            f"  --pred 是{'文件夹' if pred_is_dir else '文件'}\n"
            f"  --gt   是{'文件夹' if gt_is_dir  else '文件'}"
        )