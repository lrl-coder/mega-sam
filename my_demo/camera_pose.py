# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Test camera tracking on a single scene."""

# pylint: disable=invalid-name
# pylint: disable=g-importing-member
# pylint: disable=g-bad-import-order
# pylint: disable=g-import-not-at-top
# pylint: disable=redefined-outer-name
# pylint: disable=undefined-variable
# pylint: disable=undefined-loop-variable

import sys
import os

sys.path.append("base/droid_slam")
sys.path.append("my_demo/moge_depth_intrinsics")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tqdm import tqdm
import numpy as np
import torch
import cv2
import glob
import argparse
import torch.nn.functional as F
from utils_moge import MogePipeline
from droid import Droid


def image_stream(
    image_list,
    depth_list,
    scene_name,
    use_depth=False,
    K=None,
    stride=1,
):
  """image generator."""
  del scene_name, stride

  fx, fy, cx, cy = (
      K[0, 0],
      K[1, 1],
      K[0, 2],
      K[1, 2],
  )  # np.loadtxt(os.path.join(datapath, 'calibration.txt')).tolist()

  for t, (image_file) in enumerate(image_list):
    image = cv2.imread(image_file)
    # depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH) / 5000.
    # depth = np.float32(np.load(depth_file)) / 300.0
    # depth =  1. / pt_data["depth"]

    depth = depth_list[t]
    # mono_disp = mono_disp_list[t]
    # mono_disp = np.float32(np.load(disp_file)) #/ 300.0
    # depth = np.clip(
    #     1.0 / ((1.0 / aligns[2]) * (aligns[0] * mono_disp + aligns[1])),
    #     1e-4,
    #     1e4,
    # )
    depth = np.clip(depth, 1e-4, 1e4)
    depth[depth < 1e-2] = 0.0

    # breakpoint()
    h0, w0, _ = image.shape
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

    image = cv2.resize(image, (w1, h1), interpolation=cv2.INTER_AREA)
    image = image[: h1 - h1 % 8, : w1 - w1 % 8]

    # if t == 4 or t == 29:
    # imageio.imwrite("debug/camel_%d.png"%t, image[..., ::-1])

    image = torch.as_tensor(image).permute(2, 0, 1)
    # print("image ", image.shape)
    # breakpoint()

    depth = torch.as_tensor(depth)
    depth = F.interpolate(
        depth[None, None], (h1, w1), mode="nearest-exact"
    ).squeeze()
    depth = depth[: h1 - h1 % 8, : w1 - w1 % 8]

    mask = torch.ones_like(depth)

    intrinsics = torch.as_tensor([fx, fy, cx, cy])
    intrinsics[0::2] *= w1 / w0
    intrinsics[1::2] *= h1 / h0

    if use_depth:
      yield t, image[None], depth, intrinsics, mask
    else:
      yield t, image[None], intrinsics, mask


def video_stream(
    video_path,
    depth_list,
    scene_name,
    use_depth=False,
    K=None,
    stride=1,
):
  """video frame generator."""
  del scene_name
  if stride < 1:
    stride = 1

  fx, fy, cx, cy = (
      K[0, 0],
      K[1, 1],
      K[0, 2],
      K[1, 2],
  )

  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    raise ValueError("Failed to open video: {}".format(video_path))

  frame_idx = 0
  try:
    while True:
      if frame_idx >= len(depth_list):
        break

      ok, image = cap.read()
      if not ok:
        break

      if frame_idx % stride != 0:
        frame_idx += 1
        continue

      depth = depth_list[frame_idx]
      # mono_disp = mono_disp_list[frame_idx]
      # depth = np.clip(
      #     1.0 / ((1.0 / aligns[2]) * (aligns[0] * mono_disp + aligns[1])),
      #     1e-4,
      #     1e4,
      # )
      depth = np.clip(depth, 1e-4, 1e4)
      depth[depth < 1e-2] = 0.0

      h0, w0, _ = image.shape
      h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
      w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

      image = cv2.resize(image, (w1, h1), interpolation=cv2.INTER_AREA)
      image = image[: h1 - h1 % 8, : w1 - w1 % 8]

      image = torch.as_tensor(image).permute(2, 0, 1)

      depth = torch.as_tensor(depth)
      depth = F.interpolate(
          depth[None, None], (h1, w1), mode="nearest-exact"
      ).squeeze()
      depth = depth[: h1 - h1 % 8, : w1 - w1 % 8]

      mask = torch.ones_like(depth)

      intrinsics = torch.as_tensor([fx, fy, cx, cy])
      intrinsics[0::2] *= w1 / w0
      intrinsics[1::2] *= h1 / h0

      if use_depth:
        yield frame_idx, image[None], depth, intrinsics, mask
      else:
        yield frame_idx, image[None], intrinsics, mask

      frame_idx += 1
  finally:
    cap.release()


def save_poses(poses, scene_name, output_dir="outputs"):
  """Save only camera poses."""
  from pathlib import Path

  Path(output_dir).mkdir(parents=True, exist_ok=True)
  out_path = os.path.join(output_dir, "{}_poses.npy".format(scene_name))
  print(poses.shape)
  np.save(out_path, poses)
  print("Saved poses to {}".format(out_path))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--datapath")
  parser.add_argument("--weights", default="checkpoints/megasam_final.pth")
  parser.add_argument("--buffer", type=int, default=1024)
  parser.add_argument("--image_size", default=[240, 320])
  parser.add_argument("--disable_vis", default=True)

  parser.add_argument("--beta", type=float, default=0.3)
  parser.add_argument(
      "--filter_thresh", type=float, default=2.0
  )  # motion threhold for keyframe
  parser.add_argument("--warmup", type=int, default=8)
  parser.add_argument("--keyframe_thresh", type=float, default=2.0)
  parser.add_argument("--frontend_thresh", type=float, default=12.0)
  parser.add_argument("--frontend_window", type=int, default=25)
  parser.add_argument("--frontend_radius", type=int, default=2)
  parser.add_argument("--frontend_nms", type=int, default=1)

  parser.add_argument("--stereo", action="store_true")
  # --depth 参数未被实际使用：image_stream / video_stream 中 use_depth 已硬编码为 True，
  # args.depth 的值不会对运行逻辑产生任何影响。
  # parser.add_argument("--depth", action="store_true")
  parser.add_argument("--upsample", action="store_true")
  parser.add_argument("--output_dir", default="outputs")

  parser.add_argument("--backend_thresh", type=float, default=16.0)
  parser.add_argument("--backend_radius", type=int, default=2)
  parser.add_argument("--backend_nms", type=int, default=3)

  # parser.add_argument(
  #     "--mono_depth_path", default="Depth-Anything/video_visualization"
  # )
  # parser.add_argument("--metric_depth_path", default="UniDepth/outputs ")
  args = parser.parse_args()

  print("Running evaluation on {}".format(args.datapath))
  print(args)

  base = os.path.basename(args.datapath.rstrip("\\/"))
  scene_id = os.path.splitext(base)[0] if base else "scene"

  input_is_video = os.path.isfile(args.datapath)
  image_list = []
  if not input_is_video:
    image_list = sorted(
        glob.glob(os.path.join("%s" % (args.datapath), "*.jpg"))
    )
    image_list += sorted(
        glob.glob(os.path.join("%s" % (args.datapath), "*.png"))
    )
    if not image_list:
      raise ValueError("No images found in {}".format(args.datapath))

  # NOTE Mono is inverse depth, but metric-depth is depth!
  # mono_disp_paths = sorted(
  #     glob.glob(
  #         os.path.join("%s/%s" % (args.mono_depth_path, scene_id), "*.npy")
  #     )
  # )
  # metric_depth_paths = sorted(
  #     glob.glob(
  #         os.path.join("%s/%s" % (args.metric_depth_path, scene_id), "*.npz")
  #     )
  # )

  if input_is_video:
    cap = cv2.VideoCapture(args.datapath)
    ok, img_0 = cap.read()
    cap.release()
    if not ok:
      raise ValueError("Failed to read first frame from {}".format(args.datapath))
  else:
    img_0 = cv2.imread(image_list[0])
  # scales = []
  # shifts = []
  # mono_disp_list = []
  # fovs = []
  # for t, (mono_disp_file, metric_depth_file) in enumerate(
  #     zip(mono_disp_paths, metric_depth_paths)
  # ):
  #   da_disp = np.float32(np.load(mono_disp_file))  # / 300.0
  #   uni_data = np.load(metric_depth_file)
  #   metric_depth = uni_data["depth"]
  #
  #   fovs.append(uni_data["fov"])
  #
  #   da_disp = cv2.resize(
  #       da_disp,
  #       (metric_depth.shape[1], metric_depth.shape[0]),
  #       interpolation=cv2.INTER_NEAREST_EXACT,
  #   )
  #   mono_disp_list.append(da_disp)
  #   gt_disp = 1.0 / (metric_depth + 1e-8)
  #
  #   # avoid some bug from UniDepth
  #   valid_mask = (metric_depth < 2.0) & (da_disp < 0.02)
  #   gt_disp[valid_mask] = 1e-2
  #
  #   # avoid cases sky dominate entire video
  #   sky_ratio = np.sum(da_disp < 0.01) / (da_disp.shape[0] * da_disp.shape[1])
  #   if sky_ratio > 0.5:
  #     non_sky_mask = da_disp > 0.01
  #     gt_disp_ms = (
  #         gt_disp[non_sky_mask] - np.median(gt_disp[non_sky_mask]) + 1e-8
  #     )
  #     da_disp_ms = (
  #         da_disp[non_sky_mask] - np.median(da_disp[non_sky_mask]) + 1e-8
  #     )
  #     scale = np.median(gt_disp_ms / da_disp_ms)
  #     shift = np.median(gt_disp[non_sky_mask] - scale * da_disp[non_sky_mask])
  #   else:
  #     gt_disp_ms = gt_disp - np.median(gt_disp) + 1e-8
  #     da_disp_ms = da_disp - np.median(da_disp) + 1e-8
  #     scale = np.median(gt_disp_ms / da_disp_ms)
  #     shift = np.median(gt_disp - scale * da_disp)
  #
  #   gt_disp_ms = gt_disp - np.median(gt_disp) + 1e-8
  #   da_disp_ms = da_disp - np.median(da_disp) + 1e-8
  #
  #   scale = np.median(gt_disp_ms / da_disp_ms)
  #   shift = np.median(gt_disp - scale * da_disp)
  #
  #   scales.append(scale)
  #   shifts.append(shift)

  moge = MogePipeline()
  depth_list = []
  fovs = []
  if input_is_video:
    cap = cv2.VideoCapture(args.datapath)
    if not cap.isOpened():
      raise ValueError("Failed to open video: {}".format(args.datapath))
    try:
      while True:
        ok, frame = cap.read()
        if not ok:
          break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        depth, fov = moge.infer(frame_rgb)
        depth_list.append(depth)
        fovs.append(fov)
    finally:
      cap.release()
  else:
    for image_file in image_list:
      frame = cv2.imread(image_file)
      frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      depth, fov = moge.infer(frame_rgb)
      depth_list.append(depth)
      fovs.append(fov)
  print(depth_list)
  print("************** MoGe FOV ", np.median(fovs))
  ff = img_0.shape[1] / (2 * np.tan(np.radians(np.median(fovs) / 2.0)))
  K = np.eye(3)
  K[0, 0] = (
      ff * 1.0
  )  # pp_intrinsic[0]  * (img_0.shape[1] / (pp_intrinsic[1] * 2))
  K[1, 1] = (
      ff * 1.0
  )  # pp_intrinsic[0]  * (img_0.shape[0] / (pp_intrinsic[2] * 2))
  K[0, 2] = (
      img_0.shape[1] / 2.0
  )  # pp_intrinsic[1]) * (img_0.shape[1] / (pp_intrinsic[1] * 2))
  K[1, 2] = (
      img_0.shape[0] / 2.0
  )  # (pp_intrinsic[2]) * (img_0.shape[0] / (pp_intrinsic[2] * 2))

  # ss_product = np.array(scales) * np.array(shifts)
  # med_idx = np.argmin(np.abs(ss_product - np.median(ss_product)))
  #
  # align_scale = scales[med_idx]  # np.median(np.array(scales))
  # align_shift = shifts[med_idx]  # np.median(np.array(shifts))
  # normalize_scale = (
  #     np.percentile((align_scale * np.array(mono_disp_list) + align_shift), 98)
  #     / 2.0
  # )
  #
  # aligns = (align_scale, align_shift, normalize_scale)

  if input_is_video:
    stream = video_stream(
        args.datapath,
        depth_list,
        scene_id,
        use_depth=True,
        K=K,
    )
  else:
    stream = image_stream(
        image_list,
        depth_list,
        scene_id,
        use_depth=True,
        K=K,
    )

  for t, image, depth, intrinsics, mask in tqdm(stream):
    if not args.disable_vis:
      show_image(image[0])

    # breakpoint()
    if t == 0:
      args.image_size = [image.shape[2], image.shape[3]]
      droid = Droid(args)

    droid.track(t, image, depth, intrinsics=intrinsics, mask=mask)

  # last frame
  droid.track_final(t, image, depth, intrinsics=intrinsics, mask=mask)

  if input_is_video:
    term_stream = video_stream(
        args.datapath,
        depth_list,
        scene_id,
        use_depth=True,
        K=K,
    )
  else:
    term_stream = image_stream(
        image_list,
        depth_list,
        scene_id,
        use_depth=True,
        K=K,
    )

  traj_est, depth_est, motion_prob = droid.terminate(
      term_stream,
      _opt_intr=True,
      full_ba=True,
      scene_name=scene_id,
  )

  save_poses(traj_est, scene_id, output_dir=args.output_dir)
