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

"""Test camera tracking on a single video or a folder of videos."""

# pylint: disable=invalid-name
# pylint: disable=g-importing-member
# pylint: disable=g-bad-import-order
# pylint: disable=g-import-not-at-top
# pylint: disable=redefined-outer-name
# pylint: disable=undefined-variable
# pylint: disable=undefined-loop-variable

import sys
import os
import glob

sys.path.append("base/droid_slam")
sys.path.append("my_demo/moge_depth_intrinsics")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lietorch import SE3
from tqdm import tqdm
import numpy as np
import torch
import cv2
import argparse
import torch.nn.functional as F
from utils_moge import MogePipeline
from droid import Droid

# Supported video extensions
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v")


def video_stream(video_path, depth_list, scene_name, use_depth=False, K=None, stride=1):
  """Video frame generator."""
  del scene_name
  if stride < 1:
    stride = 1

  fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

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
  poses_th = torch.as_tensor(poses, device="cpu")
  w2c = SE3(poses_th).matrix().numpy()
  out_path = os.path.join(output_dir, "{}_poses.npy".format(scene_name))
  print(w2c.shape)
  np.save(out_path, w2c)
  print("Saved poses to {}".format(out_path))


def process_video(video_path, args, moge):
  """Run depth inference + DROID-SLAM tracking on a single video file."""
  scene_id = os.path.splitext(os.path.basename(video_path))[0]
  print("\n========== Processing: {} ==========".format(video_path))

  # Read first frame to determine image size for intrinsics
  cap = cv2.VideoCapture(video_path)
  ok, img_0 = cap.read()
  cap.release()
  if not ok:
    print("[WARN] Failed to read first frame, skipping: {}".format(video_path))
    return

  # Run MoGe depth inference on all video frames
  depth_list = []
  fovs = []
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    print("[WARN] Failed to open video, skipping: {}".format(video_path))
    return
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

  if not depth_list:
    print("[WARN] No frames decoded, skipping: {}".format(video_path))
    return

  print("************** MoGe FOV ", np.median(fovs))
  ff = img_0.shape[1] / (2 * np.tan(np.radians(np.median(fovs) / 2.0)))
  K = np.eye(3)
  K[0, 0] = ff
  K[1, 1] = ff
  K[0, 2] = img_0.shape[1] / 2.0
  K[1, 2] = img_0.shape[0] / 2.0

  # Tracking pass
  stream = video_stream(video_path, depth_list, scene_id, use_depth=True, K=K)
  droid = None
  for t, image, depth, intrinsics, mask in tqdm(stream, desc="Tracking"):
    if t == 0:
      args.image_size = [image.shape[2], image.shape[3]]
      droid = Droid(args)
    droid.track(t, image, depth, intrinsics=intrinsics, mask=mask)

  if droid is None:
    print("[WARN] No frames tracked, skipping: {}".format(video_path))
    return

  # Last frame
  droid.track_final(t, image, depth, intrinsics=intrinsics, mask=mask)

  # Termination / BA pass
  term_stream = video_stream(video_path, depth_list, scene_id, use_depth=True, K=K)
  traj_est, depth_est, motion_prob = droid.terminate(
      term_stream,
      _opt_intr=True,
      full_ba=True,
      scene_name=scene_id,
  )

  save_poses(traj_est, scene_id, output_dir=args.output_dir)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--datapath",
      required=True,
      help="Path to a single video file OR a folder containing multiple videos.",
  )
  parser.add_argument("--weights", default="checkpoints/megasam_final.pth")
  parser.add_argument("--buffer", type=int, default=1024)
  parser.add_argument("--image_size", default=[240, 320])
  parser.add_argument("--disable_vis", default=True)

  parser.add_argument("--beta", type=float, default=0.3)
  parser.add_argument("--filter_thresh", type=float, default=2.0)
  parser.add_argument("--warmup", type=int, default=8)
  parser.add_argument("--keyframe_thresh", type=float, default=2.0)
  parser.add_argument("--frontend_thresh", type=float, default=12.0)
  parser.add_argument("--frontend_window", type=int, default=25)
  parser.add_argument("--frontend_radius", type=int, default=2)
  parser.add_argument("--frontend_nms", type=int, default=1)

  parser.add_argument("--stereo", action="store_true")
  parser.add_argument("--upsample", action="store_true")
  parser.add_argument("--output_dir", default="outputs")

  parser.add_argument("--backend_thresh", type=float, default=16.0)
  parser.add_argument("--backend_radius", type=int, default=2)
  parser.add_argument("--backend_nms", type=int, default=3)

  args = parser.parse_args()

  print("Running on: {}".format(args.datapath))
  print(args)

  # Collect video paths
  datapath = args.datapath.rstrip("\\/")
  if os.path.isfile(datapath):
    # Single video file
    video_paths = [datapath]
  elif os.path.isdir(datapath):
    # Folder: collect all supported video files (non-recursive)
    video_paths = sorted([
        p for p in glob.glob(os.path.join(datapath, "*"))
        if os.path.isfile(p) and p.lower().endswith(VIDEO_EXTENSIONS)
    ])
    if not video_paths:
      raise ValueError(
          "No supported video files found in folder: {}".format(datapath)
      )
    print("Found {} video(s) in folder.".format(len(video_paths)))
  else:
    raise ValueError("--datapath must be a video file or a directory: {}".format(datapath))

  # Load MoGe once and reuse across all videos
  moge = MogePipeline()

  for i, video_path in enumerate(video_paths):
    print("\n[{}/{}] {}".format(i + 1, len(video_paths), video_path))
    try:
      process_video(video_path, args, moge)
    except Exception as e:  # pylint: disable=broad-except
      print("[ERROR] Failed to process {}: {}".format(video_path, e))
      continue

  print("\nAll done. Results saved to: {}".format(args.output_dir))
