import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import torch

from moge.model.v2 import MoGeModel as MoGeModelV2


class MogePipeline:
    """
    Inference pipeline for MoGeModelV2 to estimate the horizontal Field of View (FoV).
    """

    def __init__(
            self,
            model_name: str = "Ruicheng/moge-2-vitl",
            device: torch.device = torch.device("cuda")
    ):
        """
        Initializes the pipeline and loads the MoGe model.

        Args:
            model_name (str): Path or name of the pre-trained MoGe model.
            device (torch.device): Device to load the model onto (e.g., 'cuda').
        """
        self.device = device
        self.model = MoGeModelV2.from_pretrained(model_name).to(device)

    def infer(self, input_image: np.ndarray, resolution_level: int = 1):
        """
        Args:
            input_image (np.ndarray): The input image (H, W, 3) in RGB format.
            resolution_level (int): 0-9, higher = more tokens, more VRAM. Default 1.

        Returns:
            depth      (np.ndarray, H x W): Absolute (metric) depth map in metres.
            fov_x_deg  (float)            : Horizontal FoV in degrees.
        """

        input_tensor = torch.tensor(
            input_image / 255.0,
            dtype=torch.float32,
            device=self.device
        ).permute(2, 0, 1)

        # resolution_level 越高精度越好但显存占用越大
        output = self.model.infer(input_tensor, resolution_level=resolution_level)

        depth = output['depth'].cpu().numpy()
        intrinsics = output['intrinsics'].cpu().numpy()

        # Calculate horizontal FoV: FoV_x = 2 * arctan(cx / fx)
        fov_x_rad = 2 * np.arctan(intrinsics[0, 2] / intrinsics[0, 0])
        fov_x_deg = np.rad2deg(fov_x_rad)

        return depth, fov_x_deg