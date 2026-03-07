import os
import numpy as np
import cv2
from PIL import Image
from utils_moge import MogePipeline
from moge.utils.vis import colorize_depth


def main():
    # 初始化模型
    model = MogePipeline()
    image_path = "../example_images/02_Office.jpg"

    if not os.path.exists(image_path):
        print(f"找不到图片，请检查路径: {image_path}")
        return

    # 读取图片
    raw_image = Image.open(image_path).convert('RGB')
    image = np.array(raw_image)

    # 执行推理
    depth, fov = model.infer(image, resolution_level=1)

    # 打印结果
    print(f'Depth: {depth.shape if hasattr(depth, "shape") else depth}')
    print(f'Fov: {fov}')

    # 保存深度图可视化结果
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    depth_vis_path = os.path.join(output_dir, f"{base_name}_depth_vis.png")
    
    depth_vis = colorize_depth(depth)
    cv2.imwrite(depth_vis_path, cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))
    print(f"深度图可视化已保存至: {depth_vis_path}")


if __name__ == '__main__':
    main()