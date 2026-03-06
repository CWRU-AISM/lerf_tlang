"""Render LERF with text query for segmentation overlay."""
import json
import sys
import torch
from pathlib import Path

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.camera_paths import get_path_from_json
from nerfstudio.utils import colormaps

from PIL import Image
import numpy as np


def main():
    config_path = Path("outputs/ramen/lerf/2026-03-01_191211/config.yml")
    camera_path_file = Path("/home/aism/seg_models/lerf_ovs/ramen/camera_paths/single_frame.json")
    output_dir = Path("renders/ramen/segmentation")
    query = sys.argv[1] if len(sys.argv) > 1 else "egg"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    _, pipeline, _, _ = eval_setup(config_path, eval_num_rays_per_chunk=4096)
    pipeline.eval()

    # Set the text query for LERF segmentation
    pipeline.model.image_encoder.set_positives([query])
    print(f"Set LERF query to: '{query}'")

    # Load camera path
    with open(camera_path_file, "r") as f:
        camera_path_data = json.load(f)
    camera_path = get_path_from_json(camera_path_data)

    # Render each camera
    for i in range(len(camera_path)):
        camera = camera_path[i:i+1].to(pipeline.device)
        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera(camera)

        # Save RGB
        rgb = (outputs["rgb"].cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(rgb).save(output_dir / f"{i:05d}_rgb.png")
        print(f"  Saved {i:05d}_rgb.png")

        # Save relevancy map
        if "relevancy_0" in outputs:
            rel_colored = colormaps.apply_colormap(
                outputs["relevancy_0"],
                colormaps.ColormapOptions(colormap="turbo"),
            )
            rel_img = (rel_colored.cpu().numpy() * 255).astype(np.uint8)
            # Remove darkest pixels: set pixels below brightness threshold to black
            brightness = np.mean(rel_img, axis=-1)
            dark_mask = brightness < 80  # 0-255, adjust to remove more/less
            rel_img[dark_mask] = 0
            Image.fromarray(rel_img).save(output_dir / f"{i:05d}_relevancy_{query}.png")
            print(f"  Saved {i:05d}_relevancy_{query}.png")

        # Save composited (segmentation overlaid on RGB)
        if "composited_0" in outputs:
            comp = (outputs["composited_0"].cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(comp).save(output_dir / f"{i:05d}_composited_{query}.png")
            print(f"  Saved {i:05d}_composited_{query}.png")

    print(f"\nDone! Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
