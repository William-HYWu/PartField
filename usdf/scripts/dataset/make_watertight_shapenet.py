import argparse
import os

import mmint_utils
from tqdm import tqdm

from usdf.utils.watertight_utils import make_mesh_watertight


def generate_watertight_shapenet(category_dir, out_dir):
    """
    Generate watertight meshes for a shapenet category.

    Args:
        category_dir: path to the shapenet category directory
        out_dir: path to the output directory
    """
    for model_id in tqdm(os.listdir(category_dir)):
        model_dir = os.path.join(category_dir, model_id)
        input_mesh = os.path.join(model_dir, "models", "model_normalized.obj")
        output_mesh = os.path.join(out_dir, model_id + '.obj')
        make_mesh_watertight(input_mesh, output_mesh)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate watertight meshes for a shapenet category.')
    parser.add_argument("category_dir", type=str, help="Path to the shapenet category directory.")
    parser.add_argument("out_dir", type=str, help="Path to the output directory.")
    args = parser.parse_args()

    mmint_utils.make_dir(args.out_dir)

    generate_watertight_shapenet(args.category_dir, args.out_dir)
