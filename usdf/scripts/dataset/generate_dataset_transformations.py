import argparse
import os.path

import numpy as np
import torch
from transformations import random_rotation_matrix
import pytorch_kinematics as pk

import mmint_utils


def generate_dataset_transformations(dataset_cfg: dict, split: str):
    dataset_dir = dataset_cfg["dataset_dir"]
    meshes_dir = dataset_cfg["meshes_dir"]
    n_transforms = dataset_cfg["N_transforms"]  # Per mesh.
    translation_bound = np.array(dataset_cfg["translation_bound"])
    scale_bounds = dataset_cfg["scale_bounds"]
    z_rotate = dataset_cfg["z_rotate"]  # Only sample z rotations.
    dtype = torch.float

    tfs_fn = os.path.join(dataset_dir, "transforms.pkl.gzip")

    # Load split info.
    split_fn = os.path.join(meshes_dir, "splits", dataset_cfg["splits"][split])
    meshes = np.atleast_1d(np.loadtxt(split_fn, dtype=str))
    meshes = [m.replace(".obj", "") for m in meshes]

    # Transforms.
    transforms = []

    for mesh_idx in range(len(meshes)):
        mesh_name = meshes[mesh_idx]

        # Generate random transformations.
        for transform_idx in range(n_transforms):
            # Translation.
            translation = np.random.uniform(-translation_bound, translation_bound, size=3)

            # Rotation: random rotation matrix.
            if z_rotate:
                angle = transform_idx * 2 * np.pi / n_transforms
                rotation = pk.RotateAxisAngle(angle, "Z", dtype=dtype, degrees=False).get_matrix()[0, :3, :3]
            else:
                rotation = random_rotation_matrix()

            # Get scale.
            scale = np.random.uniform(scale_bounds[0], scale_bounds[1])
            rotation *= scale

            # Build full transformation matrix.
            transform = np.eye(4)
            transform[:3, :3] = rotation[:3, :3]
            transform[:3, 3] = translation

            # Build transform dict.
            transform_dict = {
                "example_idx": mesh_idx * n_transforms + transform_idx,
                "mesh_idx": mesh_idx,
                "mesh_name": mesh_name,
                "transform_idx": transform_idx,
                "transform": transform,
                "scale": scale,
            }
            transforms.append(transform_dict)

    # Save transforms.
    mmint_utils.save_gzip_pickle(transforms, tfs_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate dataset transformations.")
    parser.add_argument("dataset_cfg_fn", type=str, help="Path to dataset config file.")
    parser.add_argument("split", type=str, help="Split to generate.")
    args = parser.parse_args()

    dataset_cfg_ = mmint_utils.load_cfg(args.dataset_cfg_fn)["data"][args.split]
    generate_dataset_transformations(dataset_cfg_, args.split)
