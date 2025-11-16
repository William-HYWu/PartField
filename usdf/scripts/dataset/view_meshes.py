import argparse
import os

import mmint_utils
import numpy as np
import trimesh


def view_meshes(meshes_dir: str, split: str = None):
    # Load split info.
    if split is not None:
        split_fn = os.path.join(meshes_dir, "splits", split + ".txt")
        mesh_fns = np.loadtxt(split_fn, dtype=str)
    else:
        mesh_fns = [f for f in os.listdir(meshes_dir) if ".obj" in f]

    for mesh_fn in mesh_fns:
        print(mesh_fn)

        mesh_path = os.path.join(meshes_dir, mesh_fn)
        mesh = trimesh.load(mesh_path)
        mesh.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Render dataset.")
    parser.add_argument("meshes_dir", type=str, help="Meshes directory.")
    parser.add_argument("--split", "-s", type=str, default=None, required=False, help="Split to render.")
    args = parser.parse_args()

    view_meshes(args.meshes_dir, args.split)
