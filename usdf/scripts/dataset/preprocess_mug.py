import argparse
import os

import mmint_utils
import numpy as np
import trimesh
from tqdm import tqdm
from usdf.utils import vedo_utils
from vedo import Plotter, Mesh, Sphere


def preprocess_mug(mug_dir: str, out_dir: str, center: bool = False, vis: bool = False):
    """
    Preprocess mug data. Reorient to consistent coordinate system.
    """
    mug_fns = [f for f in os.listdir(mug_dir) if f.endswith(".obj")]
    mmint_utils.make_dir(out_dir)

    for mug_fn in tqdm(mug_fns):
        mug_path = os.path.join(mug_dir, mug_fn)
        mesh: trimesh.Trimesh = trimesh.load(mug_path)

        # Correct orientation.
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))

        # Center mesh.
        if center:
            mesh_center = np.mean(np.array(mesh.bounds), axis=0)
            mesh.apply_transform(trimesh.transformations.translation_matrix(-mesh_center))

        # Scale mesh to fit in unit sphere.
        r = np.linalg.norm(mesh.vertices, axis=1).max()
        scale = (1.0 / 1.03) / r
        mesh.apply_transform(trimesh.transformations.scale_matrix(scale))

        # Visualize.
        if vis:
            plt = Plotter()
            mesh_vedo = Mesh([mesh.vertices, mesh.faces])
            plt.at(0).show(mesh_vedo, Sphere(alpha=0.2), vedo_utils.draw_origin(0.2))
            plt.interactive().close()

        # Save resulting mesh.
        out_path = os.path.join(out_dir, mug_fn)
        mesh.export(out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess mug data.")
    parser.add_argument("mug_dir", type=str, help="Path to mug data.")
    parser.add_argument("out_dir", type=str, help="Path to output directory.")
    parser.add_argument("--center", action="store_true", help="Center the mesh.")
    parser.add_argument("--vis", action="store_true", help="Visualize the results.")
    parser.set_defaults(vis=False)
    parser.set_defaults(center=False)
    args = parser.parse_args()

    preprocess_mug(args.mug_dir, args.out_dir, args.center, args.vis)
