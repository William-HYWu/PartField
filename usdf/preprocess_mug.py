import argparse
import os

import mmint_utils
import numpy as np
import trimesh
from tqdm import tqdm
from usdf.utils import vedo_utils
from vedo import Plotter, Mesh, Sphere


def preprocess_mug(mug_dir: str, out_dir: str, center: bool = False, vis: bool = False, num_orientations: int = 1, vis_orientations: bool = False):
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

        # --- Start: Multi-orientation processing ---

        # Visualize original corrected mesh if requested
        if vis:
            print(f"Visualizing original orientation for {mug_fn}")
            plt = Plotter(N=1, title="Original Orientation")
            mesh_vedo = Mesh([mesh.vertices, mesh.faces])
            plt.at(0).show(mesh_vedo, Sphere(alpha=0.2), vedo_utils.draw_origin(0.2))
            plt.interactive().close()

        # Generate and save multiple orientations
        if num_orientations > 1:
            angle_increment = 2 * np.pi / num_orientations
            all_oriented_meshes = []

            for i in range(num_orientations):
                oriented_mesh = mesh.copy()
                angle = i * angle_increment
                
                # Get the mesh center to rotate around the object's own axis
                mesh_center = oriented_mesh.bounds.mean(axis=0)
                
                # Z-axis rotation in trimesh is around [0, 1, 0]
                rotation = trimesh.transformations.rotation_matrix(
                    angle, [0, 0, 1], point=mesh_center
                )
                oriented_mesh.apply_transform(rotation)
                
                all_oriented_meshes.append(oriented_mesh)

                # Save resulting mesh with orientation index
                base_name, ext = os.path.splitext(mug_fn)
                out_fn = f"{base_name}_orient_{i}{ext}"
                out_path = os.path.join(out_dir, out_fn)
                oriented_mesh.export(out_path)

            # Visualize all orientations together if requested
            if vis_orientations:
                print(f"Visualizing all {num_orientations} orientations for {mug_fn}")
                plt = Plotter(N=1, title=f"{num_orientations} Orientations")
                vedo_meshes = []
                for i, m in enumerate(all_oriented_meshes):
                    # Apply a small offset to each mesh to see them all
                    offset = np.array([i * 0.1, 0, 0]) 
                    vedo_mesh = Mesh([m.vertices + offset, m.faces]).color(i)
                    vedo_meshes.append(vedo_mesh)
                
                plt.at(0).show(vedo_meshes, Sphere(alpha=0.1), vedo_utils.draw_origin(0.5))
                plt.interactive().close()

        else:
            # Save single resulting mesh if no multi-orientation
            out_path = os.path.join(out_dir, mug_fn)
            mesh.export(out_path)
        # --- End: Multi-orientation processing ---


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess mug data.")
    parser.add_argument("mug_dir", type=str, help="Path to mug data.")
    parser.add_argument("out_dir", type=str, help="Path to output directory.")
    parser.add_argument("--center", action="store_true", help="Center the mesh.")
    parser.add_argument("--vis", action="store_true", help="Visualize the results.")
    parser.add_argument("--num_orientations", type=int, default=1, help="Number of orientations to generate.")
    parser.add_argument("--vis_orientations", action="store_true", help="Visualize all generated orientations together.")
    parser.set_defaults(vis=False)
    parser.set_defaults(center=False)
    parser.set_defaults(vis_orientations=False)
    args = parser.parse_args()

    preprocess_mug(args.mug_dir, args.out_dir, args.center, args.vis, args.num_orientations, args.vis_orientations)
