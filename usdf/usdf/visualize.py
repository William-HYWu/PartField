from typing import List

import numpy as np
import trimesh
from vedo import Plotter, Mesh, Points


def visualize_mesh(data_dict: dict, mesh: trimesh.Trimesh, gt_mesh: trimesh.Trimesh):
    vis_partial = "surface_pointcloud" in data_dict
    plt = Plotter(shape=(2, 1))
    plt.at(0).show(
        Mesh([gt_mesh.vertices, gt_mesh.faces]),
        Points(data_dict["surface_pointcloud"], c="b") if vis_partial else None,
        Points(data_dict["free_pointcloud"], c="r", alpha=0.05) if vis_partial else None,
    )
    plt.at(1).show(
        Mesh([mesh.vertices, mesh.faces]),
        Points(data_dict["surface_pointcloud"], c="b") if vis_partial else None,
        Points(data_dict["free_pointcloud"], c="r", alpha=0.05) if vis_partial else None,
    )
    plt.interactive().close()


def visualize_mesh_set(data_dict: dict, mesh_set: List[trimesh.Trimesh], gt_mesh: trimesh.Trimesh,
                       metadata_mesh_set: dict):
    vis_partial = "surface_pointcloud" in data_dict
    num_meshes = len(mesh_set)
    plot_shape = int(np.ceil(np.sqrt(num_meshes + 1)))
    final_loss = metadata_mesh_set["final_loss"]

    plt = Plotter(shape=(plot_shape, plot_shape))
    plt.at(0).show(
        "Ground Truth",
        Mesh([gt_mesh.vertices, gt_mesh.faces]),
        Points(data_dict["surface_pointcloud"], c="b") if vis_partial else None,
        Points(data_dict["free_pointcloud"], c="r", alpha=0.05) if vis_partial else None,
    )
    for i, mesh in enumerate(mesh_set):
        plt.at(i + 1).show(
            f"Mesh {i} (Loss: {final_loss[0, i].item():.4f})",
            Mesh([mesh.vertices, mesh.faces]),
            Points(data_dict["surface_pointcloud"], c="b") if vis_partial else None,
            Points(data_dict["free_pointcloud"], c="r", alpha=0.05) if vis_partial else None,
        )
    plt.interactive().close()
