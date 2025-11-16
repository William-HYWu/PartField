import os

import numpy as np
import torch.utils.data
import trimesh
from vedo import Plotter, Mesh, Points

import mmint_utils
from usdf.utils import utils, vedo_utils


class RenderDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_cfg: dict, split: str, transform=None):
        super().__init__()
        self.N_angles = dataset_cfg["N_angles"]
        self.N_surface_pointcloud = dataset_cfg["N_surface_pointcloud"]
        self.N_free_pointcloud = dataset_cfg["N_free_pointcloud"]
        self.meshes_dir = dataset_cfg["meshes_dir"]
        self.dataset_dir = dataset_cfg["dataset_dir"]
        partials_dir = os.path.join(self.dataset_dir, "partials")

        # Load split info.
        split_fn = os.path.join(self.meshes_dir, "splits", dataset_cfg["splits"][split])
        meshes = np.atleast_1d(np.loadtxt(split_fn, dtype=str))
        self.meshes = [m.replace(".obj", "") for m in meshes]
        self.meshes = np.sort(self.meshes)

        # Data arrays.
        self.mesh_idcs = []  # Index of the mesh for this example.
        self.mesh_pose = []  # Mesh pose.
        self.surface_pointclouds = []  # Surface pointclouds.
        self.free_pointclouds = []  # Free pointclouds.
        self.full_pointclouds = []  # Full pointclouds.

        # Load data.
        for mesh_idx, partial_mesh_name in enumerate(self.meshes):
            example_partial_dir = os.path.join(partials_dir, partial_mesh_name)
            for tf_idx in range(self.N_angles):
                example_angle_partial_dir = os.path.join(example_partial_dir, "tf_%d" % tf_idx)

                self.mesh_idcs.append(mesh_idx)

                info_fn = os.path.join(example_angle_partial_dir, "info.pkl.gzip")
                info_dict = mmint_utils.load_gzip_pickle(info_fn)
                self.mesh_pose.append(info_dict["w_T_o"].numpy()[0])

                surface_fn = os.path.join(example_angle_partial_dir, "pointcloud.ply")
                surface_pointcloud = utils.load_pointcloud(surface_fn)
                self.surface_pointclouds.append(surface_pointcloud)

                free_fn = os.path.join(example_angle_partial_dir, "free_pointcloud.ply")
                free_pointcloud = utils.load_pointcloud(free_fn)
                self.free_pointclouds.append(free_pointcloud)

                full_fn = os.path.join(example_angle_partial_dir, "full_pointcloud.ply")
                full_pointcloud = utils.load_pointcloud(full_fn)
                self.full_pointclouds.append(full_pointcloud)

    def __len__(self):
        return len(self.surface_pointclouds)

    def __getitem__(self, index: int):
        surface_pointcloud = self.surface_pointclouds[index]
        surface_pointcloud = surface_pointcloud[
            np.random.choice(len(surface_pointcloud), size=self.N_surface_pointcloud, replace=True)]

        free_pointcloud = self.free_pointclouds[index]
        free_pointcloud = free_pointcloud[
            np.random.choice(len(free_pointcloud), size=self.N_free_pointcloud, replace=True)]

        return {
            "mesh_idx": self.mesh_idcs[index],
            "mesh_pose": self.mesh_pose[index],
            "surface_pointcloud": surface_pointcloud,
            "free_pointcloud": free_pointcloud,
            "full_pointcloud": self.full_pointclouds[index],
        }

    def visualize_item(self, data_dict: dict):
        mesh_name = self.meshes[data_dict["mesh_idx"]]
        mesh_fn = os.path.join(self.meshes_dir, mesh_name + ".obj")
        mesh_tri = trimesh.load(mesh_fn)
        mesh_tri.apply_transform(data_dict["mesh_pose"])

        plt = Plotter((1, 1))
        plt.at(0).show(
            Mesh([mesh_tri.vertices, mesh_tri.faces], c="grey"),
            Points(data_dict["surface_pointcloud"], c="green"),
            Points(data_dict["free_pointcloud"], c="blue", alpha=0.05),
            Points(data_dict["full_pointcloud"], c="purple"),
            vedo_utils.draw_origin(0.1),
        )
        plt.close()
