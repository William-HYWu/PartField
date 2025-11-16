import os

import numpy as np
import torch.utils.data
import trimesh
import transforms3d as tf3d

import mmint_utils
from vedo import Plotter, Points, Mesh

from usdf.utils import vedo_utils, utils


class SDFDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_cfg: dict, split: str, transform=None):
        super().__init__()
        self.N_angles = dataset_cfg["N_angles"]
        self.meshes_dir = dataset_cfg["meshes_dir"]
        self.dataset_dir = dataset_cfg["dataset_dir"]
        self.N_sdf = dataset_cfg["N_sdf"]
        self.N_pc = dataset_cfg.get("N_pc", 10000)
        self.has_point_clouds = dataset_cfg.get("has_point_clouds", False)
        self.balance_semantics = dataset_cfg.get("balance_semantics", False)
        partials_dir = os.path.join(self.dataset_dir, "partials")
        sdfs_dir = os.path.join(self.dataset_dir, "sdfs")

        # Load split info.
        split_fn = os.path.join(self.dataset_dir, "splits", dataset_cfg["splits"][split])
        meshes = np.atleast_1d(np.loadtxt(split_fn, dtype=str))
        self.meshes = [m.replace(".obj", "") for m in meshes]

        # Data arrays.
        self.example_idcs = []  # Index of the partial view.
        self.mesh_idcs = []  # Index of the mesh for this example.
        self.query_points = []  # Query points for SDF.
        self.sdf = []  # SDF values at query points.
        self.object_pose = []  # Object pose.
        self.partial_pointcloud = []  # Partial pointcloud.
        self.angle = []  # Angle of the example.

        # Load data.
        for mesh_idx, partial_mesh_name in enumerate(self.meshes):
            example_sdf_dir = os.path.join(sdfs_dir, partial_mesh_name)
            example_partial_dir = os.path.join(partials_dir, partial_mesh_name)
            for angle_idx in range(self.N_angles):
                example_angle_sdf_dir = os.path.join(example_sdf_dir, "angle_%d" % angle_idx)
                example_angle_partial_dir = os.path.join(example_partial_dir, "angle_%d" % angle_idx)

                sdf_fn = os.path.join(example_angle_sdf_dir, "sdf_data.pkl.gzip")
                sdf_data = mmint_utils.load_gzip_pickle(sdf_fn)

                if self.has_point_clouds:
                    partial_fn = os.path.join(example_angle_partial_dir, "pointcloud.ply")
                    partial_pointcloud = utils.load_pointcloud(partial_fn)
                    self.partial_pointcloud.append(partial_pointcloud)

                # Get angle of the example from object pose.
                object_pose = sdf_data["object_pose"]
                ex, ey, ez = tf3d.euler.mat2euler(object_pose[:3, :3], axes="sxyz")
                self.angle.append(ez)  # Save just ez.

                # Append to data arrays.
                self.example_idcs.append(mesh_idx * self.N_angles + angle_idx)
                self.mesh_idcs.append(mesh_idx)
                self.query_points.append(sdf_data["query_points"])
                self.sdf.append(sdf_data["sdf_values"])
                self.object_pose.append(object_pose)

    def __len__(self):
        return len(self.example_idcs)

    def get_num_objects(self):
        return len(self.meshes)

    def __getitem__(self, index: int):
        # Balance number of positive and negative samples in query points.
        query_points = self.query_points[index]
        sdf = self.sdf[index]
        if self.balance_semantics:
            pos_idx = np.random.choice(np.where(sdf > 0)[0], self.N_sdf // 2, replace=False)
            neg_idx = np.random.choice(np.where(sdf <= 0)[0], self.N_sdf // 2, replace=False)
            query_points = query_points[np.concatenate([pos_idx, neg_idx])]
            sdf = sdf[np.concatenate([pos_idx, neg_idx])]

        data_dict = {
            "example_idx": self.example_idcs[index],
            "mesh_idx": self.mesh_idcs[index],
            "mesh_pose": self.object_pose[index],
            "query_points": query_points,
            "sdf": sdf,
            "angle": self.angle[index],
        }

        if self.has_point_clouds:
            pc = self.partial_pointcloud[index]

            # Sample points from partial pointcloud. TODO: Clean this up.
            if pc.shape[0] < self.N_pc:
                # Sample points from pointcloud to pad it.
                pc = np.concatenate([pc, pc[np.random.choice(pc.shape[0], self.N_pc - pc.shape[0])]])

            data_dict["partial_pointcloud"] = pc

        return data_dict

    def visualize_item(self, data_dict: dict):
        # Load mesh for this example.
        mesh_name = self.meshes[data_dict["mesh_idx"] // self.N_angles]
        mesh_fn = os.path.join(self.meshes_dir, mesh_name + ".obj")
        rot_mesh = trimesh.load(mesh_fn)
        rot_mesh.apply_transform(data_dict["mesh_pose"])

        plt = Plotter((1, 2))
        plt.at(0).show(Mesh([rot_mesh.vertices, rot_mesh.faces], c="y"),
                       Points(data_dict["partial_pointcloud"], c="b") if self.has_point_clouds else None,
                       vedo_utils.draw_origin(scale=0.1))
        plt.at(1).show(Points(data_dict["query_points"][data_dict["sdf"] < 0.0]),
                       vedo_utils.draw_origin(scale=0.1))
        plt.interactive().close()
