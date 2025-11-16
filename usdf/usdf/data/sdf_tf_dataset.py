import os
from multiprocessing import Pool

import numpy as np
import torch.utils.data
import trimesh
import transforms3d as tf3d

import mmint_utils
from vedo import Plotter, Points, Mesh

from usdf.utils import vedo_utils, utils


class MeshDataLoader:

    def __init__(self, sdfs_dir):
        self.sdfs_dir = sdfs_dir

    def __call__(self, partial_mesh_name):
        example_sdf_dir = os.path.join(self.sdfs_dir, partial_mesh_name)

        sdf_fn = os.path.join(example_sdf_dir, "sdf_data.pkl.gzip")
        sdf_data = mmint_utils.load_gzip_pickle(sdf_fn)

        return sdf_data["query_points"], sdf_data["sdf_values"]


class SDFTFDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_cfg: dict, split: str, transform=None):
        # TODO: Load partial views.
        super().__init__()
        self.meshes_dir = dataset_cfg["meshes_dir"]
        self.dataset_dir = dataset_cfg["dataset_dir"]
        self.N_sdf = dataset_cfg["N_sdf"]
        self.N_pc = dataset_cfg.get("N_pc", 10000)
        self.N_transforms = dataset_cfg.get("N_transforms", 1)  # Per mesh.
        self.has_point_clouds = dataset_cfg.get("has_point_clouds", False)
        self.balance_semantics = dataset_cfg.get("balance_semantics", False)
        partials_dir = os.path.join(self.dataset_dir, "partials")
        sdfs_dir = os.path.join(self.dataset_dir, "sdfs")

        # Load split info.
        split_fn = os.path.join(self.dataset_dir, "splits", dataset_cfg["splits"][split])
        meshes = np.atleast_1d(np.loadtxt(split_fn, dtype=str))
        self.meshes = [m.replace(".obj", "") for m in meshes]

        # Mesh data arrays.
        self.mesh_idcs = []  # Index of the mesh for this example.
        self.query_points = []  # Query points for SDF.
        self.sdf = []  # SDF values at query points.

        # Load data.
        load_mesh_data = MeshDataLoader(sdfs_dir)
        with Pool(16) as p:
            results = p.map(load_mesh_data, self.meshes)

        # Append to data arrays.
        for mesh_idx, result in enumerate(results):
            query_points, sdf = result
            self.mesh_idcs.append(mesh_idx)
            self.query_points.append(query_points)
            self.sdf.append(sdf)

        # Load transformations.
        tfs_fn = os.path.join(self.dataset_dir, "transforms.pkl.gzip")
        self.transforms = mmint_utils.load_gzip_pickle(tfs_fn)

    def get_num_objects(self):
        return len(self.meshes)

    def __len__(self):
        return len(self.transforms)

    def __getitem__(self, index: int):
        # Get transformation information.
        transform_dict = self.transforms[index]
        example_idx = transform_dict["example_idx"]  # Should be same as index.
        mesh_idx = transform_dict["mesh_idx"]
        transform_idx = transform_dict["transform_idx"]
        transform = transform_dict["transform"]
        scale = transform_dict["scale"]

        # Balance number of positive and negative samples in query points.
        query_points = utils.transform_pointcloud(self.query_points[mesh_idx], transform)
        sdf = self.sdf[mesh_idx] * scale
        if self.balance_semantics:
            pos_idx = np.random.choice(np.where(sdf > 0)[0], self.N_sdf // 2, replace=False)
            neg_idx = np.random.choice(np.where(sdf <= 0)[0], self.N_sdf // 2, replace=False)
            query_points = query_points[np.concatenate([pos_idx, neg_idx])]
            sdf = sdf[np.concatenate([pos_idx, neg_idx])]

        data_dict = {
            "example_idx": example_idx,
            "mesh_idx": mesh_idx,
            "transform_idx": transform_idx,
            "query_points": query_points,
            "sdf": sdf,
            "mesh_pose": transform,
        }

        if self.has_point_clouds:
            # TODO: Load partial point clouds.
            pass

        return data_dict

    def visualize_item(self, data_dict: dict):
        # Load mesh for this example.
        mesh_name = self.meshes[data_dict["mesh_idx"]]
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
