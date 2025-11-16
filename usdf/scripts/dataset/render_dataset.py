import argparse
import os
from multiprocessing import Pool
from typing import List

import mmint_utils
import numpy as np
import pyrender
import torch
import trimesh
import transforms3d as tf3d
from tqdm import tqdm
from usdf.utils import utils, vedo_utils
from usdf.utils.render_utils import depth_to_pointcloud, depth_to_free_points
from vedo import Plotter, Points, Mesh
import pytorch_volumetric as pv
import pytorch_kinematics as pk


class RenderDataset:

    def __init__(self, meshes_dir: str, dataset_dir: str, partials_dir: str, n_angles: int, meshes: List[str]):
        self.meshes_dir = meshes_dir
        self.dataset_dir = dataset_dir
        self.partials_dir = partials_dir
        self.N_angles = n_angles
        self.meshes = meshes

    def __call__(self, mesh_idx: int):
        mesh_fn = self.meshes[mesh_idx]
        dtype = torch.float

        # Build scene.
        scene = pyrender.Scene()

        # Add camera to the scene.
        yfov = np.pi / 2.0
        camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = tf3d.euler.euler2mat(np.pi / 2.0, 0, 0, axes="sxyz")
        camera_pose[:3, 3] = [0, -2.0, 0]
        scene.add(camera, pose=camera_pose)
        camera_to_world_tf = pk.Transform3d(matrix=torch.tensor(camera_pose, dtype=dtype))

        # Add light to the scene.
        light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi / 16.0,
                                   outerConeAngle=np.pi / 6.0)
        scene.add(light, pose=camera_pose)

        # Add renderer.
        r = pyrender.OffscreenRenderer(256, 256)

        mesh_path = os.path.join(self.meshes_dir, mesh_fn)
        obj = pv.MeshObjectFactory(mesh_path)
        mesh_tri: trimesh.Trimesh = trimesh.load(mesh_path)
        mesh = pyrender.Mesh.from_trimesh(mesh_tri)
        mesh_node = pyrender.Node(mesh=mesh, matrix=np.eye(4))
        scene.add_node(mesh_node)

        mesh_partials_path = os.path.join(self.partials_dir, mesh_fn[:-4])
        mmint_utils.make_dir(mesh_partials_path)

        for tf_idx, angle in enumerate(np.linspace(0, 2 * np.pi, self.N_angles + 1)[:-1]):
            # TODO: Option to do completely random rotations.
            w_T_o_tf = pk.RotateAxisAngle(angle, "Z", dtype=dtype, degrees=False)
            # object_pose = np.eye(4)
            # object_pose[:3, :3] = tf3d.euler.euler2mat(0, 0, angle, axes="sxyz")
            scene.set_pose(mesh_node, w_T_o_tf.get_matrix().cpu().numpy()[0])

            # Render the scene.
            color, depth = r.render(scene)

            # Convert depth to pointcloud.
            pointcloud = depth_to_pointcloud(depth, yfov)[:, :3]
            pointcloud = utils.transform_pointcloud(pointcloud, camera_pose)

            # Recover free points.
            free_pointcloud_camera = depth_to_free_points(depth, yfov, max_depth=3.0, n=200)[:, :3]
            free_pointcloud_world = camera_to_world_tf.transform_points(
                torch.tensor(free_pointcloud_camera, dtype=dtype))

            # Downsample free points.
            # First, remove points far from object.
            free_pointcloud_world = free_pointcloud_world[
                np.linalg.norm(free_pointcloud_world, axis=1) < 1.05
                ]

            # Next, voxel downsample.
            free_pointcloud_world = pv.voxel_down_sample(free_pointcloud_world, 0.02)

            # Sample points everywhere on the mesh.
            full_pointcloud_o = np.asarray(mesh_tri.sample(10000))
            full_pointcloud_w = w_T_o_tf.transform_points(torch.tensor(full_pointcloud_o, dtype=dtype))

            # Save partials.
            mesh_partials_angle_path = os.path.join(mesh_partials_path, "tf_%d" % tf_idx)
            mmint_utils.make_dir(mesh_partials_angle_path)
            utils.save_pointcloud(pointcloud, os.path.join(mesh_partials_angle_path, "pointcloud.ply"))
            utils.save_pointcloud(free_pointcloud_world,
                                  os.path.join(mesh_partials_angle_path, "free_pointcloud.ply"))
            utils.save_pointcloud(full_pointcloud_w,
                                  os.path.join(mesh_partials_angle_path, "full_pointcloud.ply"))
            mmint_utils.save_gzip_pickle({"w_T_o": w_T_o_tf.get_matrix()},
                                         os.path.join(mesh_partials_angle_path, "info.pkl.gzip"))

        scene.remove_node(mesh_node)


def render_dataset_parallel(dataset_cfg: dict, split: str, vis: bool = False):
    if not vis:
        os.environ["PYOPENGL_PLATFORM"] = "egl"

    dtype = torch.float
    free_space_surface_epsilon = 0.01

    meshes_dir = dataset_cfg["meshes_dir"]
    dataset_dir = dataset_cfg["dataset_dir"]
    N_angles = dataset_cfg["N_angles"]  # Number of angles to render from.
    mmint_utils.make_dir(dataset_dir)

    partials_dir = os.path.join(dataset_dir, "partials")
    mmint_utils.make_dir(partials_dir)

    # Load split info.
    split_fn = os.path.join(meshes_dir, "splits", dataset_cfg["splits"][split])
    meshes = np.atleast_1d(np.loadtxt(split_fn, dtype=str))

    # Parallelize rendering.
    with Pool(16) as pool:
        for _ in tqdm(
                pool.imap(RenderDataset(meshes_dir, dataset_dir, partials_dir, N_angles, meshes), range(len(meshes))),
                total=len(meshes)):
            pass


def render_dataset(dataset_cfg: dict, split: str, vis: bool = False):
    if not vis:
        os.environ["PYOPENGL_PLATFORM"] = "egl"

    dtype = torch.float
    free_space_surface_epsilon = 0.01

    meshes_dir = dataset_cfg["meshes_dir"]
    dataset_dir = dataset_cfg["dataset_dir"]
    N_angles = dataset_cfg["N_angles"]  # Number of angles to render from.
    mmint_utils.make_dir(dataset_dir)

    partials_dir = os.path.join(dataset_dir, "partials")
    mmint_utils.make_dir(partials_dir)

    # Load split info.
    split_fn = os.path.join(meshes_dir, "splits", dataset_cfg["splits"][split])
    meshes = np.atleast_1d(np.loadtxt(split_fn, dtype=str))

    # Build scene.
    scene = pyrender.Scene()

    # Add camera to the scene.
    yfov = np.pi / 2.0
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = tf3d.euler.euler2mat(np.pi / 2.0, 0, 0, axes="sxyz")
    camera_pose[:3, 3] = [0, -2.0, 0]
    scene.add(camera, pose=camera_pose)
    camera_to_world_tf = pk.Transform3d(matrix=torch.tensor(camera_pose, dtype=dtype))

    # Add light to the scene.
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi / 16.0, outerConeAngle=np.pi / 6.0)
    scene.add(light, pose=camera_pose)

    # Add renderer.
    r = pyrender.OffscreenRenderer(256, 256)

    with tqdm(total=len(meshes) * N_angles) as pbar:
        for mesh_fn in meshes:
            mesh_path = os.path.join(meshes_dir, mesh_fn)
            obj = pv.MeshObjectFactory(mesh_path)
            mesh_tri: trimesh.Trimesh = trimesh.load(mesh_path)
            mesh = pyrender.Mesh.from_trimesh(mesh_tri)
            mesh_node = pyrender.Node(mesh=mesh, matrix=np.eye(4))
            scene.add_node(mesh_node)

            mesh_partials_path = os.path.join(partials_dir, mesh_fn[:-4])
            mmint_utils.make_dir(mesh_partials_path)

            for tf_idx, angle in enumerate(np.linspace(0, 2 * np.pi, N_angles + 1)[:-1]):
                # TODO: Option to do completely random rotations.
                w_T_o_tf = pk.RotateAxisAngle(angle, "Z", dtype=dtype, degrees=False)
                # object_pose = np.eye(4)
                # object_pose[:3, :3] = tf3d.euler.euler2mat(0, 0, angle, axes="sxyz")
                scene.set_pose(mesh_node, w_T_o_tf.get_matrix().cpu().numpy()[0])

                # Render the scene.
                color, depth = r.render(scene)

                # Convert depth to pointcloud.
                pointcloud = depth_to_pointcloud(depth, yfov)[:, :3]
                pointcloud = utils.transform_pointcloud(pointcloud, camera_pose)

                # Recover free points.
                free_pointcloud_camera = depth_to_free_points(depth, yfov, max_depth=3.0, n=200)[:, :3]
                free_pointcloud_world = camera_to_world_tf.transform_points(
                    torch.tensor(free_pointcloud_camera, dtype=dtype))

                # Downsample free points.
                # First, remove points far from object.
                free_pointcloud_world = free_pointcloud_world[
                    np.linalg.norm(free_pointcloud_world, axis=1) < 1.05
                    ]

                # Next, voxel downsample.
                free_pointcloud_world = pv.voxel_down_sample(free_pointcloud_world, 0.02)

                # Sample points everywhere on the mesh.
                full_pointcloud_o = np.asarray(mesh_tri.sample(10000))
                full_pointcloud_w = w_T_o_tf.transform_points(torch.tensor(full_pointcloud_o, dtype=dtype))

                if vis:
                    mesh_tri_rot = mesh_tri.copy().apply_transform(w_T_o_tf.get_matrix().cpu().numpy()[0])

                    plt = Plotter()
                    plt.at(0).show(
                        Points(pointcloud[:, :3], c="green"),
                        Mesh([mesh_tri_rot.vertices, mesh_tri_rot.faces], alpha=0.5),
                        vedo_utils.draw_origin(0.1),
                        Points(free_pointcloud_world[:, :3], c="blue", alpha=0.01),
                        Points(full_pointcloud_w[:, :3], c="purple")
                    )
                    plt.close()

                # Save partials.
                mesh_partials_angle_path = os.path.join(mesh_partials_path, "tf_%d" % tf_idx)
                mmint_utils.make_dir(mesh_partials_angle_path)
                utils.save_pointcloud(pointcloud, os.path.join(mesh_partials_angle_path, "pointcloud.ply"))
                utils.save_pointcloud(free_pointcloud_world,
                                      os.path.join(mesh_partials_angle_path, "free_pointcloud.ply"))
                utils.save_pointcloud(full_pointcloud_w, os.path.join(mesh_partials_angle_path, "full_pointcloud.ply"))
                mmint_utils.save_gzip_pickle({"w_T_o": w_T_o_tf.get_matrix()},
                                             os.path.join(mesh_partials_angle_path, "info.pkl.gzip"))

                pbar.update(1)

            scene.remove_node(mesh_node)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Render dataset.")
    parser.add_argument("dataset_cfg", type=str, help="Dataset configuration file.")
    parser.add_argument("split", type=str, help="Split to render.")
    parser.add_argument("-v", "--vis", action="store_true", help="Visualize pointclouds.")
    parser.set_defaults(vis=False)
    args = parser.parse_args()

    dataset_cfg_ = mmint_utils.load_cfg(args.dataset_cfg)["data"][args.split]

    render_dataset(dataset_cfg_, args.split, args.vis)
