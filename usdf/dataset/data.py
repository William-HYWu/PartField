#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import numpy as np
import os
import random
import torch
import torch.utils.data
import yaml
import trimesh
import pyrender

def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])

    return [pos_tensor, neg_tensor]

# def unpack_sdf_samples_from_ram(data, subsample=None):
#     if subsample is None:
#         return data
#     pos_tensor = data[0]
#     neg_tensor = data[1]

#     # split the sample into half
#     half = int(subsample / 2)

#     pos_size = pos_tensor.shape[0]
#     neg_size = neg_tensor.shape[0]

#     pos_start_ind = random.randint(0, pos_size - half)
#     sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

#     if neg_size <= half:
#         random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
#         sample_neg = torch.index_select(neg_tensor, 0, random_neg)
#     else:
#         neg_start_ind = random.randint(0, neg_size - half)
#         sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

#     samples = torch.cat([sample_pos, sample_neg], 0)

#     return samples


def unpack_sdf_samples(filename, subsample=None):
    npz = np.load(filename)
    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]).float())
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]).float())
    # Filter points with y > 0.4
    #pos_y_filter = pos_tensor[:, 1] > 0.4
    #neg_y_filter = neg_tensor[:, 1] > 0.4
    #pos_tensor = pos_tensor[pos_y_filter]
    #neg_tensor = neg_tensor[neg_y_filter]
    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
    
    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


def get_instance_filenames(split_name, config_path="configs/dataset/mugs_dataset_config.yaml"):
    """
    Load instance filenames from YAML config file.
    
    Args:
        data_source (str): Base directory path (not used with new config, kept for compatibility)
        split_name (str): Split name ('train', 'val', or 'test') 
        config_path (str): Path to YAML config file
        
    Returns:
        list: List of relative paths to NPZ files
    """
    # Load the YAML config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get all file paths
    all_files = [item['path'] for item in config['files']]
    
    # Calculate split indices based on ratios
    total_files = len(all_files)
    train_ratio = config['dataset']['splits']['train_ratio']
    val_ratio = config['dataset']['splits']['val_ratio']
    
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    
    # Split the files
    if split_name == 'train':
        npzfiles = all_files[:train_end]
    elif split_name == 'val':
        npzfiles = all_files[train_end:val_end]
    elif split_name == 'test':
        npzfiles = all_files[val_end:]
    else:
        raise ValueError(f"Invalid split_name: {split_name}. Must be 'train', 'val', or 'test'")
    
    # Verify files exist and log warnings for missing ones
    valid_files = []
    for npz_file in npzfiles:
        if os.path.isfile(npz_file):
            valid_files.append(npz_file)
        else:
            raise Warning(f"Requested non-existent file '{npz_file}'")

    return valid_files


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"]).float()
    neg_tensor = torch.from_numpy(npz["neg"]).float()

    return [pos_tensor, neg_tensor]

def generate_random_se2(num_samples):
    se2_matrices = []
    se2_vector = []
    if num_samples > 1:
        angles = np.random.uniform(0,  2*np.pi, num_samples)
        translations = np.random.uniform(-0.1, 0.1, (num_samples, 2))
        for angle, translation in zip(angles, translations):
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            tx, ty = translation
            se2_matrix = np.array([
                [cos_a, -sin_a, tx],
                [sin_a,  cos_a, ty],
                [0,      0,     1 ]
            ])
            se2_vector.append(np.array([tx, ty, sin_a, cos_a]))
            se2_matrices.append(se2_matrix)
        return np.array(se2_matrices), np.array(se2_vector)
    else:
        angle = np.random.uniform(-0.5*np.pi,  0.5* np.pi)
        translation = np.random.uniform(-0.1, 0.1, (2,))
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        tx, ty = translation
        se2_matrix = np.array([
            [cos_a, -sin_a, tx],
            [sin_a,  cos_a, ty],
            [0,      0,     1 ]
        ])
        return se2_matrix, np.array([tx, ty, sin_a, cos_a])


def unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data
    pos_tensor = data[0]
    neg_tensor = data[1]

    pos_y_filter = pos_tensor[:, 1] < -0.3
    neg_y_filter = neg_tensor[:, 1] < -0.3
    pos_tensor = pos_tensor[pos_y_filter]
    neg_tensor = neg_tensor[neg_y_filter]

    samples = torch.cat([pos_tensor, neg_tensor], 0)

    return samples


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        subsample,
        load_ram=False,
        random_se2_num=0,
        config_path="configs/dataset/mugs_dataset_config.yaml",
    ):
        self.subsample = subsample
        self.use_random_se2 = False
        
        # Handle both old dict-style splits and new string-based splits
        if isinstance(split, str):
            self.npyfiles = get_instance_filenames(split, config_path)

        self.load_ram = load_ram

        if load_ram:
            self.loaded_data = []
            for filename in self.npyfiles:
                npz = np.load(filename)
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]).float())
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]).float())
                self.loaded_data.append(
                    [
                        pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        neg_tensor[torch.randperm(neg_tensor.shape[0])],
                    ]
                )
        if random_se2_num > 0:
            self.use_random_se2 = True
            self.random_se2_num = random_se2_num
        _, self.se2_vector = generate_random_se2(random_se2_num)

    def __len__(self):
        return len(self.npyfiles)*self.random_se2_num if self.use_random_se2 else len(self.npyfiles)

    def __getitem__(self, idx):
        if not self.use_random_se2:
            # Handle both new relative paths and old workspace-based paths
            npz_path = self.npyfiles[idx]
            if npz_path.startswith('mugs_dataset_negate/') or npz_path.startswith('mugs_dataset/') or npz_path.startswith('mugs_dataset_oriented') or npz_path.startswith('mugs_dataset_no_orient/') or npz_path.startswith('mugs_dataset_test/')  or npz_path.startswith('mugs_dataset_test_bp/'):
                # New config format - use relative path directly
                filename = npz_path
            filename = npz_path
            if self.load_ram:
                return (
                    unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample),
                    idx,
                )
            else:
                return unpack_sdf_samples(filename, self.subsample), idx
        else:
            se2_matrix, se2_vector = generate_random_se2(num_samples=1)
            base_idx = idx // self.random_se2_num
            npz_path = self.npyfiles[base_idx]
            if npz_path.startswith('mugs_dataset_negate/') or npz_path.startswith('mugs_dataset/') or npz_path.startswith('mugs_dataset_oriented') or npz_path.startswith('mugs_dataset_no_orient/') or npz_path.startswith('mugs_dataset_test/')  or npz_path.startswith('mugs_dataset_test_bp/'):
                # New config format - use relative path directly
                filename = npz_path
            filename = npz_path
            if self.load_ram:
                samples = unpack_sdf_samples_from_ram(self.loaded_data[base_idx], self.subsample)
            else:
                samples = unpack_sdf_samples(filename, self.subsample)
            rotation = se2_matrix[:2, :2]
            translation = se2_matrix[:2, 2]

            points = samples[:, :3].numpy()
            sdf_values = samples[:, 3].numpy()

            # Apply SE(2) transformation to the xy-coordinates
            points_xy = points[:, :2]
            transformed_xy = points_xy @ rotation.T + translation
            points[:, :2] = transformed_xy

            transformed_samples = np.hstack((points, sdf_values.reshape(-1, 1)))
            transformed_samples = np.hstack((transformed_samples, np.tile(se2_vector, (transformed_samples.shape[0], 1))))
            return torch.from_numpy(transformed_samples).float(), idx

def render_partial_pointcloud(
    mesh,
    camera_position=None,
    camera_target=None,
    num_points=2048,
    fov=60,
    resolution=(640, 480),
    max_depth=2.0,
    add_noise=False,
    noise_std=0.001
):
    """
    Render a partial point cloud from a mesh as seen from a camera viewpoint using pyrender.
    
    Args:
        mesh: trimesh.Trimesh object or path to mesh file
        camera_position: (3,) array, camera position in world coords. 
                        Default: [0, 0, 1.5] (above origin)
        camera_target: (3,) array, point camera looks at. 
                      Default: [0, 0, 0] (origin)
        num_points: target number of points in output cloud
        fov: field of view in degrees
        resolution: (width, height) for rendering
        max_depth: maximum depth to render (points beyond this are discarded)
        add_noise: whether to add Gaussian noise to simulate sensor noise
        noise_std: standard deviation of Gaussian noise (in world units)
    
    Returns:
        points: (N, 3) numpy array of 3D points in world coordinates
        colors: (N, 3) numpy array of RGB colors
    
    Example:
        >>> mesh = trimesh.load("mug.ply")
        >>> points, colors = render_partial_pointcloud(
        ...     mesh, 
        ...     camera_position=[1.5, 0, 0],
        ...     num_points=2048
        ... )
    """
    # Load mesh if path is provided
    if isinstance(mesh, str):
        mesh = trimesh.load(mesh)
    
    # Set default camera parameters
    if camera_position is None:
        camera_position = np.array([0.0, 0.0, 1.5])
    else:
        camera_position = np.asarray(camera_position, dtype=np.float64)
    
    if camera_target is None:
        camera_target = np.array([0.0, 0.0, 0.0])
    else:
        camera_target = np.asarray(camera_target, dtype=np.float64)
    
    # Create pyrender scene
    scene = pyrender.Scene()
    
    # Add mesh to scene
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh_pyrender)
    
    # Setup camera
    width, height = resolution
    camera = pyrender.PerspectiveCamera(yfov=np.radians(fov), aspectRatio=width/height)
    
    # Compute camera pose (look at target)
    forward = camera_target - camera_position
    forward = forward / np.linalg.norm(forward)
    
    # Choose up vector
    world_up = np.array([0.0, 0.0, 1.0])
    if np.abs(np.dot(forward, world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])
    
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    
    up = np.cross(right, forward)
    
    # Camera pose matrix (camera to world)
    camera_pose = np.eye(4)
    camera_pose[:3, 0] = right
    camera_pose[:3, 1] = up
    camera_pose[:3, 2] = -forward  # OpenGL convention
    camera_pose[:3, 3] = camera_position
    
    scene.add(camera, pose=camera_pose)
    
    # Render depth
    renderer = pyrender.OffscreenRenderer(width, height)
    color, depth = renderer.render(scene)
    renderer.delete()
    
    # Convert depth to point cloud
    # Get valid depth pixels
    valid_mask = (depth > 0) & (depth < max_depth)
    
    if not valid_mask.any():
        print("Warning: No valid depth pixels found.")
        return np.zeros((0, 3)), np.zeros((0, 3))
    
    # Get pixel coordinates
    v, u = np.where(valid_mask)
    z = depth[valid_mask]
    
    # Compute camera intrinsics
    fx = fy = (height / 2.0) / np.tan(np.radians(fov) / 2.0)
    cx = width / 2.0
    cy = height / 2.0
    
    # Back-project to camera space
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    z_cam = z
    
    # Points in camera space
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    
    # Transform to world space
    points_world = points_cam @ camera_pose[:3, :3].T + camera_pose[:3, 3]
    
    # Get colors
    colors = color[valid_mask]
    
    # Add noise if requested
    if add_noise and noise_std > 0:
        noise = np.random.normal(0, noise_std, points_world.shape)
        points_world = points_world + noise
    
    # Downsample to target number of points
    if len(points_world) > num_points:
        indices = np.random.choice(len(points_world), num_points, replace=False)
        points_world = points_world[indices]
        colors = colors[indices]
    
    return points_world, colors


def render_multiple_views(
    mesh,
    camera_positions,
    camera_target=None,
    num_points_per_view=1024,
    **kwargs
):
    """
    Render partial point clouds from multiple camera viewpoints and combine them.
    
    Args:
        mesh: trimesh.Trimesh object or path to mesh file
        camera_positions: list of (3,) arrays, camera positions
        camera_target: (3,) array, point all cameras look at
        num_points_per_view: number of points to sample from each view
        **kwargs: additional arguments passed to render_partial_pointcloud
    
    Returns:
        points: (N, 3) combined point cloud
        colors: (N, 3) combined colors
    
    Example:
        >>> mesh = trimesh.load("mug.ply")
        >>> cameras = [[1.5, 0, 0], [-1.5, 0, 0], [0, 1.5, 0], [0, -1.5, 0]]
        >>> points, colors = render_multiple_views(mesh, cameras)
    """
    all_points = []
    all_colors = []
    
    for cam_pos in camera_positions:
        points, colors = render_partial_pointcloud(
            mesh,
            camera_position=cam_pos,
            camera_target=camera_target,
            num_points=num_points_per_view,
            **kwargs
        )
        if len(points) > 0:
            all_points.append(points)
            all_colors.append(colors)
    
    if len(all_points) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    
    combined_points = np.vstack(all_points)
    combined_colors = np.vstack(all_colors)
    
    return combined_points, combined_colors


def sample_camera_positions_sphere(
    num_views=8,
    radius=1.5,
    elevation_range=(0, 60),
    target=None
):
    """
    Sample camera positions uniformly on a sphere around the target.
    
    Args:
        num_views: number of camera positions to sample
        radius: distance from target
        elevation_range: (min, max) elevation angles in degrees
        target: (3,) array, center point to look at
    
    Returns:
        camera_positions: list of (3,) numpy arrays
    
    Example:
        >>> cameras = sample_camera_positions_sphere(num_views=8, radius=1.5)
    """
    if target is None:
        target = np.array([0.0, 0.0, 0.0])
    else:
        target = np.asarray(target)
    
    positions = []
    
    # Sample azimuth angles uniformly
    azimuths = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
    
    # Sample elevation angles
    elev_min, elev_max = elevation_range
    elevations = np.random.uniform(
        np.radians(elev_min), 
        np.radians(elev_max), 
        num_views
    )
    
    for azim, elev in zip(azimuths, elevations):
        # Spherical to Cartesian
        x = radius * np.cos(elev) * np.cos(azim)
        y = radius * np.cos(elev) * np.sin(azim)
        z = radius * np.sin(elev)
        
        position = target + np.array([x, y, z])
        positions.append(position)
    
    return positions