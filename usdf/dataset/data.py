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
import torchvision

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

def load_image(image_path):
    image = torchvision.io.read_image(image_path)
    if image.shape[0] == 4:
        image = image[:3]
    image = image.float() / 255.0  # Normalize to [0, 1]
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = transform(image)
    return image

def load_points(point_path):
    npz = np.load(point_path)
    points = torch.from_numpy(npz).float()
    return points


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


class PartFieldDataset(torch.utils.data.Dataset):
    def __init__(self, data_root="data"):
        self.renders_dir = os.path.join(data_root, "renders")
        self.points_dir = os.path.join(data_root, "partfield_batch_0")
        self.data_pairs = []

        if os.path.exists(self.renders_dir):
            for obj_id in os.listdir(self.renders_dir):
                # Check if it is a directory
                if not os.path.isdir(os.path.join(self.renders_dir, obj_id)):
                    continue
                
                img_path = os.path.join(self.renders_dir, obj_id, f"{obj_id}_view00.png")
                point_path = os.path.join(self.points_dir, f"part_feat_coord_{obj_id}_0_batch.npy")

                if os.path.exists(img_path) and os.path.exists(point_path):
                    self.data_pairs.append((img_path, point_path))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        img_path, point_path = self.data_pairs[idx]
        img = load_image(img_path)
        pts_features = load_points(point_path)
        
        xyz = pts_features[:, :3]
        features = pts_features[:, 3:]
        
        return img, xyz, features