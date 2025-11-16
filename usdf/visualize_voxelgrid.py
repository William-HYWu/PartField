#!/usr/bin/env python3
"""Visualize a saved voxelgrid NPZ file produced by utils.mesh.save_voxelgrid

Usage:
    python scripts/visualize_voxelgrid.py path/to/voxelgrid.npz --downsample 2
"""
import argparse
import numpy as np
from vedo import Plotter, Points, Box


def visualize_voxelgrid(npz_path, downsample=1, point_size=5):
    data = np.load(npz_path)
    sdf = data['sdf']
    sign_mask = data['sign_mask']
    origin = data['origin']
    voxel_size = float(data['voxel_size'])

    # Get voxel coordinates
    N = sdf.shape[0]
    xs, ys, zs = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing='ij')
    coords = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3)

    # Convert to world coordinates
    coords_world = origin + coords * voxel_size

    sign_flat = sign_mask.reshape(-1)

    # Optionally downsample
    if downsample > 1:
        coords_world = coords_world[::downsample]
        sign_flat = sign_flat[::downsample]

    inside_points = coords_world[sign_flat == 1]
    outside_points = coords_world[sign_flat == 0]

    plt = Plotter(title=f"Voxelgrid: {npz_path}")

    if len(outside_points):
        p_out = Points(outside_points, r=point_size).alpha(0.2).c('blue')
        plt.add(p_out)
    if len(inside_points):
        p_in = Points(inside_points, r=point_size).alpha(0.8).c('red')
        plt.add(p_in)

    # Draw bounding box
    bbox = Box(pos=origin + np.array([N, N, N]) * voxel_size / 2.0, length=N*voxel_size)
    bbox.alpha(0.05).c('grey')
    plt.add(bbox)

    plt.show(interactive=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('npz', type=str, help='Path to the voxelgrid npz file')
    parser.add_argument('--downsample', type=int, default=1, help='Downsample stride for visualization')
    parser.add_argument('--point_size', type=int, default=5, help='Point size')
    args = parser.parse_args()

    visualize_voxelgrid(args.npz, args.downsample, args.point_size)
