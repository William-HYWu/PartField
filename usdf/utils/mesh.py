#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import numpy as np
import plyfile
import skimage.measure
import time
import torch

def decode_sdf(decoder, latent_vector, queries):
    num_samples = queries.shape[0]

    if latent_vector is None:
        inputs = queries
    else:
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], 1)

    sdf = decoder.inference(inputs)

    return sdf

def decode_sdf_from_se2(decoder, latent_vector, queries):
    num_samples = queries.shape[0]

    if latent_vector is None:
        inputs = queries
    else:
        latent_repeat = latent_vector.expand(num_samples, -1)

    sdf = decoder(latent_repeat, queries)[0]

    return sdf

def create_mesh(
    decoder, latent_vec, filename, N=256, max_batch=32 ** 3, offset=None, scale=None
):
    start = time.time()
    ply_filename = filename

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

        samples[head : min(head + max_batch, num_samples), 3] = (
            decode_sdf(decoder, latent_vec, sample_subset)
            .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    # Enforce positive SDF values outside unit sphere
    # Convert tensors to numpy arrays for processing
    sdf_values_np = sdf_values.numpy() if isinstance(sdf_values, torch.Tensor) else sdf_values
    x_coords_np = samples[:, 0].numpy() if isinstance(samples, torch.Tensor) else samples[:, 0]
    y_coords_np = samples[:, 1].numpy() if isinstance(samples, torch.Tensor) else samples[:, 1]
    z_coords_np = samples[:, 2].numpy() if isinstance(samples, torch.Tensor) else samples[:, 2]
    #revised
    # Reshape coordinates
    x_coords = x_coords_np.reshape(N, N, N)
    y_coords = y_coords_np.reshape(N, N, N)
    z_coords = z_coords_np.reshape(N, N, N)
    
    # Calculate distances from origin
    distances = np.sqrt(x_coords**2 + y_coords**2 + z_coords**2)
    
    # Find points outside unit sphere with negative SDF
    outside_sphere = distances > 1.0
    positive_sdf = sdf_values_np > 0
    violations = outside_sphere & positive_sdf
    
    if np.any(violations):
        num_violations = np.sum(violations)
        print(f"Found {num_violations} positive SDF values outside unit sphere, fixing...")
        # Set violations to a small positive value, e.g., their distance to the sphere surface
        sdf_values_np[violations] = 1.0 - distances[violations]
        
        # Update the original sdf_values
        if isinstance(sdf_values, torch.Tensor):
            sdf_values = torch.from_numpy(sdf_values_np)
        else:
            sdf_values = sdf_values_np

    end = time.time()
    print("sampling takes: %f" % (end - start))

    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
    )

def create_mesh_from_se2_vector(
    decoder, latent_vec, filename, N=256, max_batch=32 ** 3, offset=None, scale=None
):
    start = time.time()
    ply_filename = filename

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

        samples[head : min(head + max_batch, num_samples), 3] = (
            decode_sdf_from_se2(decoder, latent_vec, sample_subset)
            .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    # Enforce positive SDF values outside unit sphere
    # Convert tensors to numpy arrays for processing
    sdf_values_np = sdf_values.numpy() if isinstance(sdf_values, torch.Tensor) else sdf_values
    x_coords_np = samples[:, 0].numpy() if isinstance(samples, torch.Tensor) else samples[:, 0]
    y_coords_np = samples[:, 1].numpy() if isinstance(samples, torch.Tensor) else samples[:, 1]
    z_coords_np = samples[:, 2].numpy() if isinstance(samples, torch.Tensor) else samples[:, 2]
    #revised
    # Reshape coordinates
    x_coords = x_coords_np.reshape(N, N, N)
    y_coords = y_coords_np.reshape(N, N, N)
    z_coords = z_coords_np.reshape(N, N, N)
    
    # Calculate distances from origin
    distances = np.sqrt(x_coords**2 + y_coords**2 + z_coords**2)
    
    # Find points outside unit sphere with negative SDF
    outside_sphere = distances > 1.0
    positive_sdf = sdf_values_np > 0
    violations = outside_sphere & positive_sdf
    
    if np.any(violations):
        num_violations = np.sum(violations)
        print(f"Found {num_violations} positive SDF values outside unit sphere, fixing...")
        # Set violations to a small positive value, e.g., their distance to the sphere surface
        sdf_values_np[violations] = 1.0 - distances[violations]
        
        # Update the original sdf_values
        if isinstance(sdf_values, torch.Tensor):
            sdf_values = torch.from_numpy(sdf_values_np)
        else:
            sdf_values = sdf_values_np

    end = time.time()
    print("sampling takes: %f" % (end - start))

    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
    )

def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    print("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)


def detect_negative_sdf_outside_sphere(sdf_values, voxel_grid_origin, voxel_size, sphere_radius=1.0):
    """
    Detect negative SDF values outside the unit sphere.
    
    Args:
        sdf_values: numpy array of shape (N, N, N) containing SDF values
        voxel_grid_origin: list of three floats [x, y, z] representing grid origin
        voxel_size: float, size of each voxel
        sphere_radius: float, radius of the sphere to check outside of (default: 1.0)
    
    Returns:
        dict containing:
            - 'violations': number of negative SDF values outside sphere
            - 'total_outside': total number of points outside sphere
            - 'violation_ratio': ratio of violations to total outside points
            - 'violation_positions': array of (x, y, z) positions where violations occur
            - 'violation_sdf_values': array of SDF values at violation positions
            - 'max_violation': most negative SDF value outside sphere
            - 'mean_violation': mean SDF value of violations
    """
    N = sdf_values.shape[0]
    
    # Create coordinate grids
    x = np.linspace(voxel_grid_origin[0], voxel_grid_origin[0] + (N-1)*voxel_size, N)
    y = np.linspace(voxel_grid_origin[1], voxel_grid_origin[1] + (N-1)*voxel_size, N)
    z = np.linspace(voxel_grid_origin[2], voxel_grid_origin[2] + (N-1)*voxel_size, N)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Calculate distances from origin
    distances = np.sqrt(X**2 + Y**2 + Z**2)
    
    # Find points outside the sphere
    outside_sphere = distances > sphere_radius
    
    # Find negative SDF values outside the sphere
    negative_sdf = sdf_values < 0
    violations = outside_sphere & negative_sdf
    
    # Get violation positions and values
    violation_indices = np.where(violations)
    violation_positions = np.column_stack([
        X[violation_indices], 
        Y[violation_indices], 
        Z[violation_indices]
    ])
    violation_sdf_values = sdf_values[violation_indices]
    violation_distances = distances[violation_indices]
    
    # Calculate statistics
    num_violations = np.sum(violations)
    total_outside = np.sum(outside_sphere)
    violation_ratio = num_violations / total_outside if total_outside > 0 else 0
    
    results = {
        'violations': int(num_violations),
        'total_outside': int(total_outside),
        'violation_ratio': float(violation_ratio),
        'violation_positions': violation_positions,
        'violation_sdf_values': violation_sdf_values,
        'violation_distances': violation_distances,
        'max_violation': float(np.min(violation_sdf_values)) if num_violations > 0 else 0.0,
        'mean_violation': float(np.mean(violation_sdf_values)) if num_violations > 0 else 0.0,
        'sphere_radius': sphere_radius
    }
    
    return results


def print_sdf_violation_report(violation_results):
    """
    Print a detailed report of SDF violations outside the unit sphere.
    
    Args:
        violation_results: dict returned by detect_negative_sdf_outside_sphere
    """
    print("\n" + "="*60)
    print("SDF VIOLATION REPORT")
    print("="*60)
    
    sphere_radius = violation_results['sphere_radius']
    violations = violation_results['violations']
    total_outside = violation_results['total_outside']
    violation_ratio = violation_results['violation_ratio']
    
    print(f"Sphere radius: {sphere_radius:.2f}")
    print(f"Points outside sphere: {total_outside:,}")
    print(f"Negative SDF violations outside sphere: {violations:,}")
    print(f"Violation ratio: {violation_ratio:.4f} ({violation_ratio*100:.2f}%)")
    
    if violations > 0:
        max_violation = violation_results['max_violation']
        mean_violation = violation_results['mean_violation']
        violation_distances = violation_results['violation_distances']
        
        print(f"\nViolation Statistics:")
        print(f"  Most negative SDF value: {max_violation:.6f}")
        print(f"  Mean violation SDF value: {mean_violation:.6f}")
        print(f"  Min distance from origin: {np.min(violation_distances):.6f}")
        print(f"  Max distance from origin: {np.max(violation_distances):.6f}")
        print(f"  Mean distance from origin: {np.mean(violation_distances):.6f}")
        
        # Show worst violations
        worst_indices = np.argsort(violation_results['violation_sdf_values'])[:min(5, violations)]
        print(f"\nWorst {len(worst_indices)} violations:")
        for i, idx in enumerate(worst_indices):
            pos = violation_results['violation_positions'][idx]
            sdf_val = violation_results['violation_sdf_values'][idx]
            dist = violation_results['violation_distances'][idx]
            print(f"  {i+1}. Position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
                  f"SDF: {sdf_val:.6f}, Distance: {dist:.3f}")
    else:
        print(f"\n✅ No SDF violations detected outside sphere radius {sphere_radius}!")
    
    print("="*60)


def analyze_sdf_grid_properties(sdf_values, voxel_grid_origin, voxel_size):
    """
    Analyze various properties of the SDF grid including violations outside unit sphere.
    
    Args:
        sdf_values: numpy array of shape (N, N, N) containing SDF values
        voxel_grid_origin: list of three floats [x, y, z] representing grid origin
        voxel_size: float, size of each voxel
    
    Returns:
        dict containing comprehensive analysis results
    """
    print("Analyzing SDF grid properties...")
    
    # Basic statistics
    sdf_min = np.min(sdf_values)
    sdf_max = np.max(sdf_values)
    sdf_mean = np.mean(sdf_values)
    sdf_std = np.std(sdf_values)
    
    # Zero-level set statistics
    zero_crossings = np.sum(np.abs(sdf_values) < 1e-6)
    negative_values = np.sum(sdf_values < 0)
    positive_values = np.sum(sdf_values > 0)
    
    # Check violations outside unit sphere
    violation_results = detect_negative_sdf_outside_sphere(
        sdf_values, voxel_grid_origin, voxel_size, sphere_radius=1.0
    )
    
    # Compile comprehensive results
    analysis = {
        'basic_stats': {
            'min': float(sdf_min),
            'max': float(sdf_max),
            'mean': float(sdf_mean),
            'std': float(sdf_std),
            'total_points': int(sdf_values.size)
        },
        'value_distribution': {
            'negative_count': int(negative_values),
            'positive_count': int(positive_values),
            'zero_crossings': int(zero_crossings),
            'negative_ratio': float(negative_values / sdf_values.size),
            'positive_ratio': float(positive_values / sdf_values.size)
        },
        'sphere_violations': violation_results,
        'grid_info': {
            'shape': sdf_values.shape,
            'origin': voxel_grid_origin,
            'voxel_size': voxel_size,
            'extent': [voxel_grid_origin[i] + (sdf_values.shape[i]-1)*voxel_size 
                      for i in range(3)]
        }
    }
    
    return analysis


def check_sdf_mesh_quality(decoder, latent_vec, N=128, sphere_radius=1.0, verbose=True):
    """
    Check the quality of SDF values generated by a decoder, specifically looking
    for violations outside the unit sphere.
    
    Args:
        decoder: trained SDF decoder model
        latent_vec: latent vector for the shape
        N: grid resolution (default: 128 for faster analysis)
        sphere_radius: radius to check outside of (default: 1.0)
        verbose: whether to print detailed report
    
    Returns:
        dict containing analysis results
    """
    print(f"Checking SDF mesh quality with grid resolution {N}x{N}x{N}...")
    
    decoder.eval()
    
    # Create sampling grid
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)
    
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    z = np.linspace(-1, 1, N)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    # Evaluate SDF
    points_tensor = torch.from_numpy(points).float()
    
    if latent_vec is not None:
        latent_repeat = latent_vec.expand(points_tensor.shape[0], -1)
        inputs = torch.cat([latent_repeat, points_tensor], 1)
    else:
        inputs = points_tensor
    
    with torch.no_grad():
        if inputs.is_cuda or next(decoder.parameters()).is_cuda:
            inputs = inputs.cuda()
        sdf_pred = decoder(inputs).squeeze().cpu().numpy()
    
    sdf_values = sdf_pred.reshape(N, N, N)
    
    # Analyze the SDF grid
    analysis = analyze_sdf_grid_properties(sdf_values, voxel_origin, voxel_size)
    
    if verbose:
        print_sdf_violation_report(analysis['sphere_violations'])
        
        print(f"\nBasic SDF Statistics:")
        print(f"  Min SDF: {analysis['basic_stats']['min']:.6f}")
        print(f"  Max SDF: {analysis['basic_stats']['max']:.6f}")
        print(f"  Mean SDF: {analysis['basic_stats']['mean']:.6f}")
        print(f"  Std SDF: {analysis['basic_stats']['std']:.6f}")
        print(f"  Negative values: {analysis['value_distribution']['negative_count']:,} "
              f"({analysis['value_distribution']['negative_ratio']*100:.1f}%)")
    
    return analysis
