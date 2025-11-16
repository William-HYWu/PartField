import argparse
import os
import numpy as np
import trimesh
from tqdm import tqdm
import vedo
import pickle


def generate_sdf_data(mesh_path, number_of_points = 100000, noise_scale=0.0025):
    surface_sample_count = int(number_of_points * 47 / 50) // 2
    mesh = trimesh.load(mesh_path)
    
    # Convert Scene to single Trimesh if needed
    if isinstance(mesh, trimesh.Scene):
        print(f"Mesh contains {len(mesh.geometry)} geometries, combining them...")
        mesh = mesh.dump(concatenate=True)
    
    query_points = []
    points = mesh.sample(number_of_points)
    indices = np.random.choice(points.shape[0], surface_sample_count)
    random_surface_points = points[indices, :]
    query_points.append(random_surface_points + np.random.normal(scale=0.0025, size=(surface_sample_count, 3)))
    query_points.append(random_surface_points + np.random.normal(scale=0.00025, size=(surface_sample_count, 3)))

    unit_sphere_sample_count = number_of_points - surface_sample_count * 2
    
    # Explicit unit sphere sampling using spherical coordinates
    # Generate random radii (cube root for uniform volume distribution)
    r = np.random.random(unit_sphere_sample_count) ** (1/3)
    # Generate random angles
    theta = np.random.uniform(0, 2*np.pi, unit_sphere_sample_count)  # azimuthal angle
    phi = np.arccos(1 - 2*np.random.random(unit_sphere_sample_count))  # polar angle
    
    # Convert to Cartesian coordinates
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    unit_sphere_points = np.column_stack([x, y, z])
    
    query_points.append(unit_sphere_points)
    query_points = np.concatenate(query_points, axis=0)

    points = query_points
    
    # Compute SDF values
    sdf = trimesh.proximity.signed_distance(mesh, points)
    
    return points, sdf, mesh


def visualize_sdf_data_vedo(points, sdf, mesh=None, show_mesh=True):
    """
    Visualize SDF data using Vedo.
    
    Args:
        points (np.ndarray): Query points.
        sdf (np.ndarray): SDF values for each point.
        mesh (trimesh.Trimesh, optional): The mesh to display.
        show_mesh (bool): Whether to show the mesh in the visualization.
    """
    print("\n=== Vedo Visualization ===")
    
    # Create a plotter
    plt = vedo.Plotter(title="SDF Visualization")
    
    # Color points discretely using a binary mask and colormap: 0->blue, 1->red
    mask = (sdf > 0).astype(int)
    point_cloud = vedo.Points(points)
    point_cloud.cmap(['blue', 'red'], mask, on='points').point_size(4)
    
    # Add the point cloud to the plotter
    plt.add(point_cloud.legend("SDF Points"))
    
    # Add mesh to the plotter if provided
    if mesh is not None and show_mesh:
        # Convert trimesh to Vedo mesh
        vedo_mesh = vedo.Mesh([mesh.vertices, mesh.faces])
        vedo_mesh.c('gray').alpha(0.3) # Set color and transparency
        plt.add(vedo_mesh.legend("Original Mesh"))
        
    print("Showing interactive plot...")
    # Show the plot. The axes are automatically generated to fit the data.
    plt.show(axes=1)


def visualize_sdf_data(points, sdf, mesh=None, show_mesh=True):
    
    # Create colors for visualization
    colors = np.zeros(points.shape)
    colors[sdf < 0, 2] = 1  # Blue for outside (negative SDF)
    colors[sdf > 0, 0] = 1  # Red for inside (positive SDF)
    
    # Create scene
    scene = trimesh.Scene()
    
    # Add point cloud
    point_cloud = trimesh.PointCloud(points, colors=colors)
    scene.add_geometry(point_cloud, node_name='sdf_points')
    
    # Add mesh if provided
    if mesh is not None and show_mesh:
        scene.add_geometry(mesh, node_name='mesh')
    
    # Show visualization
    scene.show()

def save_sdf_data(points, sdf, mesh, output_path, mesh_name):
    output_path = os.path.join(output_path, mesh_name)
    os.makedirs(output_path, exist_ok=True)
    base_path = os.path.join(output_path, mesh_name)
    # Create pos/neg labels and masks based on SDF sign
    pos_mask = sdf > 0
    neg_mask = ~pos_mask
    # Concatenate points and sdf into a single (N, 4) array: [x, y, z, sdf]
    points_sdf = np.concatenate([points, sdf.reshape(-1, 1)], axis=1)
    payload = {
        'pos': points_sdf[pos_mask], 
        'neg': points_sdf[neg_mask], 
    }
    np.savez(f"{base_path}_sdf.npz", **payload)
    print(f"Saved SDF data to: {base_path}_sdf.npz")
    
    point_cloud = trimesh.PointCloud(points)
    point_cloud.export(f"{base_path}_points.ply")
    print(f"Saved points to: {base_path}_points.ply")
    
    mesh.export(f"{base_path}_mesh.obj")
    print(f"Saved mesh to: {base_path}_mesh.obj")


def process_single_mesh(mesh_path, args):
    mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
    print(f"\nProcessing: {mesh_name}")
    
    if args.num_samples > 1:
        for sample_idx in range(args.num_samples):
            print(f"\n--- Sample {sample_idx + 1}/{args.num_samples} ---")
            points, sdf, mesh = generate_sdf_data(
                mesh_path, 
                number_of_points=args.num_points,
                noise_scale=args.noise_scale
            )
            
            print(f"Generated {len(points)} points with SDF values")
            print(f"SDF range: [{sdf.min():.4f}, {sdf.max():.4f}]")
            # Save data if requested
            if args.save:
                output_dir = os.path.join(args.output_dir, f"sample_{sample_idx}")
                save_sdf_data(points, sdf, mesh, output_dir, mesh_name)
            # Visualize if requested
            if args.visualize:
                if args.vedo:
                    visualize_sdf_data_vedo(points, sdf, mesh=None, show_mesh=False)
                else:
                    visualize_sdf_data(points, sdf, mesh, args.show_mesh)
    else:
        points, sdf, mesh = generate_sdf_data(
            mesh_path, 
            number_of_points=args.num_points,
            noise_scale=args.noise_scale
        )
        
        print(f"Generated {len(points)} points with SDF values")
        print(f"SDF range: [{sdf.min():.4f}, {sdf.max():.4f}]")
        
        # Save data if requested
        if args.save:
            save_sdf_data(points, sdf, mesh, args.output_dir, mesh_name)
        
        # Visualize if requested
        if args.visualize:
            if args.vedo:
                visualize_sdf_data_vedo(points, sdf, mesh=None, show_mesh=False)
            else:
                visualize_sdf_data(points, sdf, mesh, args.show_mesh)
            


def main():
    parser = argparse.ArgumentParser(description='Generate SDF data from 3D meshes.')
    parser.add_argument("input_path", type=str, help="Path to mesh file or directory containing meshes")
    parser.add_argument("--output_dir", "-o", type=str, default="sdf_output", 
                        help="Output directory for saved data (default: sdf_output)")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to log file (default: None)")
    parser.add_argument("--visualize", "-v", action="store_true", 
                        help="Visualize mesh, points, and SDF values")
    parser.add_argument("--save", "-s", action="store_true", 
                        help="Save mesh, points, and SDF data to files")
    parser.add_argument("--vedo", action="store_true",
                        help="Use Vedo for visualization instead of trimesh's default viewer")
    parser.add_argument("--show_mesh", action="store_true", default=True,
                        help="Show mesh in visualization (default: True)")
    parser.add_argument("--num_points", type=int, default=100000,
                        help="Number of query points to sample (default: 100000)")
    parser.add_argument("--noise_scale", type=float, default=0.0025,
                        help="Scale of Gaussian noise for surface sampling (default: 0.0025)")
    parser.add_argument("--num_samples", type=int, default=1)
    
    args = parser.parse_args()
    if args.log_file is not None:
        if os.path.exists(args.log_file):
            with open(args.log_file, 'rb') as log_f:
                log_list = pickle.load(log_f)
            print(f"Loaded log file with {len(log_list)} entries from {args.log_file}")
        else:
            log_list = []
    
    # Process input
    if os.path.isfile(args.input_path):
        process_single_mesh(args.input_path, args)
    elif os.path.isdir(args.input_path):
        # Directory of meshes
        mesh_files = [f for f in os.listdir(args.input_path) 
                     if f.lower().endswith(('.obj', '.ply', '.off', '.stl'))]
        
        if not mesh_files:
            print(f"No mesh files found in {args.input_path}")
            return
            
        print(f"Found {len(mesh_files)} mesh files")
        
        for mesh_file in tqdm(mesh_files, desc="Processing meshes"):
            if args.log_file is not None:
                if mesh_file in log_list:
                    print(f"Skipping {mesh_file}, already processed.")
                    continue
            mesh_path = os.path.join(args.input_path, mesh_file)
            process_single_mesh(mesh_path, args)
            if args.log_file is not None:
                log_list.append(mesh_file)
                with open(args.log_file, 'wb') as log_f:
                    pickle.dump(log_list, log_f)
    else:
        print(f"Error: {args.input_path} is not a valid file or directory")


if __name__ == '__main__':
    main()

