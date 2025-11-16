import torch
import json
import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from sklearn.decomposition import PCA
from analyze_latent_vectors import load_latent_vectors
from models.decoder import Decoder
from utils.mesh import create_mesh


def load_experiment_specifications(experiment_directory):
    filename = os.path.join(experiment_directory, "specs.json")
    if not os.path.isfile(filename):
        raise Exception(f"Specs file not found: {filename}")
    return json.load(open(filename))


def render_mesh_to_image(mesh_path, output_path, resolution=(800, 800), camera_angle=(45, 45)):
    """
    Render a mesh to an image using matplotlib.
    
    Args:
        mesh_path: Path to .ply mesh file
        output_path: Path to save rendered image
        resolution: Image resolution (width, height)
        camera_angle: Camera elevation and azimuth angles
    """
    try:
        # Load mesh
        mesh = trimesh.load(mesh_path)
        
        # Check if mesh is valid
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError("Mesh has no vertices or faces")
        
        # Create figure
        fig = plt.figure(figsize=(resolution[0]/100, resolution[1]/100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # Get vertices and faces
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Plot mesh
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                        triangles=faces, 
                        color='steelblue', 
                        alpha=0.8,
                        edgecolor='none',
                        shade=True)
        
        # Set view angle
        ax.view_init(elev=camera_angle[0], azim=camera_angle[1])
        
        # Set equal aspect ratio
        max_range = np.array([
            vertices[:, 0].max() - vertices[:, 0].min(),
            vertices[:, 1].max() - vertices[:, 1].min(),
            vertices[:, 2].max() - vertices[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
        mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
        mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Remove axes
        ax.set_axis_off()
        
        # Save
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100, format='png')
        plt.close(fig)
        
        # Verify the file was saved
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise IOError(f"Failed to save image to {output_path}")
            
    except Exception as e:
        plt.close('all')  # Clean up any open figures
        raise RuntimeError(f"Failed to render mesh {mesh_path}: {e}")


def plot_latent_trajectories(latent_history, save_path, train_latent=None, shape_name="mugs"):
    """
    Plot the trajectories of latent codes in 2D PCA space.
    
    Args:
        latent_history: List of numpy arrays, each of shape [num_particles, latent_size]
        save_path: Path to save the plot
        shape_name: Name of the shape for the title
    """
    latent_history = latent_history.cpu().numpy()
    num_iterations = len(latent_history)
    num_particles = 1
    
    # Flatten all latent codes across time and particles for PCA
    # Shape: [num_iterations * num_particles, latent_size]
    all_latents = np.vstack(latent_history)
    
    # Fit PCA on all latent codes
    pca = PCA(n_components=2)
    if train_latent is not None:
        # train_latent = load_latent_vectors(train_latent)
        # train_latent = train_latent.weight[:]
        # # Include training latent codes in PCA fitting
        # train_latent_np = train_latent.detach().cpu().numpy()
        train_latent_np = np.load(train_latent)
        pca.fit(train_latent_np)
        train_latent_2d = pca.transform(train_latent_np)
        all_latents_2d = pca.transform(all_latents)
    else:
        all_latents_2d = pca.fit_transform(all_latents)
    latents_2d = all_latents_2d.reshape(num_iterations, num_particles, 2)
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Color map for particles
    colors = plt.cm.tab10(np.linspace(0, 1, num_particles))
    colors_train = plt.cm.tab10(np.linspace(0, 1, train_latent_np.shape[0])) if train_latent is not None else None

    # Plot training latent codes
    if colors_train is not None:
        plt.scatter(train_latent_2d[:, 0], train_latent_2d[:, 1], 
                    marker='X', s=200, color=colors_train,
                    label='Training Latents', edgecolors='black', linewidths=2)

    # Plot trajectory for each particle
    for particle_idx in range(num_particles):
        trajectory = latents_2d[:, particle_idx, :]
        
        # Plot the trajectory line with directional markers
        plt.plot(trajectory[:, 0], trajectory[:, 1], 
                alpha=0.7, linewidth=2.5, color=colors[particle_idx],
                label=f'Particle {particle_idx}')
        
        # Mark the start point (larger, hollow circle)
        plt.scatter(trajectory[0, 0], trajectory[0, 1], 
                   marker='o', s=200, facecolors='white',
                   edgecolors=colors[particle_idx], linewidths=3, zorder=10)
        
        # Mark the end point (filled star)
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1], 
                   marker='*', s=400, color=colors[particle_idx],
                   edgecolors='black', linewidths=2, zorder=10)

    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=14)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=14)
    plt.title(f'SVGD Particle Trajectories in Latent Space\n{shape_name}\n'
              f'○ = Start, ★ = End, {num_iterations} total updates', fontsize=14, pad=20)
    plt.legend(loc='upper left', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved latent trajectory plot: {save_path}")


def create_gif(decoder_path, latent_path, partical_idx, length=100, output_gif="interpolation.gif", temp_dir="temp_gif", fps=5, resolution=(800, 800)):
    """
    create gif from a sequence of latent_vectors
    """
    os.makedirs(temp_dir, exist_ok=True)
    # Load decoder and latent vectors
    decoder_state_dict = torch.load(decoder_path, map_location='cpu')
    specs = load_experiment_specifications("configs/experiments")
    
    latent_size = specs["CodeLength"]
    decoder = Decoder(latent_size, **specs["NetworkSpecs"]).cuda()
    decoder.load_state_dict(decoder_state_dict)
    decoder.eval()
    lat_vecs = np.load(latent_path)
    print(f"Loaded latent vectors from {latent_path} with shape {lat_vecs.shape}")
    
    # Store original shape for trajectory plotting
    # Expected shape: [num_iterations, num_particles, latent_dim]
    original_shape = lat_vecs.shape
    
    # Select the particle and track all timesteps for trajectory
    if len(original_shape) == 3:
        print(f"Selecting particle index {partical_idx} from latent vectors with shape {original_shape}")
        # Shape: [iterations, particles, latent_dim]
        selected_particle_trajectory = lat_vecs[:, partical_idx, :]  # [iterations, latent_dim]
        # For trajectory plotting, we need [iterations, 1, latent_dim] format
        latent_history = [lat_vecs[i:i+1, partical_idx:partical_idx+1, :] for i in range(min(length, len(lat_vecs)))]
    else:
        print(f"Using latent vectors with shape {original_shape} as single particle trajectory")
        # If shape is [iterations, latent_dim], treat as single particle
        selected_particle_trajectory = lat_vecs
        latent_history = [lat_vecs[i:i+1, :] for i in range(min(length, len(lat_vecs)))]
    
    lat_vecs = torch.from_numpy(selected_particle_trajectory).cuda()[:length]
    
    # Convert latent history to the format needed for plotting
    latent_history_np = [lh.squeeze(0) if lh.ndim == 3 else lh for lh in latent_history]
    
    # Generate meshes, trajectory plots, and render combined images
    image_paths = []
    pca = None
    latents_2d_full = None
    
    for i in range(length):
        latent = lat_vecs[i]
        latent_history = lat_vecs[:i+1]
        latent = latent.unsqueeze(0)  # Add batch dimension
        
        # Create mesh
        output_name = os.path.join(temp_dir, f"mesh_{i:03d}")
        create_mesh(decoder, latent, output_name, N=256)  # Lower resolution for speed
        
        # create_mesh writes a .ply file (see utils/mesh.py convert_sdf_samples_to_ply)
        mesh_path = output_name + ".ply"
        if not os.path.exists(mesh_path):
            # fallback to .obj if available
            alt = output_name + ".obj"
            if os.path.exists(alt):
                mesh_path = alt
            else:
                raise RuntimeError(f"Expected mesh file not found: {output_name}.ply or {output_name}.obj")
        
        # Render mesh
        mesh_image_path = os.path.join(temp_dir, f"mesh_{i:03d}.png")
        render_mesh_to_image(mesh_path, mesh_image_path, resolution=resolution)
        
        # Create trajectory plot for current frame
        trajectory_image_path = os.path.join(temp_dir, f"trajectory_{i:03d}.png")
        plot_latent_trajectories(
            latent_history, 
            trajectory_image_path, 
            train_latent="sorted_train_latents.npy",
        )
        
        # Combine mesh and trajectory side by side
        mesh_img = Image.open(mesh_image_path)
        traj_img = Image.open(trajectory_image_path)
        
        # Resize trajectory to match mesh height
        traj_img = traj_img.resize((mesh_img.height, mesh_img.height), Image.LANCZOS)
        
        # Create combined image
        combined_width = mesh_img.width + traj_img.width
        combined_img = Image.new('RGB', (combined_width, mesh_img.height), (255, 255, 255))
        combined_img.paste(mesh_img, (0, 0))
        combined_img.paste(traj_img, (mesh_img.width, 0))
        
        # Save combined image
        combined_path = os.path.join(temp_dir, f"combined_{i:03d}.png")
        combined_img.save(combined_path)
        image_paths.append(combined_path)
    
    # Create GIF
    images = [Image.open(p) for p in image_paths]
    gif_path = os.path.join(output_gif)
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=1000 / fps,
        loop=0
    )
    print(f"Saved GIF with animated trajectory plots to {gif_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create GIF from latent vector sequence")
    parser.add_argument("decoder_path", type=str, help="Path to saved decoder model")
    parser.add_argument("latent_path", type=str, help="Path to saved latent vectors")
    parser.add_argument("--partical_idx", type=int, default=0, help="Index of the latent vector particle to use")
    parser.add_argument("--length", type=int, default=100, help="Number of frames in the GIF")
    parser.add_argument("--output_gif", type=str, default="interpolation.gif", help="Output GIF file path")
    parser.add_argument("--temp_dir", type=str, default="temp_gif", help="Temporary directory for intermediate files")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the GIF")
    parser.add_argument("--resolution", type=int, nargs=2, default=(800, 800), help="Resolution of rendered images (width height)")
    parser.add_argument("-e","--expe")
    
    args = parser.parse_args()
    
    create_gif(
        decoder_path=args.decoder_path,
        latent_path=args.latent_path,
        partical_idx=args.partical_idx,
        length=args.length,
        output_gif=args.output_gif,
        temp_dir=args.temp_dir,
        fps=args.fps,
        resolution=tuple(args.resolution)
    )