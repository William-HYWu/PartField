#!/usr/bin/env python3

import torch
import json
import os
from models.decoder import Decoder
from analyze_latent_vectors import load_latent_vectors, load_scene_mapping
from utils.mesh import create_mesh,create_mesh_from_se2_vector
import numpy as np
import math

def load_experiment_specifications(experiment_directory):

    filename = os.path.join(experiment_directory, "specs.json")

    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include specifications file "
            + '"specs.json"'.format(experiment_directory)
        )

    return json.load(open(filename))


def reconstruct_from_latent(decoder_path, latent_path, scene_index, output_name):
    """
    Reconstruct a mesh using saved decoder and latent vector.
    
    Args:
        decoder_path (str): Path to saved decoder model
        latent_path (str): Path to saved latent vectors
        scene_index (int): Index of scene to reconstruct
        output_name (str): Name for output mesh file
    """
    scene_index = int(scene_index)
    # Load decoder
    print(f"Loading decoder from: {decoder_path}")
    decoder_state_dict = torch.load(decoder_path, map_location='cpu')
    specs = load_experiment_specifications("configs/experiments")
    
    # Create decoder (you might need to adjust these parameters based on your specs)
    latent_size = 128  # This should match your training configuration
    decoder = Decoder(
         latent_size, **specs["NetworkSpecs"]
    ).cuda()
    decoder.load_state_dict(decoder_state_dict)
    decoder.eval()
    
    # Load latent vectors
    print(f"Loading latent vectors from: {latent_path}")
    lat_vecs = load_latent_vectors(latent_path)
    
    # Get specific latent vector
    if scene_index >= lat_vecs.num_embeddings:
        raise ValueError(f"Scene index {scene_index} out of range (max: {lat_vecs.num_embeddings - 1})")
    
    latent_vector = lat_vecs.weight[scene_index].unsqueeze(0).cuda()  # Add batch dimension
    print(f"Using latent vector for scene {scene_index}")
    print(f"Latent vector shape: {latent_vector.shape}")
    print(f"Latent vector magnitude: {torch.norm(latent_vector):.4f}")
    
    # Create mesh
    print(f"Generating mesh...")
    create_mesh(decoder, latent_vector, output_name, N=256)
    
    print(f"Mesh saved as: {output_name}.ply")


def batch_reconstruct(decoder_path, latent_path, models_dir, scene_indices=None, output_dir="reconstructions"):
    """
    Reconstruct multiple meshes from saved latent vectors.
    
    Args:
        decoder_path (str): Path to saved decoder model
        latent_path (str): Path to saved latent vectors
        models_dir (str): Directory containing scene mapping
        scene_indices (list): List of scene indices to reconstruct (None for all)
        output_dir (str): Directory to save reconstructed meshes
    """
    specs = load_experiment_specifications("configs/experiments")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load scene mapping
    scene_mapping = load_scene_mapping(models_dir)
    
    # Load latent vectors to get number of scenes
    lat_vecs = load_latent_vectors(latent_path)
    
    if scene_indices is None:
        scene_indices = list(range(min(10, lat_vecs.num_embeddings)))  # Reconstruct first 10 scenes
        print(f"No scene indices specified, reconstructing first {len(scene_indices)} scenes")
    
    print(f"Batch reconstructing {len(scene_indices)} scenes...")
    
    for i, scene_idx in enumerate(scene_indices):
        scene_name = scene_mapping.get(str(scene_idx), f"scene_{scene_idx}")
        output_name = os.path.join(output_dir, f"reconstruction_{scene_idx}_{scene_name}")
        
        try:
            print(f"\n[{i+1}/{len(scene_indices)}] Reconstructing scene {scene_idx} ({scene_name})")
            reconstruct_from_latent(decoder_path, latent_path, scene_idx, output_name)
        except Exception as e:
            print(f"Error reconstructing scene {scene_idx}: {e}")


def interpolate_between_scenes(decoder_path, latent_path, scene_idx1, scene_idx2, 
                             num_steps=5, output_dir="interpolations"):
    """
    Create interpolations between two latent vectors.
    
    Args:
        decoder_path (str): Path to saved decoder model
        latent_path (str): Path to saved latent vectors  
        scene_idx1 (int): First scene index
        scene_idx2 (int): Second scene index
        num_steps (int): Number of interpolation steps
        output_dir (str): Directory to save interpolated meshes
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load latent vectors
    lat_vecs = load_latent_vectors(latent_path)
    
    latent1 = lat_vecs.weight[scene_idx1]
    latent2 = lat_vecs.weight[scene_idx2]
    
    print(f"Interpolating between scene {scene_idx1} and scene {scene_idx2}")
    print(f"Creating {num_steps} interpolation steps...")
    
    for i in range(num_steps):
        alpha = i / (num_steps - 1)  # From 0 to 1
        interpolated_latent = (1 - alpha) * latent1 + alpha * latent2
        interpolated_latent = interpolated_latent.unsqueeze(0)  # Add batch dimension
        
        output_name = os.path.join(output_dir, f"interpolation_{scene_idx1}_to_{scene_idx2}_step_{i:02d}_alpha_{alpha:.2f}")
        
        # Load decoder and reconstruct
        decoder_state_dict = torch.load(decoder_path, map_location='cpu')
        latent_size = interpolated_latent.shape[1]
        decoder = Decoder(latent_size, dims=[512, 512, 512, 512, 512, 512, 512, 512], 
                         dropout=[0, 1, 2, 3, 4, 5, 6, 7], dropout_prob=0.2,
                         norm_layers=[0, 1, 2, 3, 4, 5, 6, 7],
                         latent_in=[4], xyz_in_all=True, use_tanh=True, latent_dropout=False)
        decoder.load_state_dict(decoder_state_dict)
        decoder.eval()
        
        print(f"  Step {i+1}/{num_steps} (α={alpha:.2f})")
        create_mesh(decoder, interpolated_latent, output_name, N=128)  # Lower resolution for speed


def reconstruct_from_random(decoder_path, num_scenes, output_name):
    """
    Reconstruct meshes from random latent vectors.
    
    Args:
        decoder_path (str): Path to saved decoder model
        num_scenes (int): Number of random scenes to reconstruct
        output_name (str): Base name for output mesh files
    """
    # Load decoder
    print(f"Loading decoder from: {decoder_path}")
    decoder_state_dict = torch.load(decoder_path, map_location='cpu')
    specs = load_experiment_specifications("configs/experiments")
    
    latent_size = 128  # This should match your training configuration
    decoder = Decoder(
         latent_size, **specs["NetworkSpecs"]
    ).cuda()
    decoder.load_state_dict(decoder_state_dict)
    decoder.eval()
    
    import math
    def get_spec_with_default(specs, key, default):
        return specs[key] if key in specs else default
    
    #generate random se2 transform latent vectors
    angles = np.random.uniform(-0.5*np.pi,  0.5* np.pi, num_scenes)
    translations = np.random.uniform(-0.1, 0.1, (num_scenes, 2))
    for i, (angle, translation) in enumerate(zip(angles, translations)):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        tx, ty = translation
        se2_vector=torch.from_numpy(np.array([tx, ty, sin_a, cos_a])).float().cuda()
        scene_output_name = f"{output_name}_random_{i}"
        create_mesh_from_se2_vector(decoder, se2_vector, scene_output_name, N=256)
        print(f"Mesh saved as: {scene_output_name}.ply")

def examine_trained_latent_space(decoder_path, output_dir="examinations", num_samples=5):
    
    decoder_state_dict = torch.load(decoder_path, map_location='cpu')
    specs = load_experiment_specifications("configs/experiments")
    
    latent_size = 128  # This should match your training configuration
    decoder = Decoder(
         latent_size, **specs["NetworkSpecs"]
    ).cuda()
    decoder.load_state_dict(decoder_state_dict)
    decoder.eval()
    
    #compute distribution statistics of trained latent space
    angles = np.random.uniform(-0.5*np.pi,  0.5* np.pi, 500)
    translations = np.random.uniform(-0.1, 0.1, (500, 2))
    se2_vecs = []
    for i, (angle, translation) in enumerate(zip(angles, translations)):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        tx, ty = translation
        se2_vector=torch.from_numpy(np.array([tx, ty, sin_a, cos_a])).float().cuda()
        se2_vecs.append(se2_vector.unsqueeze(0))
    se2_vecs = torch.cat(se2_vecs, dim=0).float().cuda()
    
    trained_latent_vec_sapce = decoder.get_latent(se2_vecs).float().cpu().detach()
    #compute distribution statistics of trained latent space
    mean = torch.mean(trained_latent_vec_sapce, dim=0)
    std = torch.std(trained_latent_vec_sapce, dim=0)
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    
    #Initialize latent vector from trained latent space
    mean = mean.unsqueeze(0).expand(num_samples, -1).cuda()
    std = std.unsqueeze(0).expand(num_samples, -1).cuda()
    latent = torch.normal(mean=mean, std=std).cuda()
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)
    for i in range(num_samples):
        scene_output_name = os.path.join(output_dir, f"trained_latent_sample_{i}")
        create_mesh(decoder, latent[i].unsqueeze(0), scene_output_name, N=256)
        print(f"Mesh saved as: {scene_output_name}.ply")
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Reconstruct meshes from saved latent vectors")
    parser.add_argument("--decoder", required=True, help="Path to saved decoder model")
    parser.add_argument("--latents", required=True, help="Path to saved latent vectors")
    parser.add_argument("--models_dir", help="Directory containing scene mapping")
    parser.add_argument("--scene", type=int, help="Scene index to reconstruct")
    parser.add_argument("--output", default="reconstruction", help="Output name for single reconstruction")
    parser.add_argument("--batch", nargs="*", type=int, help="Batch reconstruct multiple scenes")
    parser.add_argument("--interpolate", nargs=2, type=int, help="Interpolate between two scenes")
    parser.add_argument("--steps", type=int, default=5, help="Number of interpolation steps")
    parser.add_argument("--random", type=int, help="Reconstruct from random latent vectors, specify number of scenes")
    parser.add_argument("--examine_trained_latent", action="store_true", help="Examine trained latent space statistics and generate samples")
    
    args = parser.parse_args()
    
    if args.scene is not None:
        # Single reconstruction
        reconstruct_from_latent(args.decoder, args.latents, args.scene, args.output)
        
    elif args.batch is not None:
        # Batch reconstruction
        if not args.models_dir:
            print("Warning: --models_dir not specified, using generic scene names")
        batch_reconstruct(args.decoder, args.latents, args.models_dir, args.batch)
        
    elif args.interpolate:
        # Interpolation
        interpolate_between_scenes(args.decoder, args.latents, 
                                 args.interpolate[0], args.interpolate[1], args.steps)
    
    elif args.random is not None:
        # Random latent vector reconstruction
        reconstruct_from_random(args.decoder, args.random, args.output)
    
    elif args.examine_trained_latent:
        # Examine trained latent space
        examine_trained_latent_space(args.decoder, num_samples=5)
        
    else:
        print("Please specify --scene, --batch, or --interpolate")
        parser.print_help()