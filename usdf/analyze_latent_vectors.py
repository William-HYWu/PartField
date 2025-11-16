#!/usr/bin/env python3

import torch
import json
import os
import numpy as np
import argparse
from pathlib import Path


def load_latent_vectors(latent_path):
    """
    Load latent vectors from a saved checkpoint.
    
    Args:
        latent_path (str): Path to the saved latent vectors .pth file
        
    Returns:
        torch.nn.Embedding: The latent vectors embedding layer
    """
    latent_obj = torch.load(latent_path, map_location='cpu')

    # Support multiple formats for saved latent vectors:
    # - state_dict from an Embedding ({'weight': Tensor(...)})
    # - a raw Tensor saved directly
    # - a numpy array saved with torch (will be loaded as a Tensor)
    weight = None

    # If user saved an embedding state_dict
    if isinstance(latent_obj, dict):
        # common key is 'weight'
        if 'weight' in latent_obj and isinstance(latent_obj['weight'], torch.Tensor):
            weight = latent_obj['weight']
        else:
            # fallback: look for the first tensor-like value in the dict
            for v in latent_obj.values():
                if isinstance(v, torch.Tensor):
                    weight = v
                    break

    # If it's already a tensor (or numpy array converted to tensor)
    if weight is None:
        if isinstance(latent_obj, torch.Tensor):
            weight = latent_obj
        elif isinstance(latent_obj, np.ndarray):
            weight = torch.from_numpy(latent_obj)

    if weight is None:
        raise ValueError(f"Unsupported latent file format: {latent_path}")

    # Ensure weight is 2D
    if weight.dim() != 2:
        raise ValueError(f"Expected latent weight tensor to be 2D (num_scenes x latent_dim), got shape {weight.shape}")

    num_scenes, latent_dim = weight.shape

    # Create an Embedding from the weight. Use from_pretrained to preserve values.
    try:
        lat_vecs = torch.nn.Embedding.from_pretrained(weight.clone(), freeze=False)
    except Exception:
        # fallback: create embedding and copy weights
        lat_vecs = torch.nn.Embedding(num_scenes, latent_dim)
        with torch.no_grad():
            lat_vecs.weight.copy_(weight)

    print(f"Loaded latent vectors: {num_scenes} scenes, {latent_dim} dimensions")

    return lat_vecs


def load_scene_mapping(models_dir):
    """
    Load scene index to filename mapping.
    
    Args:
        models_dir (str): Directory containing the scene mapping file
        
    Returns:
        dict: Mapping from scene index to scene filename
    """
    mapping_path = os.path.join(models_dir, "scene_index_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: Scene mapping file not found at {mapping_path}")
        return {}


def get_latent_vector_for_scene(lat_vecs, scene_index):
    """
    Get the latent vector for a specific scene.
    
    Args:
        lat_vecs (torch.nn.Embedding): Loaded latent vectors
        scene_index (int): Index of the scene
        
    Returns:
        torch.Tensor: Latent vector for the scene
    """
    if scene_index >= lat_vecs.num_embeddings:
        raise ValueError(f"Scene index {scene_index} out of range (max: {lat_vecs.num_embeddings - 1})")
    
    return lat_vecs.weight[scene_index].detach()


def analyze_latent_vectors(lat_vecs):
    """
    Analyze the latent vectors and print statistics.
    
    Args:
        lat_vecs (torch.nn.Embedding): Loaded latent vectors
    """
    weight = lat_vecs.weight.detach()
    
    print(f"\n=== Latent Vector Analysis ===")
    print(f"Number of scenes: {lat_vecs.num_embeddings}")
    print(f"Latent dimension: {lat_vecs.embedding_dim}")
    print(f"Weight shape: {weight.shape}")
    
    # Statistics
    print(f"\nStatistics:")
    print(f"  Mean magnitude: {torch.norm(weight, dim=1).mean():.4f}")
    print(f"  Max magnitude: {torch.norm(weight, dim=1).max():.4f}")
    print(f"  Min magnitude: {torch.norm(weight, dim=1).min():.4f}")
    print(f"  Mean value: {weight.mean():.4f}")
    print(f"  Std value: {weight.std():.4f}")
    print(f"  Value range: [{weight.min():.4f}, {weight.max():.4f}]")
    
    # Dimension analysis
    dim_means = weight.mean(dim=0)
    dim_stds = weight.std(dim=0)
    print(f"\nDimension Analysis:")
    print(f"  Mean across dimensions: {dim_means.mean():.4f} ± {dim_means.std():.4f}")
    print(f"  Std across dimensions: {dim_stds.mean():.4f} ± {dim_stds.std():.4f}")


def save_latent_vectors_numpy(lat_vecs, output_path):
    """
    Save latent vectors as numpy array for easier analysis.
    
    Args:
        lat_vecs (torch.nn.Embedding): Loaded latent vectors
        output_path (str): Path to save the numpy array
    """
    weight_np = lat_vecs.weight.detach().numpy()
    np.save(output_path, weight_np)
    print(f"Saved latent vectors as numpy array: {output_path}")


def find_similar_latent_vectors(lat_vecs, scene_index, top_k=5):
    """
    Find the most similar latent vectors to a given scene.
    
    Args:
        lat_vecs (torch.nn.Embedding): Loaded latent vectors
        scene_index (int): Index of the reference scene
        top_k (int): Number of similar scenes to return
        
    Returns:
        list: List of (index, similarity_score) tuples
    """
    weight = lat_vecs.weight.detach()
    reference = weight[scene_index]
    
    # Compute cosine similarity
    similarities = torch.nn.functional.cosine_similarity(weight, reference.unsqueeze(0), dim=1)
    
    # Get top k similar (excluding self)
    similarities[scene_index] = -1  # Exclude self
    top_indices = torch.topk(similarities, top_k).indices
    top_similarities = similarities[top_indices]
    
    results = [(idx.item(), sim.item()) for idx, sim in zip(top_indices, top_similarities)]
    
    print(f"\nTop {top_k} similar scenes to scene {scene_index}:")
    for i, (idx, sim) in enumerate(results):
        print(f"  {i+1}. Scene {idx}: similarity = {sim:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze and work with saved latent vectors")
    parser.add_argument("latent_path", help="Path to saved latent vectors (.pth file)")
    parser.add_argument("--models_dir", help="Directory containing scene mapping (optional)")
    parser.add_argument("--analyze", action="store_true", help="Analyze latent vectors")
    parser.add_argument("--scene_index", type=int, help="Get latent vector for specific scene")
    parser.add_argument("--similar", type=int, help="Find similar scenes to given index")
    parser.add_argument("--save_numpy", help="Save latent vectors as numpy array")
    parser.add_argument("--top_k", type=int, default=5, help="Number of similar scenes to find")
    
    args = parser.parse_args()
    
    # Load latent vectors
    print(f"Loading latent vectors from: {args.latent_path}")
    lat_vecs = load_latent_vectors(args.latent_path)
    
    # Load scene mapping if available
    scene_mapping = {}
    if args.models_dir:
        scene_mapping = load_scene_mapping(args.models_dir)
    
    # Analyze latent vectors
    if args.analyze:
        analyze_latent_vectors(lat_vecs)
    
    # Get specific scene's latent vector
    if args.scene_index is not None:
        try:
            latent_vec = get_latent_vector_for_scene(lat_vecs, args.scene_index)
            scene_name = scene_mapping.get(str(args.scene_index), f"scene_{args.scene_index}")
            print(f"\nLatent vector for scene {args.scene_index} ({scene_name}):")
            print(f"  Shape: {latent_vec.shape}")
            print(f"  Magnitude: {torch.norm(latent_vec):.4f}")
            print(f"  Mean: {latent_vec.mean():.4f}")
            print(f"  Std: {latent_vec.std():.4f}")
        except ValueError as e:
            print(f"Error: {e}")
    
    # Find similar scenes
    if args.similar is not None:
        try:
            similar_scenes = find_similar_latent_vectors(lat_vecs, args.similar, args.top_k)
            if scene_mapping:
                print("\nWith scene names:")
                for idx, sim in similar_scenes:
                    scene_name = scene_mapping.get(str(idx), f"scene_{idx}")
                    print(f"  Scene {idx} ({scene_name}): similarity = {sim:.4f}")
        except ValueError as e:
            print(f"Error: {e}")
    
    # Save as numpy array
    if args.save_numpy:
        save_latent_vectors_numpy(lat_vecs, args.save_numpy)


if __name__ == "__main__":
    main()