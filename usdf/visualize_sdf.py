"""
A script to visualize pre-computed SDF data from an .npz file using Vedo.
Loads the new format with 'pos' and 'neg' arrays containing [x, y, z, sdf] data.
"""
import argparse
import os
import numpy as np
import vedo
from usdf.utils import vedo_utils
from vedo import Sphere
from usdf.utils import vedo_utils

def load_sdf_data(npz_path):
    """Load SDF data from an NPZ file."""
    data = np.load(npz_path)
    
    # Check for 'xyz' and 'sdf' keys (new format)
    if 'xyz' in data and 'sdf' in data:
        points = data['xyz']
        sdf = data['sdf']
        print(f"Loaded xyz/sdf format: {len(points)} samples")
        
    # Check for 'pos' and 'neg' keys (old format)
    elif 'pos' in data and 'neg' in data:
        pos_data = data['pos']
        neg_data = data['neg']
        
        # Combine positive and negative samples
        all_data = np.vstack([pos_data, neg_data])
        points = all_data[:, :3]    # xyz coordinates
        sdf = all_data[:, 3]      # sdf values
        
        print(f"Loaded pos/neg format: {len(pos_data)} positive and {len(neg_data)} negative SDF samples")
    else:
        raise ValueError(f"Unknown NPZ format. Expected 'xyz'+'sdf' or 'pos'+'neg' keys, got: {list(data.keys())}")
    
    print(f"Total samples: {len(points)}")
    if sdf.size > 0:
        print(f"SDF range: [{sdf.min():.4f}, {sdf.max():.4f}]")
    else:
        print("SDF data is empty.")
    
    return points, sdf

def visualize_sdf_data(points, sdf, title="SDF Visualization"):
    """
    Visualize SDF data with vedo.
    
    Args:
        points (np.ndarray): Point coordinates (N, 3)
        sdf (np.ndarray): SDF values (N,)
        title (str): Plot title
    """
    plt = vedo.Plotter(title=title)

    # Use a discrete colormap with a binary mask: 0->blue, 1->red
    mask = (sdf > 0).astype(int)
    cloud = vedo.Points(points)
    cloud.cmap(['blue', 'red'], mask, on='points').point_size(4)
    plt.add(cloud.legend("SDF Points (red: >0, blue: <=0)"))
    
    # Add unit sphere and origin coordinate system
    unit_sphere = Sphere(alpha=0.2)
    origin_axes = vedo_utils.draw_origin(0.2)
    plt.add(unit_sphere)
    plt.add(origin_axes)
    
    # Add statistics text
    pos_count = np.sum(sdf > 0)
    neg_count = np.sum(sdf <= 0)
    stats_text = f"Positive: {pos_count}\nNegative: {neg_count}\nRange: [{sdf.min():.3f}, {sdf.max():.3f}]"
    plt.add(vedo.Text2D(stats_text, pos=(0.02, 0.95), s=0.8, c='black', bg='white', alpha=0.8))
    
    plt.show(axes=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SDF data from NPZ files")
    parser.add_argument(
        "--npz", 
        type=str, 
        help="Path to NPZ file (if not provided, uses default example)"
    )
    parser.add_argument(
        "--mug_id",
        type=str,
        default="1a1c0a8d4bad82169f0594e65f756cf5",
        help="Mug ID to visualize (default: 1a1c0a8d4bad82169f0594e65f756cf5)"
    )
    
    args = parser.parse_args()
    
    if args.npz:
        npz_path = args.npz
    else:
        # Use default path with mug_id
        base_dir = "/Users/wuhaoyang/Documents/Research/Projects/3D_Completion/Code/usdf/mugs_dataset"
        npz_path = f"{base_dir}/{args.mug_id}/{args.mug_id}_sdf.npz"
    
    if not os.path.exists(npz_path):
        print(f"Error: NPZ file not found: {npz_path}")
        exit(1)
    
    print(f"Loading SDF data from: {npz_path}")
    points, sdf = load_sdf_data(npz_path)
    
    title = f"SDF Visualization - {os.path.basename(npz_path)}"
    visualize_sdf_data(points, sdf, title)