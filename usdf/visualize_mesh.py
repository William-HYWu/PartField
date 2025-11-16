#!/usr/bin/env python3

from vedo import Plotter, Mesh, Sphere
from usdf.utils import vedo_utils
import trimesh
import argparse
import os


def visualize_mesh_file(mesh_path, show_sphere=True, show_origin=True):
    """
    Visualize a mesh file (PLY, OBJ, etc.) using vedo
    """
    if not os.path.exists(mesh_path):
        print(f"Error: File {mesh_path} does not exist")
        return
    
    # Load the mesh using trimesh (supports PLY, OBJ, STL, etc.)
    try:
        mesh: trimesh.Trimesh = trimesh.load(mesh_path)
        print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return
    
    # Create vedo plotter
    plt = Plotter(title=f"Mesh: {os.path.basename(mesh_path)}")
    
    # Convert to vedo mesh
    mesh_vedo = Mesh([mesh.vertices, mesh.faces])
    
    # Prepare objects to show
    objects_to_show = [mesh_vedo]
    
    if show_sphere:
        objects_to_show.append(Sphere(alpha=0.2))
    
    if show_origin:
        objects_to_show.append(vedo_utils.draw_origin(0.2))
    
    # Show the mesh
    plt.at(0).show(*objects_to_show)
    plt.interactive().close()


def main():
    parser = argparse.ArgumentParser(description="Visualize mesh files (PLY, OBJ, STL, etc.)")
    parser.add_argument("mesh_file", help="Path to the mesh file")
    parser.add_argument("--no-sphere", action="store_true", help="Don't show reference sphere")
    parser.add_argument("--no-origin", action="store_true", help="Don't show origin axes")
    
    args = parser.parse_args()
    
    visualize_mesh_file(
        args.mesh_file, 
        show_sphere=not args.no_sphere,
        show_origin=not args.no_origin
    )


if __name__ == "__main__":
    main()