import numpy as np
import trimesh
import open3d as o3d
from vedo import Plotter, Mesh, Points, color_map

from usdf.utils import vedo_utils


def sample_points_from_ball(n_points, ball_radius=1.1):
    """
    Sample points evenly from inside unit ball by sampling points in the unit cube
    and rejecting points outside the unit ball.

    Args:
        n_points: number points to return
        ball_radius: radius of ball to sample from
    Returns: point cloud of sampled query points
    """
    points = np.empty((0, 3), dtype=np.float32)

    while len(points) < n_points:
        # Sample points in the unit cube.
        new_points = np.random.uniform(-ball_radius, ball_radius, size=(n_points, 3))

        # Reject points outside the unit ball.
        mask = np.linalg.norm(new_points, axis=1) <= ball_radius
        points = np.concatenate([points, new_points[mask]], axis=0)

    return points[:n_points]


def get_sdf_query_points(mesh: trimesh.Trimesh, n_random: int = 10000, n_off_surface: int = 10000,
                         off_surface_sigma_a: float = 0.004, off_surface_sigma_b: float = 0.001):
    if n_random > 0:
        query_points_random = sample_points_from_ball(n_random)
    else:
        query_points_random = np.empty([0, 3], dtype=float)

    if n_off_surface > 0:
        surface_points = mesh.sample(n_off_surface)
        query_points_surface_a = surface_points + np.random.normal(0.0, off_surface_sigma_a, size=surface_points.shape)
        query_points_surface_b = surface_points + np.random.normal(0.0, off_surface_sigma_b, size=surface_points.shape)
        query_points_surface = np.concatenate([query_points_surface_a, query_points_surface_b], axis=0)
    else:
        query_points_surface = np.empty([0, 3], dtype=float)

    return np.concatenate([query_points_random, query_points_surface])


def get_sdf_values(mesh: trimesh.Trimesh, query_points: np.ndarray):
    # Convert mesh to open3d mesh.
    mesh_o3d = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh.faces)
    )

    # Build o3d scene with triangle mesh.
    tri_mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(tri_mesh_legacy)

    # Compute SDF to surface.
    query_points_ = o3d.core.Tensor(query_points, dtype=o3d.core.Dtype.Float32)
    signed_distance = scene.compute_signed_distance(query_points_)
    signed_distance_np = signed_distance.numpy()

    return signed_distance_np


def generate_sdf_data(rot_mesh: trimesh.Trimesh, n_random: int = 10000, n_off_surface: int = 10000,
                      off_surface_sigma_a: float = 0.004, off_surface_sigma_b: float = 0.001, vis: bool = False):
    # Sample the SDF points to evaluate.
    query_points = get_sdf_query_points(
        rot_mesh, n_random=n_random, n_off_surface=n_off_surface,
        off_surface_sigma_a=off_surface_sigma_a, off_surface_sigma_b=off_surface_sigma_b
    )

    # Calculate the SDF values.
    sdf_values = get_sdf_values(rot_mesh, query_points)

    if vis:
        plt = Plotter((1, 3))
        plt.at(0).show(Mesh([rot_mesh.vertices, rot_mesh.faces], c="red"),
                       vedo_utils.draw_origin(scale=0.1))

        # Downsample query points for visualization.
        idcs = np.random.choice(len(query_points), 10000, replace=False)
        vis_query_points = query_points[idcs]
        vis_sdf_values = sdf_values[idcs]
        qp_colors = [color_map(sdf_value, "jet", vmin=vis_sdf_values.min(), vmax=vis_sdf_values.max())
                     for sdf_value in vis_sdf_values]
        plt.at(1).show(Points(vis_query_points, c=qp_colors), vedo_utils.draw_origin(scale=0.1))
        plt.at(2).show(Points(query_points[sdf_values < 0.0]), vedo_utils.draw_origin(scale=0.1))
        plt.interactive().close()

    return {
        "query_points": query_points,
        "sdf_values": sdf_values
    }
