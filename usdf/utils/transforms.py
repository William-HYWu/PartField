import numpy as np


def generate_se2_matrix(theta: float, tx: float, ty: float) -> np.ndarray:
    """Generate a 3x3 SE(2) homogeneous transformation matrix.
    
    This matrix can rotate a 2D point cloud around the origin and translate it in the XY plane.
    For 3D point clouds, it rotates around the Z-axis (keeping Z unchanged) and translates in XY.
    
    Args:
        theta: rotation angle in radians (counterclockwise around Z-axis)
        tx: translation in X direction
        ty: translation in Y direction
    
    Returns:
        3x3 SE(2) matrix T where:
            [cos(θ)  -sin(θ)  tx]
            [sin(θ)   cos(θ)  ty]
            [  0        0      1]
    
    Usage for 2D points (N x 2):
        points_2d = np.array([[x1, y1], [x2, y2], ...])
        points_h = np.concatenate([points_2d, np.ones((N, 1))], axis=1)  # Nx3 homogeneous
        transformed_h = points_h @ T.T
        transformed_2d = transformed_h[:, :2]
    
    Usage for 3D points (N x 3) - rotation around Z, translation in XY:
        points_3d = np.array([[x1, y1, z1], [x2, y2, z2], ...])
        xy = points_3d[:, :2]  # Extract XY
        xy_h = np.concatenate([xy, np.ones((N, 1))], axis=1)
        xy_transformed_h = xy_h @ T.T
        xy_transformed = xy_transformed_h[:, :2]
        points_3d_transformed = np.column_stack([xy_transformed, points_3d[:, 2]])  # Keep Z unchanged
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    T = np.array([
        [cos_theta, -sin_theta, tx],
        [sin_theta,  cos_theta, ty],
        [0.0,        0.0,       1.0]
    ], dtype=np.float64)
    
    return T


def apply_se2_to_pointcloud_2d(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply SE(2) transformation to 2D point cloud.
    
    Args:
        points: (N, 2) array of 2D points
        T: (3, 3) SE(2) transformation matrix
    
    Returns:
        (N, 2) transformed points
    """
    if points.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) points, got shape {points.shape}")
    if T.shape != (3, 3):
        raise ValueError(f"Expected (3, 3) SE(2) matrix, got shape {T.shape}")
    
    N = points.shape[0]
    points_h = np.concatenate([points, np.ones((N, 1), dtype=np.float64)], axis=1)
    transformed_h = points_h @ T.T
    return transformed_h[:, :2]


def apply_se2_to_pointcloud_3d(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply SE(2) transformation to 3D point cloud (rotates around Z, translates in XY).
    
    Args:
        points: (N, 3) array of 3D points
        T: (3, 3) SE(2) transformation matrix
    
    Returns:
        (N, 3) transformed points (Z coordinates unchanged)
    """
    if points.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) points, got shape {points.shape}")
    if T.shape != (3, 3):
        raise ValueError(f"Expected (3, 3) SE(2) matrix, got shape {T.shape}")
    
    # Transform XY coordinates
    xy = points[:, :2]
    xy_transformed = apply_se2_to_pointcloud_2d(xy, T)
    
    # Keep Z unchanged
    z = points[:, 2:3]
    return np.concatenate([xy_transformed, z], axis=1)


def compute_max_radius_3d(points: np.ndarray) -> float:
    """Compute the maximum distance from origin to any point in the point cloud.
    
    Args:
        points: (N, 3) array of 3D points
    
    Returns:
        maximum radius (distance from origin)
    """
    distances = np.linalg.norm(points, axis=1)
    return np.max(distances)


def compute_safe_translation_bound(points: np.ndarray, sphere_radius: float = 1.0) -> float:
    """Compute maximum safe translation to keep point cloud within sphere after rotation.
    
    For a point cloud that will be rotated around Z-axis and translated in XY plane,
    we need: max_radius_after_rotation + |translation| <= sphere_radius
    
    Since rotation around Z doesn't change the distance from origin (only XY rotation),
    the maximum radius is unchanged by rotation. Therefore:
    max_translation = sphere_radius - max_radius
    
    Args:
        points: (N, 3) array of 3D points
        sphere_radius: radius of the unit sphere (default 1.0)
    
    Returns:
        maximum safe translation magnitude
    """
    max_radius = compute_max_radius_3d(points)
    safe_translation = sphere_radius - max_radius
    
    if safe_translation <= 0:
        raise ValueError(
            f"Point cloud already exceeds sphere radius! "
            f"max_radius={max_radius:.4f}, sphere_radius={sphere_radius}"
        )
    
    return safe_translation


def sample_random_se2_transforms(
    n_samples: int = 8,
    max_rot: float = np.pi,
    max_translate: float = 1.0,
    rng: np.random.Generator = None
) -> list:
    """Sample random SE(2) transformation matrices.
    
    Args:
        n_samples: number of transforms to generate
        max_rot: maximum rotation angle in radians (samples uniformly from [-max_rot, max_rot])
        max_translate: maximum translation magnitude (samples uniformly in circle of this radius)
        rng: numpy random generator (if None, uses default)
    
    Returns:
        list of (3, 3) SE(2) matrices
    """
    if rng is None:
        rng = np.random.default_rng()
    
    matrices = []
    for _ in range(n_samples):
        # Sample rotation uniformly
        theta = rng.uniform(-max_rot, max_rot)
        
        # Sample translation uniformly within circle
        r = max_translate * np.sqrt(rng.uniform(0, 1))
        angle = rng.uniform(0, 2 * np.pi)
        tx = r * np.cos(angle)
        ty = r * np.sin(angle)
        
        T = generate_se2_matrix(theta, tx, ty)
        matrices.append(T)
    
    return matrices


def sample_safe_se2_transforms(
    points: np.ndarray,
    n_samples: int = 8,
    max_rot: float = np.pi,
    sphere_radius: float = 1.0,
    rng: np.random.Generator = None
) -> list:
    """Sample random SE(2) transforms that guarantee point cloud stays within sphere.
    
    This function automatically computes the safe translation bound based on the
    point cloud's maximum radius to ensure all transformed points remain within
    the specified sphere radius.
    
    Args:
        points: (N, 3) array of 3D points in the point cloud
        n_samples: number of transforms to generate
        max_rot: maximum rotation angle in radians (samples uniformly from [-max_rot, max_rot])
        sphere_radius: radius of the unit sphere to stay within (default 1.0)
        rng: numpy random generator (if None, uses default)
    
    Returns:
        list of (3, 3) SE(2) matrices
    
    Example:
        >>> points = load_point_cloud("mug.ply")  # Your mug point cloud
        >>> transforms = sample_safe_se2_transforms(points, n_samples=10)
        >>> # All transforms are guaranteed to keep the mug inside unit sphere
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Compute safe translation bound
    max_translate = compute_safe_translation_bound(points, sphere_radius)
    
    print(f"Point cloud max radius: {compute_max_radius_3d(points):.4f}")
    print(f"Safe translation bound: {max_translate:.4f}")
    
    # Sample transforms with safe translation bound
    matrices = []
    for _ in range(n_samples):
        # Sample rotation uniformly
        theta = rng.uniform(-max_rot, max_rot)
        
        # Sample translation uniformly within safe circle
        r = max_translate * np.sqrt(rng.uniform(0, 1))
        angle = rng.uniform(0, 2 * np.pi)
        tx = r * np.cos(angle)
        ty = r * np.sin(angle)
        
        T = generate_se2_matrix(theta, tx, ty)
        matrices.append(T)
    
    return matrices


def verify_within_sphere(points: np.ndarray, sphere_radius: float = 1.0) -> bool:
    """Verify that all points in the point cloud are within the sphere.
    
    Args:
        points: (N, 3) array of 3D points
        sphere_radius: radius of the sphere
    
    Returns:
        True if all points are within sphere, False otherwise
    """
    max_radius = compute_max_radius_3d(points)
    return max_radius <= sphere_radius


if __name__ == "__main__":
    # Example usage
    print("=== SE(2) Matrix Generation Example ===\n")
    
    # Generate a specific SE(2) transform: rotate 45 degrees, translate by (0.5, 0.3)
    theta = np.pi / 4  # 45 degrees
    tx, ty = 0.5, 0.3
    T = generate_se2_matrix(theta, tx, ty)
    print(f"SE(2) matrix for θ={np.degrees(theta):.1f}°, t=({tx}, {ty}):")
    print(T)
    print()
    
    # Test on 3D point cloud (simulating a small mug)
    points_3d = np.array([
        [0.3, 0.0, 0.2],
        [0.0, 0.3, 0.2],
        [-0.3, 0.0, 0.2],
        [0.0, -0.3, 0.2],
        [0.3, 0.0, -0.2],
        [0.0, 0.3, -0.2],
        [-0.3, 0.0, -0.2],
        [0.0, -0.3, -0.2]
    ])
    print("Original 3D points (simulated mug):")
    print(points_3d)
    print(f"Max radius: {compute_max_radius_3d(points_3d):.4f}")
    print()
    
    # Sample SAFE transforms that guarantee staying within unit sphere
    print("=== Safe SE(2) Transforms (guaranteed within unit sphere) ===\n")
    safe_transforms = sample_safe_se2_transforms(
        points_3d, 
        n_samples=3, 
        max_rot=np.pi, 
        sphere_radius=1.0
    )
    print()
    
    for i, T_safe in enumerate(safe_transforms):
        print(f"Safe transform {i+1}:")
        print(T_safe)
        
        # Apply transform and verify it's within sphere
        transformed = apply_se2_to_pointcloud_3d(points_3d, T_safe)
        is_safe = verify_within_sphere(transformed, sphere_radius=1.0)
        max_r = compute_max_radius_3d(transformed)
        print(f"  Max radius after transform: {max_r:.4f}")
        print(f"  Within unit sphere: {is_safe} ✓" if is_safe else f"  Within unit sphere: {is_safe} ✗")
        print()
    
    # Compare with unsafe random transforms
    print("=== Unsafe Random SE(2) Transforms (may go outside sphere) ===\n")
    unsafe_transforms = sample_random_se2_transforms(n_samples=3, max_rot=np.pi, max_translate=0.8)
    for i, T_unsafe in enumerate(unsafe_transforms):
        print(f"Random transform {i+1}:")
        print(T_unsafe)
        
        # Apply transform and check if it's within sphere
        transformed = apply_se2_to_pointcloud_3d(points_3d, T_unsafe)
        is_safe = verify_within_sphere(transformed, sphere_radius=1.0)
        max_r = compute_max_radius_3d(transformed)
        print(f"  Max radius after transform: {max_r:.4f}")
        print(f"  Within unit sphere: {is_safe} ✓" if is_safe else f"  Within unit sphere: {is_safe} ✗")
        print()
