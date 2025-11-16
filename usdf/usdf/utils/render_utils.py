import numpy as np


def depth_to_pointcloud(depth, fov):
    """
    Deproject depth image to pointcloud.
    Credit to: https://github.com/mmatl/pyrender/issues/14#issuecomment-485881479

    Args:
        depth: depth image
        fov: field of view in radians
    Returns: pointcloud
    """
    fy = fx = 0.5 / np.tan(fov * 0.5)  # assume aspectRatio is one.
    height = depth.shape[0]
    width = depth.shape[1]

    mask = np.where(depth > 0)

    x = mask[1]
    y = mask[0]

    normalized_x = (x.astype(np.float32) - width * 0.5) / width
    normalized_y = (y.astype(np.float32) - height * 0.5) / height
    normalized_y = -normalized_y

    world_x = normalized_x * depth[y, x] / fx
    world_y = normalized_y * depth[y, x] / fy
    world_z = -depth[y, x]
    ones = np.ones(world_z.shape[0], dtype=np.float32)

    return np.vstack((world_x, world_y, world_z, ones)).T


def depth_to_free_points(depth, fov, min_depth=0.0, max_depth=1.0, n=10):
    """
    Recover free points from depth image by ray tracing free pixels from the camera.
    Some code modified from https://github.com/mmatl/pyrender/issues/14#issuecomment-485881479

    Args:
        depth: depth image
        fov: field of view in radians
        min_depth: minimum depth to consider
        max_depth: maximum depth to consider
        n: number of points to sample
    Returns: pointcloud of free points
    """
    fy = fx = 0.5 / np.tan(fov * 0.5)
    height = depth.shape[0]
    width = depth.shape[1]

    x = np.tile(np.linspace(0, width - 1, width, dtype=int), height)
    y = np.repeat(np.linspace(0, height - 1, height, dtype=int), width)

    normalized_x = ((x.astype(np.float32) - width * 0.5) / width).repeat(n)
    normalized_y = ((y.astype(np.float32) - height * 0.5) / height).repeat(n)
    normalized_y = -normalized_y

    # For depth, we invent several depths between min_depth and max_depth.
    depths = np.zeros((height, width, n), dtype=np.float32)
    depths[:, :] = np.linspace(min_depth, max_depth, n + 1)[1:]

    # Mask depths so that we only consider depths that are closer than the depth in the depth image.
    depth_mask = depth.copy()
    depth_mask[np.where(depth == 0.0)] = np.inf
    mask = depths < depth_mask[:, :, None]

    depths = depths.flatten()
    mask = mask.flatten()

    world_x = normalized_x * depths / fx
    world_y = normalized_y * depths / fy
    world_z = -depths
    ones = np.ones(world_z.shape[0], dtype=np.float32)

    return np.vstack((world_x[mask], world_y[mask], world_z[mask], ones[mask])).T
