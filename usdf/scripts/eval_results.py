import argparse

import torch
import numpy as np
import tqdm
import trimesh
from tqdm import trange
from vedo import Plotter, Mesh, Points
import matplotlib as mpl

from usdf.utils.model_utils import load_dataset_from_config
from usdf.utils.results_utils import load_gt_results, load_pred_results
from usdf.visualize import visualize_mesh, visualize_mesh_set
from usdf.utils.metric_utils import minimal_matching_distance, total_matching_distance


def eval_results(dataset_cfg: str, gen_dir: str, mode: str = "test", offset: int = 0):
    # Load dataset.
    dataset_cfg, dataset = load_dataset_from_config(dataset_cfg, dataset_mode=mode)
    num_examples = len(dataset)
    dataset_cfg = dataset_cfg["data"][args.mode]

    # Load ground truth information.
    gt_meshes = load_gt_results(dataset, dataset_cfg, num_examples)

    # Load predicted information.
    predictions = load_pred_results(gen_dir, num_examples)

    tmds = []
    mmds = []
    for idx, (gt_mesh, prediction) in enumerate(tqdm.tqdm(zip(gt_meshes, predictions), total=num_examples)):
        data_dict = dataset[idx]
        pred_mesh, pred_mesh_set, pred_metadata = prediction
        num_points = 5000
        # sample pointclouds from the mesh using trimesh
        pred_mesh_points = np.stack(
            [np.asarray(trimesh.sample.sample_surface_even(msh, num_points)[0]) for msh in pred_mesh_set])
        pred_mesh_points = torch.from_numpy(pred_mesh_points).float().cuda()  # (num_meshes, num_points, 3)
        gt_mesh_points = np.asarray(trimesh.sample.sample_surface_even(gt_mesh, num_points)[0])
        gt_mesh_points = torch.from_numpy(gt_mesh_points).float().cuda().unsqueeze(0)  # (1, num_points, 3)
        # compute the total matching distance
        tmd = total_matching_distance(pred_mesh_points).item()
        tmds.append(tmd)
        # print('tmd', tmd)
        # compute the minimal matching distance
        mmd = minimal_matching_distance(gt_mesh_points, pred_mesh_points).item()
        # print('mmd', mmd)
        mmds.append(mmd)

    print("TMD: %f (%f)" % (np.mean(tmds), np.std(tmds)))
    print("MMD: %f (%f)" % (np.mean(mmds), np.std(mmds)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize generated results.")
    parser.add_argument("dataset_cfg", type=str, help="Path to dataset config file.")
    parser.add_argument("gen_dir", type=str, help="Path to directory containing generated results.")
    parser.add_argument("--mode", "-m", type=str, default="test", help="Dataset mode to use.")
    parser.add_argument("--offset", "-o", type=int, default=0, help="Offset to use for visualization.")
    args = parser.parse_args()

    eval_results(args.dataset_cfg, args.gen_dir, args.mode, args.offset)
