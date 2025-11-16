import os.path
import random

import numpy as np
import torch

import mmint_utils
import yaml
from usdf import config
from usdf.utils.args_utils import get_model_dataset_arg_parser, load_model_dataset_from_args
from usdf.utils.model_utils import load_generation_cfg
from tqdm import trange, tqdm

from usdf.utils.results_utils import write_results, load_gt_results
from usdf.visualize import visualize_mesh, visualize_mesh_set


def generate(model_cfg, model, model_file, dataset, device, out_dir, gen_args: dict, vis: bool = False,
             dataset_cfg: dict = None):
    model.eval()

    # Load generate cfg, if present.
    generation_cfg = load_generation_cfg(model_cfg, model_file)
    if gen_args is not None:
        generation_cfg.update(gen_args)

    # Load generator.
    generator = config.get_generator(model_cfg, model, generation_cfg, device)

    # Load ground truth information.
    gt_meshes = load_gt_results(dataset, dataset_cfg, len(dataset))

    # Determine what to generate.
    generate_mesh = generator.generates_mesh
    generate_mesh_set = generator.generates_mesh_set

    # Create output directory.
    if out_dir is not None:
        mmint_utils.make_dir(out_dir)

    # Dump any generation arguments to out directory.
    mmint_utils.dump_cfg(os.path.join(out_dir, "metadata.yaml"), generation_cfg)

    # Go through dataset and generate!
    for idx, gt_mesh in enumerate(tqdm(gt_meshes)):
        data_dict = dataset[idx]
        metadata = {}
        mesh = None
        mesh_set = None

        if generate_mesh:
            mesh, metadata_mesh = generator.generate_mesh(data_dict, metadata)
            metadata = mmint_utils.combine_dict(metadata, metadata_mesh)

            if vis:
                visualize_mesh(data_dict, mesh, gt_mesh)

        if generate_mesh_set:
            mesh_set, metadata_mesh_set = generator.generate_mesh_set(data_dict, metadata)
            metadata = mmint_utils.combine_dict(metadata, metadata_mesh_set)

            if vis:
                visualize_mesh_set(data_dict, mesh_set, gt_mesh, metadata_mesh_set)

        write_results(out_dir, mesh, mesh_set, metadata, idx)


if __name__ == '__main__':
    parser = get_model_dataset_arg_parser()
    parser.add_argument("--out", "-o", type=str, help="Optional out directory to write generated results to.")
    parser.add_argument("--gen_args", type=yaml.safe_load, default=None, help="Generation args.")
    parser.add_argument("-v", "--vis", action="store_true", help="Visualize generated results.")
    args = parser.parse_args()

    # Seed for repeatability.
    torch.manual_seed(10)
    np.random.seed(10)
    random.seed(10)

    model_cfg_, dataset_cfg_, model_, dataset_, device_ = load_model_dataset_from_args(args)
    dataset_cfg_ = dataset_cfg_["data"][args.mode]

    out = args.out
    if out is None:
        out = os.path.join(model_cfg_["training"]["out_dir"], "out", args.mode)
        mmint_utils.make_dir(out)

    generate(model_cfg_, model_, args.model_file, dataset_, device_, out, args.gen_args, vis=args.vis,
             dataset_cfg=dataset_cfg_)
