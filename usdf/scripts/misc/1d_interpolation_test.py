import random

import numpy as np
import torch
import yaml
import tqdm
from vedo import Plotter, Mesh

from usdf import config
from usdf.utils.args_utils import get_model_dataset_arg_parser, load_model_dataset_from_args
from usdf.utils.model_utils import load_generation_cfg


def one_d_interpolation_test(model_cfg, model, model_file, device, gen_args: dict):
    model.eval()

    # Load generate cfg, if present.
    generation_cfg = load_generation_cfg(model_cfg, model_file)
    if gen_args is not None:
        generation_cfg.update(gen_args)
    generation_cfg["gen_from_known_latent"] = True

    # Load generator.
    generator = config.get_generator(model_cfg, model, generation_cfg, device)

    # Visualize interpolation.
    meshes = []
    for angle in tqdm.tqdm(np.linspace(0.0, 2 * np.pi, num=128)):
        data_dict = {"angle": angle, "example_idx": -1}

        mesh, metadata = generator.generate_mesh(data_dict, {})
        meshes.append(mesh)

    vis_meshes = [Mesh([mesh.vertices, mesh.faces]) for mesh in meshes]

    # Setup visualization.
    plot = Plotter()
    plot.at(0).add(vis_meshes[0], render=False)

    def update_plotter(slider, event):
        # idx = int(slider.value * (len(vis_meshes) - 1) / (2 * np.pi))
        idx = int(slider.value)
        plot.at(0).remove(plot.get_meshes(0)[0])
        plot.at(0).add(vis_meshes[idx])

    plot.add_slider(update_plotter, xmin=0, xmax=len(vis_meshes) - 1,  # xmax=2 * np.pi, value=0,
                    pos="bottom", title="Angle")
    plot.interactive().show()


if __name__ == '__main__':
    parser = get_model_dataset_arg_parser()
    parser.add_argument("--gen_args", type=yaml.safe_load, default=None, help="Generation args.")
    args = parser.parse_args()

    # Seed for repeatability.
    torch.manual_seed(10)
    np.random.seed(10)
    random.seed(10)

    model_cfg_, model_, dataset_, device_ = load_model_dataset_from_args(args)
    one_d_interpolation_test(model_cfg_, model_, args.model_file, device_, args.gen_args)
