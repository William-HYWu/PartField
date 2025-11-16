import os.path
import random

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import mmint_utils
from usdf import config
from usdf.utils.args_utils import get_model_dataset_arg_parser, load_model_dataset_from_args
from usdf.utils.model_utils import load_generation_cfg


def one_d_inference_test(model_cfg, model, model_file, dataset, device, gen_args: dict, animate: bool,
                         out_dir: str = None):
    model.eval()

    if out_dir is not None:
        mmint_utils.make_dir(out_dir)

    # Load generate cfg, if present.
    generation_cfg = load_generation_cfg(model_cfg, model_file)
    if gen_args is not None:
        generation_cfg.update(gen_args)

    # Load generator.
    generator = config.get_generator(model_cfg, model, generation_cfg, device)

    for idx in range(len(dataset)):
        data_dict = dataset[idx]

        latent, latent_metadata = generator.generate_latent(data_dict)

        # Visualize latent optimization.
        gt_angle = data_dict["angle"]
        z_history = latent_metadata["z_history"]
        z_history = np.concatenate(z_history, axis=1)

        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        # Draw unit circle - all latents lie on this circle (after sinusoidal embedding).
        circle = plt.Circle((0, 0), 1, color='black', fill=False)
        ax.add_patch(circle)

        # Plot the gt.
        ax.scatter(np.cos(gt_angle), np.sin(gt_angle), s=100, c='r', label="gt", marker="*")

        # Plot the starting latent.
        latent_scatter = []
        for l_idx in range(z_history.shape[0]):
            latent_scatter.append(
                ax.scatter(np.cos(z_history[l_idx, 0]), np.sin(z_history[l_idx, 0]), c='b', marker="o",
                           label="latent_%d" % l_idx)
            )
        iter_label = ax.text(0.8, 0.95, "Iteration: 0", transform=ax.transAxes, ha="center")

        # Make slider.
        def update_plotter(val):
            idx = int(val)
            for l_idx in range(z_history.shape[0]):
                latent_scatter[l_idx].set_offsets(
                    np.array([np.cos(z_history[l_idx, idx]), np.sin(z_history[l_idx, idx])]).T)
            iter_label.set_text("Iteration: %d" % idx)
            fig.canvas.draw_idle()

        if animate:
            anim = animation.FuncAnimation(fig, update_plotter, frames=z_history.shape[1], interval=10,
                                           repeat=out_dir is None)

            if out_dir is not None:
                out_fn = os.path.join(out_dir, f"opt_{idx}.gif")
                anim.save(out_fn, dpi=80, writer='ffmpeg')
            else:
                plt.show()
        else:
            fig.subplots_adjust(bottom=0.25)
            ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
            slider = plt.Slider(ax=ax_slider, label="Iteration", valmin=0, valmax=z_history.shape[1] - 1, valinit=0,
                                valstep=1)
            slider.on_changed(update_plotter)
            plt.show()


if __name__ == '__main__':
    parser = get_model_dataset_arg_parser()
    parser.add_argument("--gen_args", type=yaml.safe_load, default=None, help="Generation args.")
    parser.add_argument("--animate", "-a", action="store_true", help="Animate the latent optimization.")
    parser.set_defaults(animate=False)
    parser.add_argument("--out_dir", "-o", type=str, default=None, help="Output directory.")
    args = parser.parse_args()

    # Seed for repeatability.
    torch.manual_seed(10)
    np.random.seed(10)
    random.seed(10)

    model_cfg_, model_, dataset_, device_ = load_model_dataset_from_args(args)
    one_d_inference_test(model_cfg_, model_, args.model_file, dataset_, device_, args.gen_args, args.animate,
                         args.out_dir)
