import argparse
import os

import matplotlib as mpl
import cv2

from usdf.utils.model_utils import load_dataset_from_config
from usdf.utils.results_utils import load_pred_results, load_gt_results

import matplotlib.pyplot as plt


def make_slices_images(dataset_cfg: str, gen_dir: str, mode: str = "test", vis: bool = False):
    # Load dataset.
    dataset_cfg, dataset = load_dataset_from_config(dataset_cfg, dataset_mode=mode)
    num_examples = len(dataset)
    dataset_cfg = dataset_cfg["data"][mode]

    # Load ground truth information.
    gt_meshes = load_gt_results(dataset, dataset_cfg, num_examples)

    # Load predicted information.
    pred_meshes, pred_slices, metadata = load_pred_results(gen_dir, num_examples)

    for idx, pred_slice in enumerate(pred_slices):
        mean_image = pred_slice["mean"]
        uncertainty_image = pred_slice["uncertainty"]

        if vis:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(mean_image)
            ax2.imshow(uncertainty_image)
            plt.show()

        jet = mpl.colormaps.get_cmap('jet')
        cNorm = mpl.colors.Normalize(vmin=uncertainty_image.min(), vmax=0.2)  # uncertainty_image.max())
        scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=jet)
        uncertainty_color_image = (scalarMap.to_rgba(uncertainty_image.flatten()) * 255) \
            .astype("uint8").reshape(100, 100, -1)

        jet = mpl.colormaps.get_cmap('jet')
        cNorm = mpl.colors.Normalize(vmin=mean_image.min(), vmax=mean_image.max())
        scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=jet)
        mean_color_image = (scalarMap.to_rgba(mean_image.flatten()) * 255) \
            .astype("uint8").reshape(100, 100, -1)

        if vis and False:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(mean_color_image)
            ax2.imshow(uncertainty_color_image)
            plt.show()

        cv2.imwrite(os.path.join(gen_dir, "slice_mean_%d.png" % idx), cv2.cvtColor(mean_color_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(gen_dir, "slice_uncertainty_%d.png" % idx),
                    cv2.cvtColor(uncertainty_color_image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize generated results.")
    parser.add_argument("dataset_cfg", type=str, help="Path to dataset config file.")
    parser.add_argument("gen_dir", type=str, help="Path to directory containing generated results.")
    parser.add_argument("--mode", "-m", type=str, default="test", help="Dataset mode to use.")
    parser.add_argument("--vis", "-v", action="store_true", help="Visualize results.")
    parser.set_defaults(vis=False)
    args = parser.parse_args()

    make_slices_images(args.dataset_cfg, args.gen_dir, args.mode, args.vis)
