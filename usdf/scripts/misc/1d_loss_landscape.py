import random

import numpy as np
import torch
import yaml

import matplotlib.pyplot as plt

from usdf.utils.args_utils import get_model_dataset_arg_parser, load_model_dataset_from_args


def surface_loss_fn(model, latent, data_dict, device):
    # Pull out relevant data.
    surface_coords_ = torch.from_numpy(data_dict["partial_pointcloud"]).to(device).float().unsqueeze(0)
    surface_coords_ = surface_coords_.repeat(latent.shape[0], 1, 1)

    # If we are using the angle as the input, we must apply the sinusoidal embedding.
    if model.use_angle:
        latent = torch.cat([torch.sin(latent), torch.cos(latent)], dim=-1)

    # Predict with updated latents.
    pred_dict_ = model.forward(surface_coords_, latent)

    # Loss: all points on surface should have SDF = 0.0.
    sdf_loss = torch.mean(torch.abs(pred_dict_["sdf"]), dim=-1)
    loss = sdf_loss
    return loss.mean(), loss


def one_d_loss_landscape(model_cfg, model, model_file, dataset, device, offset: int = 0):
    model.eval()
    n = 256  # Number of points to evaluate loss landscape at.

    for idx in range(offset, len(dataset)):
        data_dict = dataset[idx]

        angles = np.linspace(0.0, 2 * np.pi, num=n)
        loss_values = []

        for angle in angles:
            latent = torch.tensor(angle).to(device).float().unsqueeze(0)

            loss, _ = surface_loss_fn(model, latent, data_dict, device)
            loss_values.append(loss.item())

        # Plot loss landscape.
        plt.plot(angles, loss_values)
        plt.xlabel("Angle")
        plt.ylabel("Loss")
        plt.show()


if __name__ == '__main__':
    parser = get_model_dataset_arg_parser()
    parser.add_argument("--offset", "-o", type=int, default=0, help="Offset into dataset to use.")
    args = parser.parse_args()

    model_cfg_, model_, dataset_, device_ = load_model_dataset_from_args(args)
    one_d_loss_landscape(model_cfg_, model_, args.model_file, dataset_, device_, args.offset)
