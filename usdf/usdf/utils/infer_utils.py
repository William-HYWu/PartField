from collections import Callable

import numpy as np
import torch
import torch.optim as optim

import torch.nn as nn
from tqdm import trange


def inference_by_optimization(model: nn.Module, loss_fn: Callable, init_fn: Callable, latent_size: int,
                              num_examples: int, num_latent: int, data_dict: dict,
                              vis_fn: Callable = None,
                              inf_params=None, device: torch.device = None, verbose: bool = False):
    """
    Helper with basic inference by optimization structure. Repeatedly calls loss function with the specified
    data/loss function and updates latent inputs accordingly.

    Args:
        model (nn.Module): network model
        loss_fn (Callable): loss function. Should take in model, current latent, data dictionary, and device and return loss.
        init_fn (Callable): initialization function. Should init the given embedding.
        latent_size (int): specify latent space size.
        num_examples (int): number of examples to run inference on.
        num_latent (int): number of latents to generate per example.
        data_dict (dict): data dictionary for example(s) we are inferring for.
        vis_fn (Callable): visualization function.
        inf_params (dict): inference hyper-parameters.
        device (torch.device): pytorch device.
        verbose (bool): be verbose.
    """
    if inf_params is None:
        inf_params = {}
    model.eval()

    # Load inference hyper parameters.
    lr = inf_params.get("lr", 3e-2)
    num_steps = inf_params.get("iter_limit", 300)

    # Initialize latent code.
    z_init_weights = init_fn(num_examples, num_latent, latent_size, device)  # Call provided init function.
    z_ = nn.Embedding(num_examples * num_latent, latent_size, dtype=torch.float32)
    z_.weight = nn.Parameter(z_init_weights.reshape([num_examples * num_latent, latent_size]))
    z_.requires_grad_(True).to(device)

    optimizer = optim.Adam(z_.parameters(), lr=lr)

    # Start optimization procedure.
    z = z_.weight.reshape([num_examples, num_latent, latent_size])

    # Store history of latents/loss.
    z_history = []
    loss_history = []

    iter_idx = 0
    if verbose:
        range_ = trange(num_steps)
    else:
        range_ = range(num_steps)
    for iter_idx in range_:
        optimizer.zero_grad()

        # Store latent history.
        z_history.append(z.detach().cpu().numpy().copy())

        if vis_fn is not None:
            vis_fn(z)

        loss, loss_ind = loss_fn(model, z, data_dict, device)

        # Store loss history.
        loss_history.append(loss_ind.detach().cpu().numpy().copy())

        loss.backward()
        optimizer.step()

        if verbose:
            range_.set_postfix(loss=loss.item())
    if verbose:
        range_.close()

    _, final_loss = loss_fn(model, z, data_dict, device)
    z_history.append(z.detach().cpu().numpy().copy())
    loss_history.append(final_loss.detach().cpu().numpy().copy())

    results_z = z_.weight.reshape([num_examples, num_latent, latent_size])
    return results_z, {"final_loss": final_loss, "iters": iter_idx + 1,
                       "z_history": z_history, "loss_history": loss_history}
