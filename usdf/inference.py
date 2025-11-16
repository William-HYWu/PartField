#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import os
import random
import time
import torch
from dataset import data
from models.decoder import Decoder
from utils import mesh
from tqdm import tqdm
import numpy as np
import torch.autograd as autograd
import wandb
import matplotlib.pyplot as plt
from analyze_latent_vectors import load_latent_vectors
from sklearn.decomposition import PCA
import math

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seeds set to {seed}")
    
class RBF(torch.nn.Module):
    def __init__(self, sigma=None):
        super(RBF, self).__init__()
        self.sigma = sigma

    def forward(self, X, Y):
        #X corresponds to x_j, Y corresponds to x_i
        n_particles = X.size(0)
        
        XX = X.pow(2).sum(dim=1, keepdim=True)  # [n, 1]
        YY = Y.pow(2).sum(dim=1, keepdim=True)  # [n, 1]
        XY = X @ Y.t()  # [n, n]
        
        pairwise_dists = XX + YY.t() - 2 * XY  # [n, n]
        # [x1^2+x1^2-2*x1*x1, x1^2+x2^2-2*x1*x2, ..., x1^2+xn^2-2*x1*xn]
        # [x2^2+x1^2-2*x2*x1, x2^2+x2^2-2*x2*x2, ..., x2^2+xn^2-2*x2*xn]
        # ...
        # [xn^2+x1^2-2*xn*x1, xn^2+xn^2-2*xn*xn, ..., xn^2+xn^3-4*xi*xi]

        if self.sigma is None:
            pdist_np = pairwise_dists.detach().cpu().numpy()
            mask = np.eye(n_particles, dtype=bool)
            pdist_no_diag = pdist_np[~mask]
            if len(pdist_no_diag) > 0:
                h = np.median(pdist_no_diag)
                sigma = np.sqrt(h / np.log(n_particles + 1))
            else:
                sigma = 1.0
        else:
            sigma = self.sigma
        sigma=max(sigma, 1e-2)
            
        gamma = 1.0 / (2 * sigma ** 2)
        K = torch.exp(-gamma * pairwise_dists)
        diff = X.unsqueeze(1) - Y.unsqueeze(0)  # [n, n, d]
        # [x1 - x1, x1 - x2, ..., x1 - xn]
        # [x2 - x1, x2 - x2, ..., x2 - xn]
        # ...
        # [xn - x1, xn - x2, ..., xn - xn]
        
        grad_K = -2 * gamma * K.unsqueeze(-1) * diff  # [n, n, d]
        #[K(x1,x1)*(x1 - x1) + K(x1,x2)*(x1 - x2) + ... + K(x1,xn)*(x1 - xn)]
        #[K(x2,x1)*(x2 - x1) + K(x2,x2)*(x2 - x2) + ... + K(x2,xn)*(x2 - xn)]
        # ...
        #[K(xn,x1)*(xn - x1) + K(xn,x2)*(xn - x2) + ... + K(xn,xn)*(xn - xn)]
        
        grad_K = grad_K.transpose(0,1)  # [n, n, d]
        
        grad_K = grad_K.sum(dim=1)  # [n, d] - sum over particles
        #[K(x1,x1), K(x1,x2), ..., K(x1,xn)]
        #[K(x2,x1), K(x2,x2), ..., K(x2,xn)]
        # ...
        #[K(xn,x1), K(xn,x2), ..., K(xn,xn)]
        

        return K, grad_K
    
def reconstruct_svgd(
    decoder,
    num_iterations,
    latent_size,
    test_sdf,
    stat,
    clamp_dist,
    train_init=True,
    num_samples=10000,
    lr=5e-4,
    l2reg=False,
    num_particles=8,
    log_wandb=False,
    shape_name="unknown",
    replace_worst=True,
    use_train_latents=False,
    train_vectors=None,
    repulsion_weight=1.0,
):
    sdf_data = data.unpack_sdf_samples_from_ram(test_sdf, num_samples).to(device)
    num_samples = sdf_data.shape[0]
    # Outer loop for SVGD refinement
    outer_iter_limit = 10
    num_perturb = num_particles // 2
    latent_sigma = 0.1
    inner_iter = num_iterations // outer_iter_limit
    # If training latent vectors are provided, compute their empirical mean/std.
    # We can either use the raw training vectors as initialization (use_train_latents=True)
    # or sample initial particles from the empirical Gaussian N(mean, std) estimated from them.
    if train_vectors is not None and train_init:
        # Ensure tensor on CPU for numpy ops and also keep device tensors for sampling
        train_vecs_cpu = train_vectors.detach().cpu()
        train_mean = torch.mean(train_vecs_cpu, dim=0)
        train_std = torch.std(train_vecs_cpu, dim=0)

        # Save sorted training latents (2D PCA) for visualization if desired
        try:
            pca = PCA(n_components=2)
            train_latent_np = train_vecs_cpu.numpy()
            pca.fit(train_latent_np)
            train_latent_2d = pca.transform(train_latent_np)
            x = train_latent_2d[:, 0]
            y = train_latent_2d[:, 1]
            angles = np.arctan2(x, y)
            sorted_indices = np.argsort(angles)
            sorted_latents = train_latent_np[sorted_indices]
            print("Saved sorted training latents for later visualizations.")
            np.save("sorted_train_latents.npy", sorted_latents)
        except Exception:
            # PCA/visualization is optional; continue if it fails
            pass

        if use_train_latents:
            print("Using training latent vectors.")
            assert train_vectors.shape[0] >= num_particles, "Not enough training latent vectors for the number of particles."
            # Detach and clone to create leaf tensors that can be optimized
            latent = train_vectors[:num_particles].detach().clone().to(device)
        else:
            print("Sampling initial particles from empirical Gaussian of training latent vectors.")
            # Initialize particles by sampling from empirical Gaussian (mean,std) of train vectors
            mean = train_mean.unsqueeze(0).expand(num_particles, -1).to(device)
            std = train_std.unsqueeze(0).expand(num_particles, -1).to(device)
            # Numerical stability: ensure std not too small
            std = std.abs()
            std = torch.clamp(std, min=1e-6)
            latent = torch.normal(mean=mean, std=std).to(device)
    else:
        # No training latents provided: fall back to original initialization
        print("No training latent vectors provided; using prior initialization.")
        train_vecs_cpu = train_vectors.detach().cpu()
        train_mean = torch.mean(train_vecs_cpu, dim=0)
        train_std = torch.std(train_vecs_cpu, dim=0)
        try:
            pca = PCA(n_components=2)
            train_latent_np = train_vecs_cpu.numpy()
            pca.fit(train_latent_np)
            train_latent_2d = pca.transform(train_latent_np)
            x = train_latent_2d[:, 0]
            y = train_latent_2d[:, 1]
            angles = np.arctan2(x, y)
            sorted_indices = np.argsort(angles)
            sorted_latents = train_latent_np[sorted_indices]
            print("Saved sorted training latents for later visualizations.")
            np.save("sorted_train_latents.npy", sorted_latents)
        except Exception:
            # PCA/visualization is optional; continue if it fails
            pass
        
        if type(stat) == type(0.1):
            latent = torch.ones(num_particles, latent_size).normal_(mean=0, std=1.0/math.sqrt(latent_size)).to(device)
        else:
            # Generate different random samples for each particle
            mean = stat[0].detach().unsqueeze(0).expand(num_particles, -1)
            std = stat[1].detach().unsqueeze(0).expand(num_particles, -1)
            latent = torch.normal(mean, std).to(device)


    loss_l1 = torch.nn.L1Loss(reduction='none')
    svgd_kernel = RBF()

    # Track latent trajectories for PCA visualization
    latent_history = []  # Store latent codes at each iteration
    latent_history.append(latent.detach().cpu().numpy().copy())

    # Outer loop for particle replacement
    for outer_iter in range(outer_iter_limit):
        latent = latent.requires_grad_(True)
        opt = torch.optim.Adam([latent], lr=lr)
        
        # SVGD inner loop
        pbar = tqdm(range(inner_iter), desc=f"SVGD Outer {outer_iter+1}/{outer_iter_limit}", unit="it")
        for e in pbar:
            decoder.eval()
            sdf_data = data.unpack_sdf_samples_from_ram(test_sdf, num_samples).to(device)
            num_samples = sdf_data.shape[0]
            perm = torch.randperm(sdf_data.size(0))
            sdf_data = sdf_data[perm]
            xyz = sdf_data[:, 0:3]
            sdf_gt = sdf_data[:, 3].unsqueeze(1)
            sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

            # Process in batches to avoid OOM - update on each batch
            batch_size = 10000  # Adjust based on available memory
            total_loss_epoch = 0.0
            num_batches = 0
            
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                batch_samples = end_idx - i
                
                xyz_batch = xyz[i:end_idx]
                sdf_gt_batch = sdf_gt[i:end_idx]
                
                # Expand for all particles
                xyz_expand = xyz_batch.unsqueeze(0).expand(num_particles, batch_samples, 3)
                latent_inputs = latent.unsqueeze(1).expand(num_particles, batch_samples, latent_size)
                
                inputs_batch = torch.cat([latent_inputs, xyz_expand], dim=2).reshape(-1, latent_size + 3)
                pred_sdf_batch = decoder.inference(inputs_batch).reshape(num_particles, batch_samples, 1)
                pred_sdf_batch = torch.clamp(pred_sdf_batch, -clamp_dist, clamp_dist)

                # Expand ground truth SDF to match particles
                sdf_target_batch = sdf_gt_batch.unsqueeze(0).expand(num_particles, batch_samples, 1)
                batch_loss = loss_l1(pred_sdf_batch, sdf_target_batch).mean(dim=(1,2))
                
                # Add L2 regularization to each particle
                if l2reg:
                    batch_loss = batch_loss + 1e-4 * torch.mean(latent.pow(2), dim=1)
                #batch_loss: shape [num_particles]
                
                # Compute gradients for this batch only
                grad_latent = -autograd.grad(batch_loss.sum(), latent, retain_graph=False)[0]

                K_XX, grad_k = svgd_kernel(latent, latent)

                # Apply repulsion weight to control diversity
                phi = (K_XX.t() @ grad_latent + repulsion_weight * grad_k) / num_particles
                
                # Update latent with this batch's gradient
                latent.grad = -phi
                opt.step()
                opt.zero_grad()
                
                total_loss_epoch += batch_loss.mean().item()
                num_batches += 1
            
            # Store latent codes after each iteration for trajectory tracking
            latent_history.append(latent.detach().cpu().numpy().copy())
            
            # Average loss across all batches for logging
            avg_loss = total_loss_epoch / num_batches
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
            
            # Log to wandb
            if log_wandb:
                wandb.log({
                    f"{shape_name}/svgd_loss_mean": avg_loss,
                    f"{shape_name}/iteration": outer_iter * inner_iter + e,
                    f"{shape_name}/outer_iteration": outer_iter,
                })
        if not replace_worst:
            continue

    return avg_loss, latent.detach(), sdf_data, latent_history

def reconstruct(
    decoder,
    num_iterations,
    latent_size,
    test_sdf,
    stat,
    clamp_dist,
    num_samples=100000,
    lr=5e-4,
    l2reg=False,
    log_wandb=False,
    shape_name="unknown",
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)
    se2_matrices, _ = data.generate_random_se2(5)
    se2_matrices = torch.from_numpy(se2_matrices).float().to(device)
    latents = []
    sdf_datas = []

    for se2_matrix in se2_matrices:

        if type(stat) == type(0.1):
            latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).to(device)
        else:
            latent = torch.normal(stat[0].detach(), stat[1].detach()).to(device)

        latent.requires_grad = True

        optimizer = torch.optim.Adam([latent], lr=lr)

        loss_num = 0
        loss_l1 = torch.nn.L1Loss()
        

        for e in tqdm(range(num_iterations), desc="Reconstructing", unit="it"):

            decoder.eval()
            sdf_data = data.unpack_sdf_samples_from_ram(
                test_sdf, num_samples
            ).to(device)
            perm = torch.randperm(sdf_data.size(0))
            sdf_data = sdf_data[perm]
            xyz = sdf_data[:, 0:3]
            xy = xyz[:, :2]
            rotation = se2_matrix[:2, :2]
            translation = se2_matrix[:2, 2]
            transformed_xy = xy @ rotation.T + translation
            xyz[:, :2] = transformed_xy
            sdf_data[:, 0:3] = xyz
            
            sdf_gt = sdf_data[:, 3].unsqueeze(1)

            sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

            adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

            optimizer.zero_grad()
            num_samples = sdf_data.shape[0]

            latent_inputs = latent.expand(num_samples, -1)

            inputs = torch.cat([latent_inputs, xyz], 1).to(device)

            pred_sdf = decoder.inference(inputs)

            # TODO: why is this needed?
            if e == 0:
                pred_sdf = decoder.inference(inputs)
            pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

            loss = loss_l1(pred_sdf, sdf_gt)
            if l2reg:
                loss += 1e-4 * torch.mean(latent.pow(2))
            loss.backward()
            optimizer.step()

            loss_num = loss.cpu().data.numpy()
            
            # Log to wandb
            if log_wandb:
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({
                    f"{shape_name}/standard_loss": loss_num,
                    f"{shape_name}/learning_rate": current_lr,
                    f"{shape_name}/iteration": e,
                })
        latents.append(latent.detach())
        sdf_datas.append(sdf_data)
    return loss_num, latents, sdf_datas


def plot_latent_trajectories(latent_history, save_path, train_latent=None, shape_name="mugs"):
    """
    Plot the trajectories of latent codes in 2D PCA space.
    
    Args:
        latent_history: List of numpy arrays, each of shape [num_particles, latent_size]
        save_path: Path to save the plot
        shape_name: Name of the shape for the title
    """
    num_iterations = len(latent_history)
    num_particles = latent_history[0].shape[0]
    
    # Flatten all latent codes across time and particles for PCA
    # Shape: [num_iterations * num_particles, latent_size]
    all_latents = np.vstack(latent_history)
    
    # Fit PCA on all latent codes
    pca = PCA(n_components=2)
    if train_latent is not None:
        # Include training latent codes in PCA fitting
        train_latent_np = train_latent.detach().cpu().numpy()
        pca.fit(train_latent_np)
        train_latent_2d = pca.transform(train_latent_np)
        all_latents_2d = pca.transform(all_latents)
    else:
        all_latents_2d = pca.fit_transform(all_latents)
    latents_2d = all_latents_2d.reshape(num_iterations, num_particles, 2)
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Color map for particles
    colors = plt.cm.tab10(np.linspace(0, 1, num_particles))
    colors_train = plt.cm.tab10(np.linspace(0, 1, train_latent_np.shape[0])) if train_latent is not None else None

    # Plot training latent codes
    if colors_train is not None:
        plt.scatter(train_latent_2d[:, 0], train_latent_2d[:, 1], 
                    marker='X', s=200, color=colors_train,
                    label='Training Latents', edgecolors='black', linewidths=2)

    # Plot trajectory for each particle
    for particle_idx in range(num_particles):
        trajectory = latents_2d[:, particle_idx, :]
        
        # Plot the trajectory line with directional markers
        plt.plot(trajectory[:, 0], trajectory[:, 1], 
                alpha=0.7, linewidth=2.5, color=colors[particle_idx],
                label=f'Particle {particle_idx}')
        
        # Mark the start point (larger, hollow circle)
        plt.scatter(trajectory[0, 0], trajectory[0, 1], 
                   marker='o', s=200, facecolors='white',
                   edgecolors=colors[particle_idx], linewidths=3, zorder=10)
        
        # Mark the end point (filled star)
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1], 
                   marker='*', s=400, color=colors[particle_idx],
                   edgecolors='black', linewidths=2, zorder=10)

    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=14)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=14)
    plt.title(f'SVGD Particle Trajectories in Latent Space\n{shape_name}\n'
              f'○ = Start, ★ = End, {num_iterations} total updates', fontsize=14, pad=20)
    plt.legend(loc='upper left', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved latent trajectory plot: {save_path}")


if __name__ == "__main__":
    
    set_random_seeds(42)

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--model_name",
        "-m",
        dest="model_name",
        required=False,
        default="best_model_epoch_500_loss_0.001576.pth",
        help="The model name to use for reconstruction. This is the name of the "
        + "file in the checkpoints subdirectory of the experiment directory.",
    )
    arg_parser.add_argument(
        "--name",
        "-n",
        dest="name",
        required=False,
        default="mugs",
        help="A name for this reconstruction, used for the output mesh and latent code "
        + "filenames.",
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=800,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip meshes which have already been reconstructed.",
    )
    arg_parser.add_argument(
        "--use_svgd",
        "-u",
        default=False,
        help="Use SVGD to reconstruct multiple diverse outputs.",
    )
    arg_parser.add_argument(
        "--use_wandb",
        "-w",
        action="store_true",
        help="Use Weights & Biases for logging loss curves.",
    )
    arg_parser.add_argument(
        "--wandb_project",
        dest="wandb_project",
        default="usdf-reconstruction",
        help="Weights & Biases project name.",
    )
    arg_parser.add_argument(
        "--replace_worst",
        default=False,
        action="store_true",
        help="Disable replacing worst particles in SVGD.",
    )
    arg_parser.add_argument(
        "--latents",
        default=None,
        help="Path to latent codes file for PCA plotting.",
    )
    arg_parser.add_argument(
        "--gpu",
        dest="gpu_id",
        type=int,
        default=0,
        help="GPU device ID to use (default: 0).",
    )
    arg_parser.add_argument(
        "--use_train_latents",
        action="store_true",
        help="Use training latent vectors for initialization."
    )
    arg_parser.add_argument(
        "--repulsion_weight",
        type=float,
        default=1.0,
        help="Weight for SVGD repulsion term (default: 1.0). Higher values increase diversity."
    )
    arg_parser.add_argument(
        "--use_se2",
        type=bool,
        default=False,
        help="Use SE(2) transformations for initialization.",
    )
    
    arg_parser.add_argument(
        "--train_init",
        action="store_true",
        help="Initialize SVGD particles from training latent vectors.",
    )
    args = arg_parser.parse_args()
    
    # Set GPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"Using GPU: {args.gpu_id} ({torch.cuda.get_device_name(args.gpu_id)})")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).to(device)
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    specs_filename = os.path.join(args.experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    latent_size = specs["CodeLength"]

    decoder = Decoder(latent_size, **specs["NetworkSpecs"])

    saved_model_state = torch.load(
        os.path.join(
            "checkpoints", args.name, args.model_name
        )
    )

    decoder.load_state_dict(saved_model_state)

    decoder = decoder.to(device)

    npz_filenames = data.get_instance_filenames(split_name="test",config_path="configs/dataset/mugs_dataset_config_oriented_testset.yaml")

    random.shuffle(npz_filenames)

    err_sum = 0.0
    num_particles = 8  # Number of diverse outputs per shape
    save_latvec_only = False
    rerun = 0

    reconstruction_dir = os.path.join(
        "reconstruct", args.name
    )

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, "meshes"
    )
    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)

    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, "codes"
    )
    if not os.path.isdir(reconstruction_codes_dir):
        os.makedirs(reconstruction_codes_dir)
    
    reconstruction_points_dir = os.path.join(
        reconstruction_dir, "points"
    )
    
    if args.latents is not None:
        train_latent_vecs = load_latent_vectors(args.latents).to(device)
        train_latent_vecs = train_latent_vecs.weight[:]
    elif args.use_se2:
        _, se2_vecs = data.generate_random_se2(num_samples=100)
        se2_vecs = torch.from_numpy(se2_vecs).float().to(device)
        train_latent_vecs = decoder.get_latent(se2_vecs).detach()
        args.latents = "SE2_generated_latents"
    else:
        train_latent_vecs = None

    if not os.path.isdir(reconstruction_points_dir):
        os.makedirs(reconstruction_points_dir)

    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"reconstruction_{args.name}_{args.model_name}",
            config={
                "model_name": args.model_name,
                "experiment": args.name,
                "iterations": int(args.iterations),
                "use_svgd": args.use_svgd,
                "num_particles": num_particles,
                "latent_size": latent_size,
            }
        )
        print(f"Initialized wandb project: {args.wandb_project}")

    for ii, npz in enumerate(npz_filenames):
        if "npz" not in npz:
            continue
        full_filename = npz
        data_sdf = data.read_sdf_samples_into_ram(full_filename)
        
        # Extract shape name for logging
        shape_name = os.path.basename(npz)[:-4]  # Remove .npz extension

        # SVGD latent code optimization for multiple plausible outputs
        if args.use_svgd:
            err, latents, sample_sdf_data, latent_history = reconstruct_svgd(
                decoder,
                int(args.iterations),
                latent_size,
                data_sdf,
                0.01,
                0.1,
                train_init=args.train_init,
                num_samples=specs["SamplesPerScene"],
                lr=5e-3,
                l2reg=True,
                num_particles=num_particles,
                log_wandb=args.use_wandb,
                shape_name=shape_name,
                replace_worst=args.replace_worst,
                use_train_latents=args.use_train_latents,
                train_vectors=train_latent_vecs,
                repulsion_weight=args.repulsion_weight,
            )
            
            # Plot latent trajectories
            trajectory_dir = os.path.join(reconstruction_dir, "trajectories")
            if not os.path.isdir(trajectory_dir):
                os.makedirs(trajectory_dir)
            trajectory_filename = os.path.join(trajectory_dir, f"{shape_name}_trajectory.png")
            plot_latent_trajectories(latent_history, trajectory_filename,train_latent_vecs, shape_name)
            #save latent history as npy
            latent_history_filename = os.path.join(trajectory_dir, f"{shape_name}_latent_history.npy")
            np.save(latent_history_filename, np.array(latent_history))
            print(f"Saved latent history: {latent_history_filename}")
            
        else:
            err, latents, sample_sdf_data = reconstruct(
                decoder,
                int(args.iterations),
                latent_size,
                data_sdf,
                0.01,
                0.1,
                num_samples=specs["SamplesPerScene"],
                lr=5e-3,
                l2reg=True,
                log_wandb=args.use_wandb,
                shape_name=shape_name,
            )
        print(f"Reconstruction error: {err:.6f}")
        # Standard latent code optimization for single output
        err_sum += err
        decoder.eval()

        # Save SDF data for visualization
        for i,sdf_data in enumerate(sample_sdf_data):
            print(f"Saving SDF data for particle {i}")
            sdf_filename = os.path.join(reconstruction_points_dir, f"{npz[:-4]}_{i}.npz")
            if not os.path.isdir(os.path.dirname(sdf_filename)):
                os.makedirs(os.path.dirname(sdf_filename))
            np.savez(sdf_filename, 
                    xyz=sdf_data[:, :3].cpu().numpy(), 
                sdf=sdf_data[:, 3].cpu().numpy())
            print(f"Saved SDF data: {sdf_filename}")
        if args.use_svgd:
            for k in range(num_particles):
                mesh_filename = os.path.join(reconstruction_meshes_dir, f"{npz[:-4]}_svgd{k}")
                latent_filename = os.path.join(reconstruction_codes_dir, f"{npz[:-4]}_svgd{k}.pth")

                if not save_latvec_only:
                    if not os.path.exists(os.path.dirname(mesh_filename)):
                        os.makedirs(os.path.dirname(mesh_filename))
                    with torch.no_grad():
                        mesh.create_mesh(
                            decoder, latents[k], mesh_filename, N=256, max_batch=int(2 ** 18)
                        )
                print(f"Reconstructed {mesh_filename} with error {err:.6f}")

                if not os.path.exists(os.path.dirname(latent_filename)):
                    os.makedirs(os.path.dirname(latent_filename))
                torch.save(latents[k].unsqueeze(0), latent_filename)
        else:
            for k in range(len(latents)):
                print(f"Latent vector {k} norm: {torch.norm(latents[k]).item():.6f}")
                mesh_filename = os.path.join(reconstruction_meshes_dir, f"{npz[:-4]}_{k}")
                latent_filename = os.path.join(reconstruction_codes_dir, f"{npz[:-4]}_{k}.pth")
                
                if not os.path.exists(mesh_filename) or not args.skip or rerun > 0:
                    if not os.path.exists(os.path.dirname(mesh_filename)):
                        os.makedirs(os.path.dirname(mesh_filename))
                    with torch.no_grad():
                        mesh.create_mesh(
                            decoder, latents[k], mesh_filename, N=256, max_batch=int(2 ** 18)
                        )
                    print(f"Reconstructed {mesh_filename} with error {err:.6f}")
                    if not os.path.exists(os.path.dirname(latent_filename)):
                        os.makedirs(os.path.dirname(latent_filename))
                    torch.save(latents[k].unsqueeze(0), latent_filename)
    
    # Finish wandb run
    if args.use_wandb:
        wandb.finish()
        print("Wandb run finished.")
