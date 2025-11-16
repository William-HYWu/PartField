#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import math
import json
import time
import random
import numpy as np
from tqdm import tqdm
import copy
import wandb

from models.decoder import Decoder
from dataset.data import SDFSamples

specifications_filename = "specs.json"

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

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


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):

        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(specs):

    schedule_specs = specs["LearningRateSchedule"]

    schedules = []

    for schedule_specs in schedule_specs:

        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )
        elif schedule_specs["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Final"],
                    schedule_specs["Length"],
                )
            )
        elif schedule_specs["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules

def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def get_mean_latent_vector_magnitude(latent_vectors):
    # Accepts: nn.Embedding / nn.Parameter / plain Tensor
    if hasattr(latent_vectors, "weight"):  # Embedding/module
        data = latent_vectors.weight.data.detach()
    else:
        data = latent_vectors.detach()
    return torch.mean(torch.norm(data, dim=1))
        
def load_experiment_specifications(experiment_directory):

    filename = os.path.join(experiment_directory, specifications_filename)

    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include specifications file "
            + '"specs.json"'.format(experiment_directory)
        )

    return json.load(open(filename))


def main_function(experiment_directory, continue_from, batch_split, seed=42, wandb_project="usdf-deepsdf", wandb_name=None, name="mugs", gpu_id=0,random_se2_num=0):

    # Set random seeds for reproducibility
    set_random_seeds(seed)
    
    # Set GPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU: {gpu_id} ({torch.cuda.get_device_name(gpu_id)})")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")

    # Initialize wandb
    if wandb_name is None:
        wandb_name = f"exp_{os.path.basename(experiment_directory)}_seed{seed}_gpu{gpu_id}"
    
    wandb.init(
        project=wandb_project,
        name=wandb_name,
        config={
            "experiment_directory": experiment_directory,
            "batch_split": batch_split,
            "seed": seed,
            "gpu_id": gpu_id,
        }
    )

    specs = load_experiment_specifications(experiment_directory)

    # Log experiment specs to wandb
    wandb.config.update({
        "latent_size": specs["CodeLength"],
        "num_epochs": specs["NumEpochs"],
        "samples_per_scene": specs["SamplesPerScene"],
        "scenes_per_batch": specs["ScenesPerBatch"],
        "clamping_distance": specs["ClampingDistance"],
        "snapshot_frequency": specs["SnapshotFrequency"],
        "network_specs": specs["NetworkSpecs"],
        "learning_rate_schedule": specs["LearningRateSchedule"],
        "code_regularization": get_spec_with_default(specs, "CodeRegularization", True),
        "code_reg_lambda": get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4),
    })

    latent_size = specs["CodeLength"]

    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )

    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    lr_schedules = get_learning_rate_schedules(specs)

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)

    def signal_handler(sig, frame):
        print("Stopping early...")
        sys.exit(0)

    def adjust_learning_rate(lr_schedules, optimizer, epoch):

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    signal.signal(signal.SIGINT, signal_handler)

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = True

    do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)

    code_bound = get_spec_with_default(specs, "CodeBound", None)

    decoder = Decoder(latent_size, **specs["NetworkSpecs"]).to(device)

    num_epochs = specs["NumEpochs"]

    sdf_dataset = SDFSamples(
            split = "train", subsample=num_samp_per_scene, load_ram=False, random_se2_num=random_se2_num, config_path="configs/dataset/mugs_dataset_config_oriented_single.yaml"
    )

    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    print("loading data with {} threads".format(num_data_loader_threads))

    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )

    print("torch num_threads: {}".format(torch.get_num_threads()))

    num_scenes = len(sdf_dataset)

    print("There are {} scenes".format(num_scenes))

    #lat_vecs = torch.nn.Embedding(num_scenes, latent_size, max_norm=code_bound)
    #torch.nn.init.normal_(
    #    lat_vecs.weight.data,
    #    0.0,
    #    get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
    #)

    loss_l1 = torch.nn.L1Loss(reduction="sum")

    optimizer_all = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            }
        ]
    )

    # Best model tracking
    best_loss = float('inf')
    best_epoch = 0
    models_dir = os.path.join("checkpoints", name)
    os.makedirs(models_dir, exist_ok=True)

    start_epoch = 1

    for epoch in tqdm(range(start_epoch, num_epochs + 1), desc="Overall Training Progress", unit="epoch"):

        start = time.time()

        decoder.train()

        adjust_learning_rate(lr_schedules, optimizer_all, epoch)
        
        current_lrs = [param_group["lr"] for param_group in optimizer_all.param_groups]
        wandb.log({
            "epoch": epoch,
            "lr_decoder": current_lrs[0]
        }, step=epoch)
        
        epoch_loss = 0.0
        epoch_sdf_loss = 0.0
        epoch_reg_loss = 0.0
        epoch_gradient_penalty = 0.0
        num_batches = 0
        se2_vecs_all = sdf_dataset.se2_vector

        for sdf_data, indices in tqdm(sdf_loader, desc="Loading SDF Data", unit="batch"):

            # Process the input data
            sdf_data = sdf_data.reshape(-1, 8)

            num_sdf_samples = sdf_data.shape[0]
            sdf_data.requires_grad = False
            
            perm = torch.randperm(num_sdf_samples)
            sdf_data = sdf_data[perm]

            xyz = sdf_data[:, 0:3]
            sdf_gt = sdf_data[:, 3].unsqueeze(1)
            se2_vec = sdf_data[:, 4:8]

            if enforce_minmax:
                sdf_gt = torch.clamp(sdf_gt, minT, maxT)

            xyz = torch.chunk(xyz, batch_split)
            indices = torch.chunk(
                indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1)[perm],
                batch_split,
            )

            sdf_gt = torch.chunk(sdf_gt, batch_split)
            se2_vec = torch.chunk(se2_vec, batch_split)

            batch_loss = 0.0
            batch_sdf_loss = 0.0
            batch_reg_loss = 0.0

            optimizer_all.zero_grad()

            for i in range(batch_split):
                optimizer_all.zero_grad()
                #batch_vecs = lat_vecs(indices[i])
                num_sdf_batch = sdf_gt[i].shape[0]

                #input = torch.cat([batch_vecs, xyz[i]], dim=1).to(device)

                # NN optimization
                pred_sdf, latent_vec, points = decoder(se2_vec[i].to(device), xyz[i].to(device))

                gradient_sdf = gradient(pred_sdf, points)
                grad_loss = ((gradient_sdf.norm(2, dim=-1) - 1) ** 2).mean()

                if enforce_minmax:
                    pred_sdf = torch.clamp(pred_sdf, minT, maxT)

                sdf_loss = loss_l1(pred_sdf, sdf_gt[i].to(device)) / num_sdf_batch
                chunk_loss = sdf_loss

                # KL divergence regularization: encourage latent_vec to match N(0, I)
                # if do_code_regularization:
                #     # latent_vec: shape (num_sdf_batch, latent_size) or similar
                #     # Compute empirical mean and variance across the batch (per-dimension)
                #     mu = torch.mean(latent_vec, dim=0)
                #     var = torch.var(latent_vec, dim=0, unbiased=False)
                #     # numerical stability
                #     var_eps = var + 1e-8
                #     kl = 0.5 * torch.sum(var_eps + mu.pow(2) - 1.0 - torch.log(var_eps))
                #     reg_loss = (0.01 * min(1, epoch / 100) * kl) / num_sdf_batch

                #     chunk_loss = chunk_loss + reg_loss.to(device)
                #     batch_reg_loss += reg_loss.item()
                #     epoch_reg_loss += reg_loss.item()

                if do_code_regularization:
                    #batch_vecs shape (num_sdf_batch, latent_size)
                    l2_size_loss = torch.sum(torch.norm(latent_vec, dim=1).pow(2))
                    reg_loss = (
                        code_reg_lambda * min(1, epoch / 100) * l2_size_loss
                    ) / num_sdf_batch

                    chunk_loss = chunk_loss + reg_loss.to(device)
                    batch_reg_loss += reg_loss.item()
                    epoch_reg_loss += reg_loss.item()
                #if is sine net, add gradient penalty
                chunk_loss = chunk_loss + 0.1*grad_loss

                chunk_loss.backward()

                batch_loss += chunk_loss.item()
                batch_sdf_loss += sdf_loss.item()
                epoch_loss += chunk_loss.item()
                epoch_sdf_loss += sdf_loss.item()
                epoch_gradient_penalty += grad_loss.item()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
                optimizer_all.step()  

            #optimizer_all.step()
            num_batches += 1

        avg_epoch_loss = epoch_loss / (num_batches*batch_split)
        avg_sdf_loss = epoch_sdf_loss / (num_batches*batch_split)
        avg_reg_loss = epoch_reg_loss / (num_batches*batch_split) if do_code_regularization else 0.0
        avg_gradient_penalty = epoch_gradient_penalty / (num_batches*batch_split)
        
        with torch.no_grad():
            decoder.eval()
            se2_vecs_all = torch.tensor(se2_vecs_all, dtype=torch.float32)
            latent_vecs = decoder.get_latent(se2_vecs_all.to(device)).detach().cpu()

        latent_weights = latent_vecs
        if len(latent_vecs.shape) == 1:
            latent_weights = latent_vecs.unsqueeze(1)
        latent_magnitude = torch.mean(torch.norm(latent_weights, dim=1))
        latent_mean = torch.mean(latent_weights)
        latent_std = torch.std(latent_weights)
        
        end = time.time()
        epoch_time = end - start
        
        log_dict = {
            "epoch": epoch,
            "loss/total": avg_epoch_loss,
            "loss/sdf": avg_sdf_loss,
            "loss/regularization": avg_reg_loss,
            "loss/gradient_penalty": avg_gradient_penalty,
            "metrics/latent_magnitude": latent_magnitude.item(),
            "metrics/latent_mean": latent_mean.item(),
            "metrics/latent_std": latent_std.item(),
            "metrics/epoch_time": epoch_time,
            "metrics/samples_per_sec": (num_batches * scene_per_batch * num_samp_per_scene) / epoch_time,
        }
        
        decoder_grad_norm = 0.0
        latent_grad_norm = 0.0
        for param in decoder.parameters():
            if param.grad is not None:
                decoder_grad_norm += param.grad.data.norm(2).item() ** 2
        #for param in latent_vecs.parameters():
        #    if param.grad is not None:
        #        latent_grad_norm += param.grad.data.norm(2).item() ** 2
        # latent grad norm: only if latent_vecs is a tensor that requires grad and has grad
        latent_grad_norm_sq = 0.0
        if isinstance(latent_vecs, torch.Tensor):
            if getattr(latent_vecs, "requires_grad", False) and (latent_vecs.grad is not None):
                latent_grad_norm_sq = latent_vecs.grad.detach().norm(2).item() ** 2
        # if latent_vecs is a module (older code), fall back to summing its params:
        elif hasattr(latent_vecs, "parameters"):
            for p in latent_vecs.parameters():
                if p.grad is not None:
                    latent_grad_norm_sq += p.grad.data.norm(2).item() ** 2
        
        log_dict.update({
            "gradients/decoder_norm": decoder_grad_norm ** 0.5,
            "gradients/latent_norm": latent_grad_norm ** 0.5,
        })
        
        wandb.log(log_dict, step=epoch)
        
        print("Epoch {} - Loss: {:.6f} (SDF: {:.6f}, Reg: {:.6f}) - Time: {:.2f}s".format(
            epoch, avg_epoch_loss, avg_sdf_loss, avg_reg_loss, epoch_time))

        if epoch in checkpoints:
            wandb.log({"checkpoint": epoch}, step=epoch)
            
            # Save checkpoint latent vectors
            checkpoint_latent_path = os.path.join(models_dir, f"latent_vectors_epoch_{epoch}.pth")
            torch.save(latent_vecs, checkpoint_latent_path)
            print(f"Saved checkpoint latent vectors at epoch {epoch}")

        # Save latent vectors at regular intervals (every 500 epochs)
        if epoch % 500 == 0:
            regular_latent_path = os.path.join(models_dir, f"latent_vectors_epoch_{epoch}_loss_{avg_epoch_loss:.6f}.pth")
            torch.save(latent_vecs, regular_latent_path)
            print(f"Saved regular checkpoint latent vectors at epoch {epoch}")

        if epoch % 100 == 0:
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_epoch = epoch
                
                # Save best model
                best_model_path = os.path.join(models_dir, f"best_model_epoch_{epoch}_loss_{avg_epoch_loss:.6f}_{wandb_name}.pth")
                best_model = copy.deepcopy(decoder.state_dict())
                torch.save(best_model, best_model_path)
                
                # Save corresponding latent vectors
                latent_vectors_path = os.path.join(models_dir, f"best_latent_vectors_epoch_{epoch}_loss_{avg_epoch_loss:.6f}_{wandb_name}.pth")
                best_latent_vectors = copy.deepcopy(latent_vecs)
                torch.save(best_latent_vectors, latent_vectors_path)
                
                print(f"Saved best model at epoch {epoch} with loss {avg_epoch_loss:.6f}")
                print(f"Saved corresponding latent vectors: {latent_vectors_path}")
                
                # Log to wandb
                wandb.log({
                    "best_model_saved": epoch,
                    "best_loss": best_loss,
                    "latent_vectors_saved": epoch,
                }, step=epoch)
                
                # Remove previous best model to save space (optional)
                if best_epoch != epoch:
                    prev_best_path = os.path.join(models_dir, f"best_model_epoch_{best_epoch}.pth")
                    if os.path.exists(prev_best_path):
                        os.remove(prev_best_path)
                        print(f" Removed previous best model from epoch {best_epoch}")

    # Save final latent vectors
    final_latent_path = os.path.join(models_dir, f"final_latent_vectors_epoch_{epoch}_loss_{avg_epoch_loss:.6f}.pth")
    torch.save(latent_vecs, final_latent_path)
    print(f"Saved final latent vectors: {final_latent_path}")
    
    # Also save a mapping from scene indices to scene names for easier reconstruction
    scene_mapping_path = os.path.join(models_dir, "scene_index_mapping.json")
    if hasattr(sdf_dataset, 'npz_filenames'):
        scene_mapping = {i: os.path.basename(filename) for i, filename in enumerate(sdf_dataset.npz_filenames)}
        with open(scene_mapping_path, 'w') as f:
            json.dump(scene_mapping, f, indent=2)
        print(f"Saved scene index mapping: {scene_mapping_path}")

    wandb.finish()


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--name",
        "-n",
        dest="name",
        required=False,
        default="mugs",
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--batch_split",
        dest="batch_split",
        default=1,
        help="This splits the batch into separate subbatches which are "
        + "processed separately, with gradients accumulated across all "
        + "subbatches. This allows for training with large effective batch "
        + "sizes in memory constrained environments.",
    )
    arg_parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    arg_parser.add_argument(
        "--wandb_project",
        dest="wandb_project",
        type=str,
        default="usdf-deepsdf",
        help="Wandb project name (default: usdf-deepsdf).",
    )
    arg_parser.add_argument(
        "--wandb_name",
        dest="wandb_name",
        type=str,
        default=None,
        help="Wandb run name (default: auto-generated).",
    )
    arg_parser.add_argument(
        "--gpu",
        dest="gpu_id",
        type=int,
        default=0,
        help="GPU device ID to use (default: 0).",
    )
    arg_parser.add_argument(
        "--random_se2_num",
        dest="random_se2_num",
        type=int,
        default=0,
        help="Number of random SE(2) transformations to apply (default: 0).",
    )

    args = arg_parser.parse_args()

    main_function(
        args.experiment_directory, 
        args.continue_from, 
        int(args.batch_split), 
        args.seed,
        args.wandb_project,
        args.wandb_name,
        args.name,
        args.gpu_id,
        args.random_se2_num
    )

