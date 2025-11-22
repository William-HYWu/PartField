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
from dataset.data import PartFieldDataset

specifications_filename = "specs.json"

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

    decoder = Decoder(latent_size, **specs["NetworkSpecs"]).to(device)

    num_epochs = specs["NumEpochs"]

    feature_dataset = PartFieldDataset(data_root="data")

    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    print("loading data with {} threads".format(num_data_loader_threads))

    feature_loader = data_utils.DataLoader(
        feature_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )

    print("torch num_threads: {}".format(torch.get_num_threads()))

    num_scenes = len(feature_dataset)

    print("There are {} scenes".format(num_scenes))

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
    models_dir = os.path.join("checkpoints", name)
    os.makedirs(models_dir, exist_ok=True)

    start_epoch = 1
    criterion = torch.nn.MSELoss()

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
        epoch_chunk_count = 0
        num_batches = 0

        for imgs, points, features, indices in tqdm(feature_loader, desc="Loading Feature Data", unit="batch"):
            num_points = points.shape[1]
            channels, height, width = imgs.shape[1], imgs.shape[2], imgs.shape[3]
            imgs = imgs.unsqueeze(1).repeat(1, num_points, 1, 1, 1).view(-1, channels, height, width)
            points = points.view(-1, points.shape[2])
            features = features.view(-1, features.shape[2])

            perm = torch.randperm(imgs.shape[0])
            imgs = imgs[perm]
            points = points[perm]
            features = features[perm]

            imgs_chunks = torch.chunk(imgs, batch_split)
            points_chunks = torch.chunk(points, batch_split)
            features_chunks = torch.chunk(features, batch_split)

            optimizer_all.zero_grad()
            batch_loss = 0.0

            for img_chunk, point_chunk, feature_chunk in zip(imgs_chunks, points_chunks, features_chunks):
                pred_features = decoder(img_chunk.to(device), point_chunk.to(device))
                loss = criterion(pred_features, feature_chunk.to(device))
                loss.backward()

                batch_loss += loss.item()
                epoch_loss += loss.item()
                epoch_chunk_count += 1

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=grad_clip)

            optimizer_all.step()
            num_batches += 1

        avg_epoch_loss = epoch_loss / max(1, epoch_chunk_count)
        
        end = time.time()
        epoch_time = end - start
        
        log_dict = {
            "epoch": epoch,
            "loss/total": avg_epoch_loss,
            "metrics/epoch_time": epoch_time,
            "metrics/samples_per_sec": (num_batches * scene_per_batch * num_samp_per_scene) / epoch_time,
        }
        
        wandb.log(log_dict, step=epoch)
        
        print("Epoch {} - Loss: {:.6f} - Time: {:.2f}s".format(
            epoch, avg_epoch_loss, epoch_time))

        if epoch in checkpoints:
            wandb.log({"checkpoint": epoch}, step=epoch)
            
        if epoch % 100 == 0:
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                
                # Save best model
                best_model_path = os.path.join(models_dir, f"best_model_epoch_{epoch}_loss_{avg_epoch_loss:.6f}_{wandb_name}.pth")
                best_model = copy.deepcopy(decoder.state_dict())
                torch.save(best_model, best_model_path)
                print(f"Saved best model at epoch {epoch} with loss {avg_epoch_loss:.6f}")
                wandb.log({
                    "best_model_saved": epoch,
                    "best_loss": best_loss,
                }, step=epoch)

    print("Training complete.")

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
        default="442",
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

