import torch
import torch.utils.data as data_utils
import os
import json
import numpy as np
from tqdm import tqdm
import argparse

from models.decoder import Decoder
from dataset.data import PartFieldDataset

def load_experiment_specifications(experiment_directory):
    filename = os.path.join(experiment_directory, "specs.json")
    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include specifications file "
            + '"specs.json"'.format(experiment_directory)
        )
    return json.load(open(filename))

def main(experiment_directory, checkpoint, data_root="data", gpu_id=0):
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load specs
    specs = load_experiment_specifications(experiment_directory)
    latent_size = specs["CodeLength"]
    
    # Initialize model
    decoder = Decoder(latent_size, **specs["NetworkSpecs"]).to(device)
    
    # Load checkpoint
    if os.path.isfile(checkpoint):
        checkpoint_path = checkpoint
    else:
        raise ValueError(f"Checkpoint file not found: {checkpoint}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint_dict = torch.load(checkpoint_path, map_location=device)
    decoder.load_state_dict(checkpoint_dict)
    decoder.eval()

    # Initialize Dataset
    dataset = PartFieldDataset(data_root=data_root)
    dataloader = data_utils.DataLoader(
        dataset,
        batch_size=1, # Process one by one
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    # Output directory
    epoch_name = os.path.basename(checkpoint_path).split('.')[0]
    output_dir = os.path.join(experiment_directory, "predictions", epoch_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving predictions to: {output_dir}")

    with torch.no_grad():
        for img, points, features, obj_id in tqdm(dataloader, desc="Saving Predictions"):
            img = img.to(device)
            points = points.to(device) # (1, N, 3)
            
            # 1. Get latent
            latent = decoder.get_latent(img) # (1, latent_size)
            
            # 2. Expand latent
            num_points = points.shape[1]
            latent_expanded = latent.unsqueeze(1).expand(-1, num_points, -1) # (1, N, latent_size)
            
            # 3. Concatenate
            model_input = torch.cat([latent_expanded, points], dim=2) # (1, N, latent_size + 3)
            
            # 4. Inference (Chunking if necessary)
            # Flatten for inference
            model_input_flat = model_input.view(-1, model_input.shape[-1]) # (N, latent_size + 3)
            
            pred_feats_list = []
            chunk_size = 10000
            for i in range(0, model_input_flat.shape[0], chunk_size):
                chunk = model_input_flat[i:i+chunk_size]
                pred_chunk = decoder.inference(chunk)
                pred_feats_list.append(pred_chunk.cpu().numpy())
            
            pred_feats = np.concatenate(pred_feats_list, axis=0)
            
            # Concatenate points and predicted features
            points_np = points.squeeze(0).cpu().numpy()
            output_data = np.concatenate([points_np, pred_feats], axis=1)

            # 5. Save
            save_path = os.path.join(output_dir, f"{obj_id[0]}.npy")
            np.save(save_path, output_data)

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save predictions from a trained model")
    parser.add_argument("--experiment", "-e", required=True, help="Experiment directory")
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to the checkpoint .pth file")
    parser.add_argument("--data_root", "-d", default="data", help="Root directory of data")
    parser.add_argument("--gpu", "-g", type=int, default=0, help="GPU ID")
    
    args = parser.parse_args()
    
    main(args.experiment, args.checkpoint, args.data_root, args.gpu)