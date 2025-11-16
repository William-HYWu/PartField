import os
import yaml
import argparse
import random

def generate_dataset_config(data_dir, output_path, train_ratio=0.8, val_ratio=0.1):
    """
    Scans a directory for .npz files and generates a YAML dataset configuration file.
    """
    
    # Find all NPZ files recursively
    npz_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".npz"):
                # Store the path including the data directory
                npz_files.append(os.path.join(root, file))

    if not npz_files:
        print(f"Warning: No .npz files found in {data_dir}")
        return

    # Create the file list for the YAML
    file_list = [{"path": path} for path in npz_files]

    # Create the YAML structure
    config = {
        "dataset": {
            "name": "mugs_dataset",
            "splits": {
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": 1.0 - train_ratio - val_ratio,
            }
        },
        "files": file_list
    }

    # Write the YAML file
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Successfully generated {output_path} with {len(npz_files)} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dataset config YAML file from a directory of NPZ files.")
    parser.add_argument(
        "data_dir", 
        type=str, 
        help="Directory containing the NPZ files (e.g., 'mugs_dataset/')."
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="configs/dataset/mugs_dataset_config1.yaml",
        help="Path to save the output YAML file."
    )
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Proportion of data for training.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Proportion of data for validation.")
    
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    generate_dataset_config(args.data_dir, args.output_path, args.train_ratio, args.val_ratio)