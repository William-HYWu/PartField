import argparse
import os.path

import mmint_utils
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split dataset into train/test.")
    parser.add_argument("meshes_dir", type=str, help="Meshes directory.")
    parser.add_argument("dataset_dir", type=str, help="Dataset directory.")
    args = parser.parse_args()

    meshes_dir = args.meshes_dir
    dataset_dir = args.dataset_dir
    mmint_utils.make_dir(dataset_dir)
    splits_dir = os.path.join(meshes_dir, "splits")
    mmint_utils.make_dir(splits_dir)

    mesh_fns = [f for f in os.listdir(meshes_dir) if ".obj" in f]
    np.random.shuffle(mesh_fns)

    num_train = int(0.9 * len(mesh_fns))
    train_fns = mesh_fns[:num_train]
    test_fns = mesh_fns[num_train:]

    with open(os.path.join(splits_dir, "train.txt"), "w") as f:
        for fn in train_fns:
            f.write(fn + "\n")

    with open(os.path.join(splits_dir, "test.txt"), "w") as f:
        for fn in test_fns:
            f.write(fn + "\n")
