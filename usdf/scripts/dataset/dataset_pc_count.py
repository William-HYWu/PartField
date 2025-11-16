import argparse

import mmint_utils
from usdf.data.sdf_dataset import SDFDataset
import usdf.config as config


def dataset_pc_count(dataset: SDFDataset):
    pc_sizes = []
    for idx in range(len(dataset)):
        data_dict = dataset[idx]

        pc = data_dict["partial_pointcloud"]
        pc_sizes.append(pc.shape[0])

    print(pc_sizes)
    print("Max: ", max(pc_sizes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_cfg", type=str, help="Path to dataset config file.")
    parser.add_argument("mode", type=str, help="Mode to run in.")
    args = parser.parse_args()

    dataset_cfg = mmint_utils.load_cfg(args.dataset_cfg)
    dataset = config.get_dataset(args.mode, dataset_cfg)

    dataset_pc_count(dataset)
