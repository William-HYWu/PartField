import argparse

from tqdm import trange

import mmint_utils
from usdf import config


def visualize_dataset(model_config: dict, split: str):
    train_dataset = config.get_dataset(split, model_config)

    assert hasattr(train_dataset, 'visualize_item'), "Dataset does not have a visualize_item method."

    for idx in trange(len(train_dataset)):
        data_dict = train_dataset[idx]
        train_dataset.visualize_item(data_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize dataset.')
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('split', type=str, default='train', help='Dataset split to visualize.')
    args = parser.parse_args()

    model_config_ = mmint_utils.load_cfg(args.config)
    visualize_dataset(model_config_, args.split)
