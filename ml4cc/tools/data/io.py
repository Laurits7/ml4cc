import uproot
import random
import numpy as np
import awkward as ak
from torch.utils.data import ConcatDataset, Subset


def load_root_file(path: str, tree_path: str = "sim", branches: list = None) -> ak.Array:
    """ Loads the CEPC dataset .root file.

    Parameters:
        path : str
            Path to the .root file
        tree_path : str
            Path in the tree in the .root file.
        branches : list
            [default: None] Branches to be loaded from the .root file. By default, all branches will be loaded.

    Returns:
        array : ak.Array
            Awkward array containing the .root file data
    """
    with uproot.open(path) as in_file:
        tree = in_file[tree_path]
        arrays = tree.arrays(branches)
    return arrays


def save_array_to_file(data: ak.Array, output_path: str) -> None:
    print(f"Saving {len(data)} processed entries to {output_path}")
    ak.to_parquet(data, output_path, row_group_size=1024)


class RowGroup:
    def __init__(self, filename, row_group, num_rows):
        self.filename = filename
        self.row_group = row_group
        self.num_rows = num_rows



def train_val_split_shuffle(
    concat_dataset: ConcatDataset,
    val_split: float = 0.2,
    seed: int = 42,
    max_waveforms_for_training: int = -1,
    row_group_size: int = 1024,
):
    total_len = len(concat_dataset)
    indices = list(range(total_len))
    random.seed(seed)
    random.shuffle(indices)

    split = int(total_len * val_split)
    if max_waveforms_for_training == -1:
        train_end_idx = None
    else:
        num_train_rows = int(np.ceil(max_waveforms_for_training / row_group_size))
        train_end_idx = split + num_train_rows
    val_indices = indices[:split]
    train_indices = indices[split:train_end_idx]

    train_subset = Subset(concat_dataset, train_indices)
    val_subset = Subset(concat_dataset, val_indices)

    return train_subset, val_subset
