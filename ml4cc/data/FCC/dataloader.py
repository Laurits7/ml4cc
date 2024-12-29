import os
import glob
import math
import torch
import awkward as ak
from ml4cc.tools.data import io
from omegaconf import DictConfig
from collections.abc import Sequence
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, IterableDataset, ConcatDataset


class FCCIterableDataset(Dataset):
    def __init__(self, data_path: str, cfg: DictConfig, dataset_type: str, pred_dataset: bool = False):
        self.data_path = data_path
        self.cfg = cfg
        self.dataset_type = dataset_type
        self.pred_dataset = pred_dataset
        self.row_groups = self.load_row_groups()
        self.num_rows = self.num_rows = sum([rg.num_rows for rg in self.row_groups])
        print(f"There are {'{:,}'.format(self.num_rows)} waveforms in the {dataset_type} dataset.")
        super().__init__()

    def load_row_groups(self) -> list:
        all_row_groups = []
        input_files = []
        if isinstance(self.data_path, list):
            input_files = self.data_path
        elif isinstance(self.data_path, str):
            if os.path.isdir(self.data_path):
                self.data_path = os.path.expandvars(self.data_path)
                input_files = glob.glob(os.path.join(self.data_path, "*.parquet"))
            elif "*" in self.data_path:
                input_files = glob.glob(self.data_path)
            elif os.path.isfile(self.data_path):
                input_files = [self.data_path]
            else:
                raise ValueError(f"Unexpected data_path: {self.data_path}")
        for data_path in input_files:
            metadata = ak.metadata_from_parquet(data_path)
            num_row_groups = metadata["num_row_groups"]
            col_counts = metadata["col_counts"]
            all_row_groups.extend(
                [io.RowGroup(data_path, row_group, col_counts[row_group]) for row_group in range(num_row_groups)]
            )
        return all_row_groups
