import os
import math
import torch
import random
import numpy as np
import awkward as ak
from omegaconf import DictConfig
from collections.abc import Sequence
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, IterableDataset, ConcatDataset, Subset


class RowGroup:
    def __init__(self, filename, row_group, num_rows):
        self.filename = filename
        self.row_group = row_group
        self.num_rows = num_rows


class CEPCDataset(Dataset):
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.row_groups = self.load_row_groups()

    def load_row_groups(self) -> Sequence[RowGroup]:
        metadata = ak.metadata_from_parquet(self.data_path)
        num_row_groups = metadata["num_row_groups"]
        col_counts = metadata["col_counts"]
        return [RowGroup(self.data_path, row_group, col_counts[row_group]) for row_group in range(num_row_groups)]

    def __getitem__(self, index):
        return self.row_groups[index]

    def __len__(self):
        return len(self.row_groups)


class IterableCEPCDataset(IterableDataset):
    def __init__(self, dataset: Dataset, dataset_type: str):
        self.dataset = dataset
        self.dataset_type = dataset_type
        self.row_groups = [d for d in self.dataset]
        self.num_rows = sum([rg.num_rows for rg in self.row_groups])
        print(f"There are {'{:,}'.format(self.num_rows)} waveforms in the {dataset_type} dataset.")

    def build_tensors(self, data: ak.Array):
        waveform = torch.tensor(data.waveform)
        target = torch.tensor(data.target)
        return waveform, target

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            row_groups_to_process = self.row_groups
        else:
            per_worker = int(math.ceil(float(len(self.row_groups)) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            row_groups_start = worker_id * per_worker
            row_groups_end = row_groups_start + per_worker
            row_groups_to_process = self.row_groups[row_groups_start:row_groups_end]

        for row_group in row_groups_to_process:
            # load one chunk from one file
            data = ak.from_parquet(row_group.filename, row_groups=[row_group.row_group])
            tensors = self.build_tensors(data)

            # return individual jets from the dataset
            for idx_wf in range(len(data)):
                yield tensors[0][idx_wf], tensors[1][idx_wf]


class CEPCDataModule(LightningDataModule):
    def __init__(
            self,
            cfg: DictConfig,
            training_task: str,
            samples: list
    ):
        """ The base class for CEPC dataset.

        Parameters:
            cfg : DictConfig
                The configuration for the
            training_task : str
                The task for which data is loaded. Options: ['peakFinding' ; 'clusterization']
            samples : str
                Which samples to load. Options: ['kaon', 'pion', 'all']
        """
        self.cfg = cfg
        self.training_task = training_task
        self.samples = samples
        self.test_loader = None
        self.train_loader = None
        self.val_loader = None
        self.test_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        super().__init__()

    def get_dataset_path(self, sample: str, dataset_type: str) -> str:
        return os.path.join(self.cfg.data_dir,
                            f"{sample}_{dataset_type}.parquet")  # TODO: Check where data_dir is in cfg

    def setup(self, stage: str):
        if stage == 'fit':
            train_datasets = []
            for sample in self.samples:
                data_path = self.get_dataset_path(sample=sample, dataset_type="train")
                full_train_dataset = CEPCDataset(data_path=data_path)
                train_datasets.append(full_train_dataset)
            train_concat_dataset = ConcatDataset(train_datasets)
            train_subset, val_subset = train_val_split_shuffle(
                concat_dataset=train_concat_dataset,
                val_split=self.hparams.fraction_valid,
                max_waveforms_for_training=-1,
                row_group_size=self.hparams.row_group_size
            )
            self.train_dataset = IterableCEPCDataset(
                dataset=train_subset,
                dataset_type="train",
            )
            self.val_dataset = IterableCEPCDataset(
                dataset=val_subset,
                dataset_type="validation",
            )
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.num_dataloader_workers,
                prefetch_factor=self.cfg.prefetch_factor,
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.num_dataloader_workers,
                prefetch_factor=self.cfg.prefetch_factor,
            )

        elif stage == "test":
            test_datasets = []
            for sample in self.samples:
                data_path = self.get_dataset_path(sample=sample, dataset_type="test")
                test_dataset = CEPCDataset(data_path=data_path)
                test_datasets.append(test_dataset)
            test_concat_dataset = ConcatDataset(test_datasets)
            self.test_dataset = IterableCEPCDataset(
                dataset=test_concat_dataset,
                dataset_type="test",
            )
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.cfg.batch_size
            )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


def train_val_split_shuffle(
        concat_dataset: ConcatDataset,
        val_split: float = 0.2,
        seed: int = 42,
        max_waveforms_for_training: int = -1,
        row_group_size: int = 1024
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
