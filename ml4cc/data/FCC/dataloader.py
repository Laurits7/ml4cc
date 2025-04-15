import os
import glob
import math
import torch
import numpy as np
import awkward as ak
from ml4cc.tools.data import io
from omegaconf import DictConfig
# from collections.abc import Sequence
from lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset



class IterableFCCDataset(IterableDataset):
    def __init__(self, data_path: str, cfg: DictConfig, dataset_type: str):
        self.data_path = data_path
        self.cfg = cfg
        self.dataset_type = dataset_type
        self.row_groups = self.load_row_groups()
        self.num_rows = sum([rg.num_rows for rg in self.row_groups])
        self.window_size = self.cfg.datasets.CEPC.slidig_window.size
        self.stride = self.cfg.datasets.CEPC.slidig_window.stride
        self.waveform_len = 1200
        print(f"There are {'{:,}'.format(self.num_rows)} waveforms in the {dataset_type} dataset.")
        print(f"Each waveform will be split into sliding windows of size {self.window_size}")
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

    def __len__(self):
            return self.num_rows * int(self.waveform_len / self.window_size)

    def build_tensors(self, data: ak.Array):
        waveform = ak.Array(data.waveform)
        target = ak.Array(data.target)
        waveforms = []
        targets = []
        waveform_indices = []
        for wf_idx, (wf, t) in enumerate(zip(waveform, target)):
            for start_idx in range(0, len(wf) - self.window_size + 1, self.stride):
                waveforms.append(wf[start_idx: start_idx + self.window_size])
                if 1 in t[start_idx: start_idx + self.window_size]:
                    targets.append(1)
                else:
                    targets.append(0)
                waveform_indices.append(wf_idx)
        waveforms = torch.tensor(waveforms, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        waveform_indices = torch.tensor(waveform_indices, dtype=torch.float32)
        return waveforms.unsqueeze(-1), targets, waveform_indices

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
            waveforms, targets, waveform_indices = self.build_tensors(data)

            # return individual waveform (pieces) from the dataset
            for idx_wf in range(len(waveforms)):
                yield waveforms[idx_wf], targets[idx_wf], waveform_indices[idx_wf]


class FCCDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.test_loader = None
        self.train_loader = None
        self.val_loader = None
        self.test_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.save_hyperparameters()
        super().__init__()

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str):
        if stage == "fit":
            train_dir = os.path.join(self.cfg.datasets.FCC.data_dir, "train")
            val_dir = os.path.join(self.cfg.datasets.FCC.data_dir, "val")
            self.train_dataset = IterableFCCDataset(
                data_path=train_dir,
                cfg=self.cfg,
                dataset_type="train",
            )
            self.val_dataset = IterableFCCDataset(
                data_path=val_dir,
                cfg=self.cfg,
                dataset_type="validation",
            )
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.cfg.training.batch_size,
                num_workers=self.cfg.training.num_dataloader_workers,
                prefetch_factor=self.cfg.training.prefetch_factor,
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.cfg.training.batch_size,
                num_workers=self.cfg.training.num_dataloader_workers,
                prefetch_factor=self.cfg.training.prefetch_factor,
            )

        elif stage == "test":
            test_dir = os.path.join(self.cfg.datasets.FCC.data_dir, "test")
            self.test_dataset = IterableFCCDataset(
                data_path=test_dir,
                cfg=self.cfg,
                dataset_type="test",
            )
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.cfg.training.batch_size)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


class ClusterizationIterableFCCDataset(IterableDataset):
    def __init__(self, data_path: str, cfg: DictConfig, dataset_type: str, pred_dataset: bool = False):
        self.dataset = data_path
        self.cfg = cfg
        self.dataset = self.load_row_groups()
        self.dataset_type = dataset_type
        self.row_groups = [d for d in self.dataset][:2]
        self.pred_dataset = pred_dataset
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


    def build_tensors(self, data: ak.Array):
        if self.pred_dataset:
            pred_peaks = ak.Array(data.detected_peaks)
            # gen_peaks = ak.Array(data.target)
            # peaks = pred_peaks * (gen_peaks == 1)  # If we want to consider only the correctly classified peaks.
            peaks = pred_peaks
        else:
            peaks  = ak.Array(data.target)
        targets = ak.sum(peaks == 1, axis = -1)
        targets = torch.tensor(targets, dtype=torch.float32)
        peaks = torch.tensor(peaks, dtype=torch.float32)
        return peaks, targets

    def __len__(self):
        return self.num_rows

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


class ClusterizationFCCDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.test_loader = None
        self.train_loader = None
        self.val_loader = None
        self.test_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.save_hyperparameters()
        super().__init__()

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str):
        if stage == "fit":
            train_dir = os.path.join(self.cfg.datasets.FCC.data_dir, "train")
            val_dir = os.path.join(self.cfg.datasets.FCC.data_dir, "val")
            self.train_dataset = ClusterizationIterableFCCDataset(
                data_path=train_dir,
                cfg=self.cfg,
                dataset_type="train",
            )
            self.val_dataset = ClusterizationIterableFCCDataset(
                data_path=val_dir,
                cfg=self.cfg,
                dataset_type="validation",
            )
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.cfg.training.batch_size,
                num_workers=self.cfg.training.num_dataloader_workers,
                prefetch_factor=self.cfg.training.prefetch_factor,
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.cfg.training.batch_size,
                num_workers=self.cfg.training.num_dataloader_workers,
                prefetch_factor=self.cfg.training.prefetch_factor,
            )

        elif stage == "test":
            test_dir = os.path.join(self.cfg.datasets.FCC.data_dir, "test")
            self.test_dataset = ClusterizationIterableFCCDataset(
                data_path=test_dir,
                cfg=self.cfg,
                dataset_type="test",
            )
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.cfg.training.batch_size)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader



#####################################################################################
#####################################################################################
########################    One-step cluster counting      ########################
#####################################################################################
#####################################################################################


class OneStepIterableDataSet(IterableDataset):
    def __init__(self, data_paths: list, cfg: DictConfig, dataset_type: str):
        self.data_paths = data_paths
        self.cfg = cfg
        self.dataset_type = dataset_type
        self.row_groups = self.load_row_groups()[:2] # TODO: temporary for prototyping
        self.num_rows = self.num_rows = sum([rg.num_rows for rg in self.row_groups])
        print(f"There are {'{:,}'.format(self.num_rows)} waveforms in the {dataset_type} dataset.")
        super().__init__()

    def load_row_groups(self) -> list:
        all_row_groups = []
        input_files = []
        if isinstance(self.data_paths, list):
            input_files = self.data_paths
        elif isinstance(self.data_paths, str):
            if os.path.isdir(self.data_paths):
                self.data_paths = os.path.expandvars(self.data_paths)
                input_files = glob.glob(os.path.join(self.data_paths, "*.parquet"))
            elif "*" in self.data_paths:
                input_files = glob.glob(self.data_paths)
            elif os.path.isfile(self.data_paths):
                input_files = [self.data_paths]
            else:
                raise ValueError(f"Unexpected data_path: {self.data_paths}")
        for data_path in input_files:
            metadata = ak.metadata_from_parquet(data_path)
            num_row_groups = metadata["num_row_groups"]
            col_counts = metadata["col_counts"]
            all_row_groups.extend(
                [io.RowGroup(data_path, row_group, col_counts[row_group]) for row_group in range(num_row_groups)]
            )
        return all_row_groups

    def build_tensors(self, data: ak.Array):
        targets = np.array(data.target == 1, dtype=int)
        targets = np.sum(targets, axis=-1)
        targets = torch.tensor(targets, dtype=torch.float32)
        waveform = torch.tensor(ak.Array(data.waveform), dtype=torch.float32)
        return waveform, targets

    def __len__(self):
        return self.num_rows

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


class OneStepFCCDataModule(LightningDataModule):
    # Should merge with the peakFinding DataModule
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.test_loader = None
        self.train_loader = None
        self.val_loader = None
        self.test_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.save_hyperparameters()
        super().__init__()

    def get_dataset_paths(self, dataset_type: str) -> list:
        wcp_path = os.path.join(self.cfg.datasets.FCC.data_dir, dataset_type, "*.parquet")
        return list(glob.glob(wcp_path))

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str):
        if stage == "fit":
            train_paths = self.get_dataset_paths(dataset_type="train")
            val_paths = self.get_dataset_paths(dataset_type="val")
            self.train_dataset = OneStepIterableDataSet(
                data_paths=train_paths,
                cfg=self.cfg,
                dataset_type="train",
            )
            self.val_dataset = OneStepIterableDataSet(
                data_paths=val_paths,
                cfg=self.cfg,
                dataset_type="validation",
            )
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=128,
                num_workers=self.cfg.training.num_dataloader_workers,
                prefetch_factor=self.cfg.training.prefetch_factor,
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=128,
                num_workers=self.cfg.training.num_dataloader_workers,
                prefetch_factor=self.cfg.training.prefetch_factor,
            )

        elif stage == "test":
            test_paths = self.get_dataset_paths(dataset_type="test")
            self.test_dataset = OneStepIterableDataSet(
                data_paths=test_paths,
                cfg=self.cfg,
                dataset_type="test",
            )
            self.test_loader = DataLoader(self.test_dataset, batch_size=128)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader