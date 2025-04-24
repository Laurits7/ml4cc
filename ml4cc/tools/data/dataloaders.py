import os
import glob
import math
import torch
import numpy as np
import awkward as ak
from ml4cc.tools.data import io
# from ml4cc.tools import general as g
from omegaconf import DictConfig
from lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data import Dataset, DataLoader, IterableDataset, ConcatDataset


#####################################################################################
#####################################################################################
######################             Base classes            ##########################
#####################################################################################
#####################################################################################


class RowGroupDataset(Dataset):
    def __init__(self, data_loc: str):
        self.data_loc = data_loc
        self.input_paths = io.get_all_paths(data_loc)
        self.row_groups =  io.get_row_groups(self.input_paths)

    def __getitem__(self, index):
        return self.row_groups[index]

    def __len__(self):
        return len(self.row_groups)


class BaseIterableDataset(IterableDataset):
    """ Base iterable dataset class to be used for different types of trainings."""
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.row_groups = [row_group for row_group in self.dataset]
        self.num_rows = sum([rg.num_rows for rg in self.row_groups])
        print(f"There are {'{:,}'.format(self.num_rows)} waveforms in the dataset.")
        super().__init__()

    def build_tensors(self, data: ak.Array):
        """ Builds the input and target tensors from the data.
        
        Parameters:
            data : ak.Array
                The data used to build the tensors. The data is a chunk of the dataset loaded from a .parquet file.
        Returns:
            features : torch.Tensor
                The input features of the data
            targets : torch.Tensor
                The target values of the data
        """
        raise NotImplementedError("Please implement the build_tensors method in your subclass")

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
                yield tensors[0][idx_wf], tensors[1][idx_wf] # features, targets


class BaseDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig, iter_dataset: IterableDataset, data_type: str):
        """ Base data module class to be used for different types of trainings.
        Parameters:
            cfg : DictConfig
                The configuration file used to set up the data module.
            iter_dataset : IterableDataset
                The iterable dataset to be used for training and validation. Need to define a separate class for each training type, e.g. one_step, two_step_peak_finding, two_step_clusterization, two_step_minimal etc.
            data_type : str
                The type of the data. In case of CEPC it can be "kaon" or "pion". In case of FCC it is the different energies.
        """
        self.cfg = cfg
        self.task = self.cfg.training.type
        self.data_type = data_type
        self.iter_dataset = iter_dataset
        self.train_loader = None
        self.val_loader = None
        self.train_dataset = None
        self.val_dataset = None
        self.save_hyperparameters()
        super().__init__()

    def get_CEPC_dataset_path(self, dataset_type: str) -> str:
        """Returns the directory of the dataset files for CEPC.
        Parameters:
            dataset_type : str
                The type of the dataset. Can be "train" or "test"

        Returns:
            Case: dataset_type == train
                train_loc : str
                    The directory of the train dataset files
                val_loc : str
                    The directory of the val dataset files
            Case: dataset_type == test
                test_dir : str
                    The directory of the test dataset files
        """
        if dataset_type == "train":
            train_loc = os.path.join(self.cfg.dataset.data_dir, self.task, "train")
            val_loc = os.path.join(self.cfg.dataset.data_dir, self.task, "val")
            return train_loc, val_loc
        elif dataset_type == "test":
            if self.cfg.dataset.train_dataset == "combined":
                test_dir = os.path.join(self.cfg.dataset.data_dir, self.task, "test")
            elif self.cfg.dataset.train_dataset == "separate":
                test_dir = os.path.join(self.cfg.dataset.data_dir, self.task, "test", f"*{self.data_type}*.parquet")
            return test_dir
        else:
            raise ValueError(f"Unexpected train dataset type: {self.cfg.dataset.train_dataset}. Please use 'combined' or 'separate'.")

    def get_FCC_dataset_path(self, dataset_type: str) -> str:
        """Returns the directory of the dataset files for FCC
        Parameters:
            dataset_type : str
                The type of the dataset. Can be "train" or "test"

        Returns:
            Case: dataset_type == train
                train_loc : str
                    The directory of the train dataset files
                val_loc : str
                    The directory of the val dataset files
            Case: dataset_type == test
                test_dir : str
                    The directory of the test dataset files
        """
        if dataset_type == "train":
            if self.cfg.dataset.train_dataset == "combined":
                train_loc = os.path.join(self.cfg.dataset.data_dir, self.task, "train")
                val_loc = os.path.join(self.cfg.dataset.data_dir, self.task, "val")
            elif self.cfg.dataset.train_dataset == "separate":
                train_loc = os.path.join(self.cfg.dataset.data_dir, self.task, "train", f"{self.data_type}_*.parquet")
                val_loc = os.path.join(self.cfg.dataset.data_dir, self.task, "val", f"{self.data_type}_*.parquet")
            else:
                raise ValueError(f"Unexpected train dataset type: {self.cfg.dataset.train_dataset}. Please use 'combined' or 'separate'.")
            return train_loc, val_loc
        elif dataset_type == "test":
            if self.cfg.dataset.train_dataset == "combined":
                test_dir = os.path.join(self.cfg.dataset.data_dir, self.task, "test")
            elif self.cfg.dataset.train_dataset == "separate":
                test_dir = os.path.join(self.cfg.dataset.data_dir, self.task, "test", f"{self.data_type}_*.parquet")
            return test_dir
        else:
            raise ValueError(f"Unexpected dataset type: {dataset_type}. Please use 'train' or 'test'.")

    def setup(self, stage: str) -> None:
        if stage == "fit":
            if self.cfg.dataset.name == "CEPC":
                data_dir = self.get_CEPC_dataset_path(dataset_type="train")
                full_train_dataset = RowGroupDataset(data_loc=data_dir)
                train_dataset, val_dataset = io.train_val_split_shuffle(
                    concat_dataset=full_train_dataset,
                    val_split=self.cfg.training.data.fraction_valid,
                    max_waveforms_for_training=-1,
                    row_group_size=self.cfg.datasets.CEPC.row_group_size,
                )
            elif self.cfg.dataset.name == "FCC":
                train_dir = self.get_dataset_paths(dataset_type="train")
                val_dir = self.get_dataset_paths(dataset_type="val")
                self.train_dataset = RowGroupDataset(data_loc=train_dir)
                self.val_dataset = RowGroupDataset(data_loc=val_dir)
            self.train_dataset = self.iter_dataset(
                dataset=train_dataset,
            )
            self.val_dataset = self.iter_dataset(
                dataset=val_dataset,
            )
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.cfg.training.dataloader.batch_size,
                num_workers=self.cfg.training.num_dataloader_workers,
                prefetch_factor=self.cfg.training.prefetch_factor,
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.cfg.training.dataloader.batch_size,
                num_workers=self.cfg.training.num_dataloader_workers,
                prefetch_factor=self.cfg.training.prefetch_factor,
            )
        elif stage == "test":
            # TODO: Add test dataloader - different for FCC energies and CEPC particles.
            raise ValueError("Please do not use this datamodule for testing. Evaluate test dataset separatly.")
        else:
            raise ValueError(f"Unexpected stage: {stage}")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


class OneStepIterableDataset(BaseIterableDataset):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

    def build_tensors(self, data: ak.Array):
        """ Builds the input and target tensors from the data.
        
        Parameters:
            data : ak.Array
                The data used to build the tensors. The data is a chunk of the dataset loaded from a .parquet file.
        Returns:
            features : torch.Tensor
                The input features of the data
            targets : torch.Tensor
                The target values of the data
        """
        targets = np.array(data.target == 1, dtype=int)
        targets = np.sum(targets, axis=-1)
        targets = torch.tensor(targets, dtype=torch.float32)
        waveform = torch.tensor(ak.Array(data.waveform), dtype=torch.float32)
        return waveform, targets


class TwoStepPeakFindingIterableDataset(BaseIterableDataset):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

    def build_tensors(self, data: ak.Array):
        """ This iterable dataset is to be used for the first step (peak finding). For this, we have a target for each waveform window. When building the tensors, we flatten the waveforms, so we predict one value for each window. We target both primary and secondary peaks, setting a target of 1 for both of them, whereas background has a target of 0.

        Parameters:
            data : ak.Array
                The data used to build the tensors. The data is a chunk of the dataset loaded from a .parquet file.
        Returns:
            features : torch.Tensor
                The input features of the data
            targets : torch.Tensor
                The target values of the data
        """
        waveforms = ak.Array(data.waveform)
        wf_targets = ak.Array(data.target)
        wf_windows = ak.flatten(waveforms, axis=-2)
        wf_targets = ak.values_astype((wf_targets == 1) + (wf_targets == 2), int)
        target_windows = ak.flatten(wf_targets, axis=-1)  # TODO: Check shape
        wf_windows = torch.tensor(wf_windows, dtype=torch.float32)
        target_windows = torch.tensor(target_windows, dtype=torch.float32)
        return wf_windows, target_windows  # TODO: Unsqueeze?


class TwoStepClusterizationIterableDataset(BaseIterableDataset):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

    def build_tensors(self, data: ak.Array):
        """ This iterable dataset is to be used for the second step (clusterization). Here we use the predictions from the first step (peak finding) as input.

        Parameters:
            data : ak.Array
                The data used to build the tensors. The data is a chunk of the dataset loaded from a .parquet file.
        Returns:
            features : torch.Tensor
                The input features of the data
            targets : torch.Tensor
                The target values of the data
        """
        peaks  = ak.Array(data.predicted_peaks)  # TODO: Add predicted_peaks as the branch name in postprocessing.
        targets = ak.sum(data.target == 1, axis = -1)
        targets = torch.tensor(targets, dtype=torch.float32)
        peaks = torch.tensor(peaks, dtype=torch.float32)
        return peaks, targets


class TwoStepMinimalIterableDataset(BaseIterableDataset):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

    def build_tensors(self, data: ak.Array):
        """ This iterable dataset is to be used for the minimal two-step approach, where we only target the primary peaks with the peak finding. In principle this allows us to skip clusterization step, as we can sum all the predicted peaks. This approach is used for evaluating how much clusterization adds on top of the peak finding.
        The difference with the vanilla peak-finding in the vanilla two-step approach is, that we use only "primary" peaks as targets.

        Parameters:
            data : ak.Array
                The data used to build the tensors. The data is a chunk of the dataset loaded from a .parquet file.
        Returns:
            features : torch.Tensor
                The input features of the data
            targets : torch.Tensor
                The target values of the data
        """
        waveforms = ak.Array(data.waveform)
        wf_targets = ak.Array(data.target)
        wf_windows = ak.flatten(waveforms, axis=-2)
        wf_targets = ak.values_astype((wf_targets == 1), int)
        target_windows = ak.flatten(wf_targets, axis=-1)  # TODO: Check shape
        wf_windows = torch.tensor(wf_windows, dtype=torch.float32)
        target_windows = torch.tensor(target_windows, dtype=torch.float32)
        return wf_windows, target_windows  # TODO: Unsqueeze?

