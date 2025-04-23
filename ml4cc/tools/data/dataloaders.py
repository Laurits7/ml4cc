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
    def __init__(self, cfg: DictConfig, task: str, iter_dataset: IterableDataset):
        self.cfg = cfg
        self.task = task  # one-step or two-step
        # self.
        self.iter_dataset = iter_dataset
        self.batch_size = cfg.training.dataloader.batch_size[self.model]
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
            data_dir : str
                The directory of the dataset files
        """
        # TODO: Do test dataset path for different particle types and combination
        data_dir = os.path.join(self.cfg.dataset.data_dir, self.task, dataset_type)
        return data_dir

    def get_FCC_dataset_path(self, dataset_type: str) -> str:
        """Returns the directory of the dataset files for FCC
        Parameters:
            dataset_type : str
                The type of the dataset. Can be "train" or "test"

        Returns:
            train_dir : str
                The directory of the train dataset files
            val_dir : str
                The directory of the val dataset files
        """
        # TODO: DO train and test dataset for different energies? and combination
        train_dir = os.path.join(self.cfg.dataset.data_dir, self.task, dataset_type)
        val_dir = os.path.join(self.cfg.dataset.data_dir, self.task, dataset_type)
        return train_dir, val_dir

    def setup(self, stage: str) -> None:
        if stage == "fit":
            if self.cfg.dataset.name == "CEPC":
                data_dir = self.get_CEPC_dataset_path(dataset_type="train")
                full_train_dataset = RowGroupDataset(data_loc=data_dir)  # TODO: Maybe ConcatDataset step needed.
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
                batch_size=self.batch_size,
                num_workers=self.cfg.training.num_dataloader_workers,
                prefetch_factor=self.cfg.training.prefetch_factor,
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.cfg.training.num_dataloader_workers,
                prefetch_factor=self.cfg.training.prefetch_factor,
            )
        elif stage == "test":
            raise ValueError("Please do not use this datamodule for testing. Evaluate test dataset separatly.")
        else:
            raise ValueError(f"Unexpected stage: {stage}")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

# TODO: Add test dataloader - different for FCC energies and CEPC particles.
# Possibility to split by energy FCC trainig.

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
        """ Builds the input and target tensors from the data.
        # TODO: Add better description of how and why the tensors are built as they are.
        
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


class TwoStepClusterizationIterableDataset(BaseIterableDataset):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

    def build_tensors(self, data: ak.Array):
        """ Builds the input and target tensors from the data.
        # TODO: Add better description of how and why the tensors are built as they are.

        Parameters:
            data : ak.Array
                The data used to build the tensors. The data is a chunk of the dataset loaded from a .parquet file.
        Returns:
            features : torch.Tensor
                The input features of the data
            targets : torch.Tensor
                The target values of the data
        """
        # targets = np.array(data.target == 1, dtype=int)
        # targets = np.sum(targets, axis=-1)
        # targets = torch.tensor(targets, dtype=torch.float32)
        # waveform = torch.tensor(ak.Array(data.waveform), dtype=torch.float32)
        pass
        # return waveform, targets


class TwoStepMinimalIterableDataset(BaseIterableDataset):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

    def build_tensors(self, data: ak.Array):
        """ Builds the input and target tensors from the data.
        # TODO: Add better description of how and why the tensors are built as they are.

        Parameters:
            data : ak.Array
                The data used to build the tensors. The data is a chunk of the dataset loaded from a .parquet file.
        Returns:
            features : torch.Tensor
                The input features of the data
            targets : torch.Tensor
                The target values of the data
        """
        # targets = np.array(data.target == 1, dtype=int)
        # targets = np.sum(targets, axis=-1)
        # targets = torch.tensor(targets, dtype=torch.float32)
        # waveform = torch.tensor(ak.Array(data.waveform), dtype=torch.float32)
        pass
        # return waveform, targets