import os
import math
import torch
import numpy as np
import awkward as ak
from omegaconf import DictConfig
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, IterableDataset
from ml4cc.tools.data import io


##########################################################################
##########################################################################
######################             Base classes            ###############
##########################################################################
##########################################################################


class RowGroupDataset(Dataset):
    def __init__(self, data_loc: str):
        self.data_loc = data_loc
        self.input_paths = io.get_all_paths(data_loc)
        self.row_groups = io.get_row_groups(self.input_paths)

    def __getitem__(self, index):
        return self.row_groups[index]

    def __len__(self):
        return len(self.row_groups)


class BaseIterableDataset(IterableDataset):
    """Base iterable dataset class to be used for different types of trainings."""

    def __init__(self, dataset: Dataset, device: str, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset
        self.row_groups = [row_group for row_group in self.dataset]
        self.num_rows = sum([rg.num_rows for rg in self.row_groups])
        self.device = device
        print(f"There are {'{:,}'.format(self.num_rows)} waveforms in the dataset.")

    def build_tensors(self, data: ak.Array):
        """Builds the input and target tensors from the data.

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

    def _move_to_device(self, batch):
        if isinstance(batch, (tuple, list)):
            return [self._move_to_device(x) for x in batch]
        return batch.to(self.device)

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
                # features, targets
                yield self._move_to_device(tensors[0][idx_wf]), self._move_to_device(tensors[1][idx_wf]), self._move_to_device(tensors[2][idx_wf])


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        iter_dataset: IterableDataset,
        data_type: str,
        debug_run: bool = False,
        device: str = "cpu",
        clusterization: bool = False,
    ):
        """Base data module class to be used for different types of trainings.
        Parameters:
            cfg : DictConfig
                The configuration file used to set up the data module.
            iter_dataset : IterableDataset
                The iterable dataset to be used for training and validation.
                Need to define a separate class for each training type, e.g. one_step, two_step_peak_finding,
                two_step_clusterization, two_step_minimal etc.
            data_type : str
                The type of the data. In case of CEPC it can be "kaon" or "pion".
                In case of FCC it is the different energies.
        """
        self.cfg = cfg
        # self.task = "two_step" if self.cfg.training.type == "two_step_minimal" else self.cfg.training.type
        self.task = "two_step"
        self.debug_run = debug_run
        self.data_type = data_type
        self.iter_dataset = iter_dataset
        self.device = device
        self.train_loader = None
        self.val_loader = None
        self.test_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.clusterization = clusterization
        self.num_row_groups = 2 if debug_run else None
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
            if self.clusterization:
                train_loc = os.path.join(
                    self.cfg.dataset.data_dir,
                    "two_step_pf",
                    "predictions",
                    "train",
                    f"{self.data_type}_*.parquet",
                )
                val_loc = os.path.join(
                    self.cfg.dataset.data_dir,
                    "two_step_pf",
                    "predictions",
                    "val",
                    f"{self.data_type}_*.parquet",
                )
            else:
                train_loc = os.path.join(self.cfg.dataset.data_dir, self.task, "train")
                val_loc = os.path.join(self.cfg.dataset.data_dir, self.task, "val")
            return train_loc, val_loc
        elif dataset_type == "test":
            if self.cfg.dataset.test_dataset == "combined":
                if self.clusterization:
                    test_dir = os.path.join(self.cfg.dataset.data_dir, "two_step_pf", "predictions", "test")
                else:
                    test_dir = os.path.join(self.cfg.dataset.data_dir, self.task, "test")
            elif self.cfg.dataset.test_dataset == "separate":
                if self.clusterization:
                    test_dir = os.path.join(
                        self.cfg.dataset.data_dir,
                        "two_step_pf",
                        "predictions",
                        "test",
                        f"{self.data_type}_*.parquet",
                    )
                else:
                    test_dir = os.path.join(self.cfg.dataset.data_dir, self.task, "test", f"*{self.data_type}*.parquet")
            else:
                raise ValueError(
                    f"Unexpected method for test dataset: {self.cfg.dataset.train_dataset}. Options:\
                                 'combined', 'separate']"
                )
            return test_dir
        else:
            raise ValueError(
                f"Unexpected train dataset type: {self.cfg.dataset.train_dataset}.\
                             Please use 'combined' or 'separate'."
            )

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
                if self.clusterization:
                    train_loc = os.path.join(
                        self.cfg.training.output_dir, "two_step_pf", "predictions", "train"
                    )
                    val_loc = os.path.join(
                        self.cfg.training.output_dir, "two_step_pf", "predictions", "val"
                    )
                else:
                    train_loc = os.path.join(self.cfg.dataset.data_dir, self.task, "train")
                    val_loc = os.path.join(self.cfg.dataset.data_dir, self.task, "val")
            elif self.cfg.dataset.train_dataset == "separate":
                if self.clusterization:
                    train_loc = os.path.join(
                        self.cfg.training.output_dir,
                        "two_step_pf",
                        "predictions",
                        "train",
                        f"{self.data_type}_*.parquet",
                    )
                    val_loc = os.path.join(
                        self.cfg.training.output_dir,
                        "two_step_pf",
                        "predictions",
                        "val",
                        f"{self.data_type}_*.parquet",
                    )
                else:
                    train_loc = os.path.join(
                        self.cfg.dataset.data_dir, self.task, "train", f"{self.data_type}_*.parquet"
                    )
                    val_loc = os.path.join(self.cfg.dataset.data_dir, self.task, "val", f"{self.data_type}_*.parquet")
            else:
                raise ValueError(
                    f"Unexpected train dataset type: {self.cfg.dataset.train_dataset}.\
                                 Please use 'combined' or 'separate'."
                )
            return train_loc, val_loc
        elif dataset_type == "test":
            if self.cfg.dataset.test_dataset == "combined":
                if self.clusterization:
                    test_dir = os.path.join(
                        self.cfg.training.output_dir, "two_step_pf", "predictions", "test"
                    )
                else:
                    test_dir = os.path.join(self.cfg.dataset.data_dir, self.task, "test")
            elif self.cfg.dataset.test_dataset == "separate":
                if self.clusterization:
                    test_dir = os.path.join(
                        self.cfg.training.output_dir,
                        "two_step_pf",
                        "predictions",
                        "test",
                        f"{self.data_type}_*.parquet",
                    )
                else:
                    test_dir = os.path.join(self.cfg.dataset.data_dir, self.task, "test", f"{self.data_type}_*.parquet")
            else:
                raise ValueError(
                    f"Unexpected method for testing dataset: {self.cfg.dataset.train_dataset}. Options:\
                                 'combined', 'separate']"
                )
            return test_dir
        else:
            raise ValueError(f"Unexpected dataset type: {dataset_type}. Please use 'train' or 'test'.")

    def setup(self, stage: str) -> None:
        batch_size = self.cfg.training.dataloader.batch_size if not self.debug_run else 1
        if stage == "fit":
            if self.cfg.dataset.name == "CEPC":
                train_dir, val_dir = self.get_CEPC_dataset_path(dataset_type="train")
                self.train_dataset = RowGroupDataset(data_loc=train_dir)[: self.num_row_groups]
                self.val_dataset = RowGroupDataset(data_loc=val_dir)[: self.num_row_groups]
            elif self.cfg.dataset.name == "FCC":
                train_dir, val_dir = self.get_FCC_dataset_path(dataset_type="train")
                self.train_dataset = RowGroupDataset(data_loc=train_dir)[: self.num_row_groups]
                self.val_dataset = RowGroupDataset(data_loc=val_dir)[: self.num_row_groups]
            self.train_dataset = self.iter_dataset(
                dataset=self.train_dataset,
                device=self.device,
                cfg=self.cfg
            )
            self.val_dataset = self.iter_dataset(
                dataset=self.val_dataset,
                device=self.device,
                cfg=self.cfg
            )
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                persistent_workers=True,
                num_workers=self.cfg.training.dataloader.num_dataloader_workers,
                # prefetch_factor=self.cfg.training.dataloader.prefetch_factor,
                
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                persistent_workers=True,
                num_workers=self.cfg.training.dataloader.num_dataloader_workers,
                # prefetch_factor=self.cfg.training.dataloader.prefetch_factor,
            )
        elif stage == "test":
            if self.cfg.dataset.name == "CEPC":
                test_dir = self.get_CEPC_dataset_path(dataset_type="test")
            elif self.cfg.dataset.name == "FCC":
                test_dir = self.get_FCC_dataset_path(dataset_type="test")
            else:
                raise ValueError(f"Unexpected dataset type: {self.cfg.dataset.name}. Please use 'CEPC' or 'FCC'.")
            self.test_dataset = RowGroupDataset(data_loc=test_dir)[: self.num_row_groups]
            self.test_dataset = self.iter_dataset(
                dataset=self.test_dataset,
                device=self.device,
                cfg=self.cfg
            )
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                persistent_workers=True,
                num_workers=self.cfg.training.dataloader.num_dataloader_workers,
                # prefetch_factor=self.cfg.training.dataloader.prefetch_factor,
            )
        else:
            raise ValueError(f"Unexpected stage: {stage}")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


class OneStepIterableDataset(BaseIterableDataset):
    def __init__(self, dataset: Dataset, device: str, cfg: DictConfig):
        super().__init__(dataset, device=device, cfg=cfg)

    def build_tensors(self, data: ak.Array):
        """Builds the input and target tensors from the data.

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
        mask = ak.ones_like(targets)
        targets = torch.tensor(targets, dtype=torch.float32)
        waveform = torch.tensor(ak.Array(data.waveform), dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.bool)
        return waveform, targets, mask


class OneStepWindowedIterableDataset(BaseIterableDataset):
    def __init__(self, dataset: Dataset, device: str, cfg: DictConfig):
        super().__init__(dataset, device=device, cfg=cfg)

    def build_tensors(self, data: ak.Array):
        """Builds the input and target tensors from the data.

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
        padded_windows = ak.fill_none(ak.pad_none(waveforms, 15, axis=-1), 0) # All windows are padded to 15 size
        zero_window = [0]*15
        none_padded_waveforms = ak.pad_none(padded_windows, self.cfg.dataset.max_peak_cands, axis=-2)
        padded_waveforms = ak.fill_none(none_padded_waveforms, zero_window, axis=-2)

        targets = np.sum(data.target == 1, axis=-1)
        mask = ak.ones_like(targets)
        targets = torch.tensor(targets, dtype=torch.float32)
        waveform = torch.tensor(ak.Array(padded_waveforms), dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.bool)
        return waveform, targets, mask


class TwoStepPeakFindingIterableDataset(BaseIterableDataset):
    def __init__(self, dataset: Dataset, device: str, cfg: DictConfig):
        super().__init__(dataset, device=device, cfg=cfg)

    def build_tensors(self, data: ak.Array, padding_size: int = 1650):
        """This iterable dataset is to be used for the first step (peak finding). For this, we have a target for each
        waveform window. When building the tensors, we flatten the waveforms, so we predict one value for each window.
        We target both primary and secondary peaks, setting a target of 1 for both of them, whereas background has a
        target of 0.

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
        padded_windows = ak.fill_none(ak.pad_none(waveforms, 15, axis=-1), 0) # All windows are padded to 15 size
        zero_window = [0]*15
        none_padded_waveforms = ak.pad_none(padded_windows, self.cfg.dataset.max_peak_cands, axis=-2)
        padded_waveforms = ak.fill_none(none_padded_waveforms, zero_window, axis=-2)
        padded_waveforms = ak.Array(padded_waveforms)

        wf_targets = ak.Array(data.target)
        wf_targets = ak.values_astype((wf_targets == 1) + (wf_targets == 2), int)
        padded_targets = ak.pad_none(wf_targets, self.cfg.dataset.max_peak_cands, axis=-1)
        padded_targets = ak.fill_none(padded_targets, -1, axis=-1)  # Fill the padded targets with 0

        wf_windows = torch.tensor(padded_waveforms, dtype=torch.float32)
        target_windows = torch.tensor(padded_targets, dtype=torch.float32)
        mask = ak.ones_like(padded_targets)
        mask = torch.tensor(mask, dtype=torch.bool)
        return wf_windows, target_windows, mask


class TwoStepClusterizationIterableDataset(BaseIterableDataset):
    def __init__(self, dataset: Dataset, device: str, cfg: DictConfig):
        super().__init__(dataset, device=device, cfg=cfg)

    def build_tensors(self, data: ak.Array):
        """This iterable dataset is to be used for the second step (clusterization).
        Here we use the predictions from the first step (peak finding) as input.

        Parameters:
            data : ak.Array
                The data used to build the tensors. The data is a chunk of the dataset loaded from a .parquet file.
        Returns:
            features : torch.Tensor
                The input features of the data
            targets : torch.Tensor
                The target values of the data
        """
        peaks = ak.Array(data.pred)
        mask = ak.fill_none(ak.pad_none(ak.ones_like(data.target), self.cfg.dataset.max_peak_cands, clip=True), 0,)
        targets = ak.sum(data.target == 1, axis=-1)
        targets = torch.tensor(targets, dtype=torch.float32)
        peaks = torch.tensor(peaks, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.bool)
        return peaks, targets, mask


class TwoStepMinimalIterableDataset(BaseIterableDataset):
    def __init__(self, dataset: Dataset, device: str, cfg: DictConfig):
        super().__init__(dataset, device=device, cfg=cfg)

    def build_tensors(self, data: ak.Array):
        """This iterable dataset is to be used for the minimal two-step approach,where we only target the primary
        peaks with the peak finding. In principle this allows us to skip clusterization step, as we can sum all the
        predicted peaks. This approach is used for evaluating how much clusterization adds on top of the peak finding.
        The difference with the vanilla peak-finding in the vanilla two-step approach is, that we use only "primary"
        peaks as targets.

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
        padded_windows = ak.fill_none(ak.pad_none(waveforms, 15, axis=-1), 0) # All windows are padded to 15 size
        zero_window = [0]*15
        none_padded_waveforms = ak.pad_none(padded_windows, self.cfg.dataset.max_peak_cands, axis=-2)
        padded_waveforms = ak.fill_none(none_padded_waveforms, zero_window, axis=-2)
        padded_waveforms = ak.Array(padded_waveforms)

        wf_targets = ak.Array(data.target)
        wf_targets = ak.values_astype((wf_targets == 1), int)
        padded_targets = ak.pad_none(wf_targets, self.cfg.dataset.max_peak_cands, axis=-1)
        padded_targets = ak.fill_none(padded_targets, -1, axis=-1)  # Fill the padded targets with 0

        wf_windows = torch.tensor(padded_waveforms, dtype=torch.float32)
        target_windows = torch.tensor(padded_targets, dtype=torch.float32)
        mask = ak.ones_like(padded_targets)
        mask = torch.tensor(mask, dtype=torch.bool)
        return wf_windows, target_windows, mask


class TwoStepMinimalDataModule(BaseDataModule):
    def __init__(self, cfg: DictConfig, data_type: str, debug_run: bool = False, device: str = "cpu"):
        """Data module for the minimal two-step approach. This is a simplified version of the two-step approach,
        where we only target the primary peaks with the peak finding. In principle this allows us to skip clusterization
        step, as we can sum all the predicted peaks. This approach is used for evaluating how much clusterization adds
        on top of the peak finding. The difference with the vanilla peak-finding in the vanilla two-step approach is,
        that we use only "primary" peaks as targets.
        """
        iter_dataset = TwoStepMinimalIterableDataset
        super().__init__(cfg=cfg, iter_dataset=iter_dataset, data_type=data_type, debug_run=debug_run, device=device)


class TwoStepPeakFindingDataModule(BaseDataModule):
    def __init__(self, cfg: DictConfig, data_type: str, debug_run: bool = False, device: str = "cpu"):
        """Data module for the two-step approach. This is a simplified version of the two-step approach, where we only
        target the primary peaks with the peak finding. In principle this allows us to skip clusterization step, as we
        can sum all the predicted peaks. This approach is used for evaluating how much clusterization adds on top of the
        peak finding. The difference with the vanilla peak-finding in the vanilla two-step approach is, that we use only
        "primary" peaks as targets.
        """
        iter_dataset = TwoStepPeakFindingIterableDataset
        super().__init__(cfg=cfg, iter_dataset=iter_dataset, data_type=data_type, debug_run=debug_run, device=device)


class TwoStepClusterizationDataModule(BaseDataModule):
    def __init__(self, cfg: DictConfig, data_type: str, debug_run: bool = False, device: str = "cpu"):
        """Data module for the two-step approach. This is a simplified version of the two-step approach, where we only
        target the primary peaks with the peak finding. In principle this allows us to skip clusterization step, as we
        can sum all the predicted peaks. This approach is used for evaluating how much clusterization adds on top of
        the peak finding. The difference with the vanilla peak-finding in the vanilla two-step approach is, that we use
        only "primary" peaks as targets.
        """
        iter_dataset = TwoStepClusterizationIterableDataset
        super().__init__(
            cfg=cfg,
            iter_dataset=iter_dataset,
            data_type=data_type,
            debug_run=debug_run,
            device=device,
            clusterization=True,
        )


class OneStepDataModule(BaseDataModule):
    def __init__(self, cfg: DictConfig, data_type: str, debug_run: bool = False, device: str = "cpu"):
        """Data module for the one-step approach. This is a simplified version of the two-step approach, where we only
        target the primary peaks with the peak finding. In principle this allows us to skip clusterization step, as we
        can sum all the predicted peaks. This approach is used for evaluating how much clusterization adds on top of the
        peak finding. The difference with the vanilla peak-finding in the vanilla two-step approach is, that we use only
        "primary" peaks as targets.
        """
        iter_dataset = OneStepIterableDataset
        super().__init__(cfg=cfg, iter_dataset=iter_dataset, data_type=data_type, debug_run=debug_run, device=device)


class OneStepWindowedDataModule(BaseDataModule):
    def __init__(self, cfg: DictConfig, data_type: str, debug_run: bool = False, device: str = "cpu"):
        """Data module for the one-step approach. This is a simplified version of the two-step approach, where we only
        target the primary peaks with the peak finding. In principle this allows us to skip clusterization step, as we
        can sum all the predicted peaks. This approach is used for evaluating how much clusterization adds on top of the
        peak finding. The difference with the vanilla peak-finding in the vanilla two-step approach is, that we use only
        "primary" peaks as targets.
        """
        iter_dataset = OneStepWindowedIterableDataset
        super().__init__(cfg=cfg, iter_dataset=iter_dataset, data_type=data_type, debug_run=debug_run, device=device)