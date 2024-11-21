from tqdm import tqdm
import numpy as np
from omegaconf import DictConfig
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, IterableDataset


class CEPCDataset(Dataset):
    def __init__(
            self,
            cfg: DictConfig,
            task: str,
            dataset: str,
            samples: str = 'all'
    ):
        """ The base class for CEPC dataset.

        Parameters:
            cfg : DictConfig
                The configuration for the
            task : str
                The task for which data is loaded. Options: ['peakFinding' ; 'clusterization']
            dataset : str
                The dataset for which the data is loaded. Options: ['train', 'validation', 'test']
            samples : str
                Which samples to load. Options: ['kaon', 'pion', 'all']
        """
        self.cfg = cfg
        self.task = task
        self.dataset = dataset
        self.samples = samples
        super().__init__()

    def __getitem__(self, index):  # TODO
        temp = 'foobar'
        return temp[index]

    def __len__(self):  # TODO
        dataset = 'foobar'
        return len(dataset)


class CEPCDataLoader(DataLoader):
    def __init__(self):
        super().__init__()


class CEPCDataModule(LightningDataModule):
    def __init__(self):
        super.__init__()

    def setup(self):
        if stage == 'fit':
            self.train_dataset = 'xyz'
            self.val_dataset = 'xyz'
        elif stage == "test":
            self.test_dataset = 'xyz'

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass
