import os
import random
import numpy as np
import awkward as ak
from ml4cc.tools.data import io
from omegaconf import DictConfig


def save_processed_data(arrays: ak.Array, path: str, cfg: DictConfig, data_type: str = "one_step_data", dataset: str = "") -> None:
    """
    Parameters:
        arrays: ak.Array
            The awkward array to be saved into file
        path: str
            The path of the output file
        data_type: str
            [default: one_step_data] Data type for the training, either one-step or two-step
        dataset: str
            [default: ""] The dataset to be used for the training, either train, val or test

    Returns:
        None
    """
    dataset_dir = f"{dataset}/" if dataset != "" else dataset
    output_path = path.replace("data/", f"{data_type}/{dataset_dir}")
    output_path = output_path.replace(cfg.host.data_dir, cfg.host.slurm.queue.preprcessing.output_dir)
    output_path = output_path.replace(".root", ".parquet")
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    io.save_array_to_file(data=arrays, output_path=output_path)


def indices_to_booleans(indices: ak.Array, array_to_slice: ak.Array) -> ak.Array:
    """ Creates boolean array from indices for masking 'array_to_slice'

    Parameters:
        indices : ak.Array
            The array containing the indices to pick in the `array_to_slice`
        array_to_slice : ak.Array
            The array for which the mask is created.

    Returns:
        mask : ak.Array
            Boolean array used to mask the `array_to_slice`.
    """
    whole_set, in_set = ak.unzip(ak.cartesian([ak.local_index(array_to_slice), indices], nested=True))
    return ak.any(whole_set == in_set, axis=-1)


def train_val_test_split(arrays: ak.Array, cfg: DictConfig) -> tuple:
    """ Splits the data into train, val and test sets.
    
    Parameters:
        arrays : ak.Array
            The array to be split into train, val and test sets
        cfg : DictConfig
            The configuration to be used for processing
    Returns:
        train_indices : list
            The indices of the train set
        test_indices : list
            The indices of the test set
        val_indices : list
            The indices of the val set
    """
    total_len = len(arrays)
    indices = list(range(total_len))
    random.seed(42)
    random.shuffle(indices)
    num_train_rows = int(np.ceil(total_len*cfg.train_val_test_split[0]))
    num_val_rows =  int(np.ceil(total_len*cfg.train_val_test_split[1]))
    train_indices = indices[:num_train_rows]
    val_indices = indices[num_train_rows: num_train_rows + num_val_rows]
    test_indices = indices[num_train_rows + num_val_rows:]
    return train_indices, test_indices, val_indices


def save_split_data(arrays, path, train_indices, val_indices, test_indices, cfg: DictConfig, data_type: str = "one_step_data") -> None:
    """ Saves the split data into train, val and test sets.

    Parameters:
        arrays : ak.Array
            The array to be split into train, val and test sets
        path : str
            The path of the output file
        train_indices : list
            The indices of the train set
        val_indices : list
            The indices of the val set
        test_indices : list
            The indices of the test set
        data_type : str
            [default: one_step_data] Data type for the training, either one-step or two-step

    Returns:
        None
    """
    train_array = arrays[train_indices]
    val_array = arrays[val_indices]
    test_array = arrays[test_indices]
    save_processed_data(train_array, path, data_type=data_type, cfg=cfg, dataset="train")
    save_processed_data(val_array, path, data_type=data_type, cfg=cfg, dataset="val")
    save_processed_data(test_array, path, data_type=data_type, cfg=cfg, dataset="test")


def process_onestep_root_file(path: str, cfg: DictConfig) -> None:
    """ Processes the .root file into a more ML friendly format for one-step training.

    Parameters:
        path : str
            Path to the .root file to be processed
        cfg : DictConfig
            The configuration to be used for processing

    Returns:
        None
    """
    arrays = io.load_root_file(path=path, tree_path=cfg.dataset.tree_path, branches=cfg.dataset.branches)
    primary_ionization_mask = indices_to_booleans(arrays['time'][arrays['tag'] == 1], arrays['wf_i'])
    secondary_ionization_mask = indices_to_booleans(arrays['time'][arrays['tag'] == 2], arrays['wf_i'])
    target = (primary_ionization_mask * 1) + (secondary_ionization_mask * 2)
    processed_array = ak.Array({
        "waveform": arrays['wf_i'],
        "target": target
    })

    if cfg.dataset.name == "FCC":
        train_indices, test_indices, val_indices = train_val_test_split(arrays, cfg.preprocessing)
        save_split_data(processed_array, path, train_indices, val_indices, test_indices, cfg=cfg, data_type="one_step_data")
    elif cfg.dataset.name == "CEPC":
        save_processed_data(processed_array, path, cfg=cfg, data_type="one_step_data")
    else:
        raise ValueError(f"Unknown experiment: {cfg.dataset.name}")


def process_twostep_root_file(path: str, cfg: DictConfig, nleft: int = 5, nright: int = 9) -> None:
    """ Processes the peakfinding .root file into the format Guang used for 2-step training.

    Parameters:
        path : str
            Path to the .parquet file to be processed
        cfg : DictConfig
            The configuration to be used for processing
        nleft : int
            Defines the distance from the peak to the left edge in the waveform window
        nleft : int
            Defines the distance from the peak to the right edge in the waveform window

    Returns:
        None
    """
    arrays = io.load_root_file(path=path, tree_path=cfg.dataset.tree_path, branches=cfg.dataset.branches)
    all_windows = []
    all_targets = []
    for event_idx in range(len(arrays.time)):
        wf_windows = []
        target_window = []
        peak_indices = np.array(arrays.time[event_idx], dtype=int)
        for peak_idx, peak_loc in enumerate(peak_indices):
            if peak_loc < nleft: continue
            wf_window = arrays.wf_i[event_idx][peak_loc - nleft: peak_loc + nright + 1]
            target = arrays.tag[event_idx][peak_idx]
            wf_windows.append(wf_window)
            target_window.append(target)
        all_targets.append(target_window)
        all_windows.append(wf_windows)
    processed_array = ak.Array({
        "target": all_targets,
        "waveform": all_windows,
        "wf_i": arrays.wf_i,
    })
    if cfg.dataset.name == "FCC":
        train_indices, test_indices, val_indices = train_val_test_split(arrays, cfg.preprocessing)
        save_split_data(processed_array, path, train_indices, val_indices, test_indices, cfg=cfg, data_type="two_step_data")
    elif cfg.dataset.name == "CEPC":
        save_processed_data(processed_array, path, data_type="two_step_data", cfg=cfg)
    else:
        raise ValueError(f"Unknown experiment: {cfg.dataset.name}")
