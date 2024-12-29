import os
import glob
import time
import hydra
import random
import numpy as np
import awkward as ak
from ml4cc.tools.data import io
from omegaconf import DictConfig


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


def process_root_file(path: str, cfg: DictConfig) -> None:
    """ Processes the .root file into a more ML friendly format.

    Parameters:
        path : str
            Path to the .root file to be processed
        cfg : DictConfig
            The configuration to be used for processing

    Returns:
        None
    """
    arrays = io.load_root_file(path=path, tree_path=cfg.tree_path, branches=cfg.branches)
    # Shuffle and divide into test and train
    total_len = len(arrays)
    indices = list(range(total_len))
    random.seed(42)
    random.shuffle(indices)
    num_train_rows = int(np.ceil(total_len*cfg.train_frac))
    train_indices = indices[:num_train_rows]
    test_indices = indices[num_train_rows:]

    primary_ionization_mask = indices_to_booleans(arrays['time'][arrays['tag'] == 1], arrays['wf_i'])
    secondary_ionization_mask = indices_to_booleans(arrays['time'][arrays['tag'] == 2], arrays['wf_i'])
    target = (primary_ionization_mask * 1) + (secondary_ionization_mask * 2)
    processed_array = ak.Array({
        "waveform": arrays['wf_i'],
        "target": target
    })
    train_array = processed_array[train_indices]
    test_array = processed_array[test_indices]
    save_processed_data(train_array, path, "train")
    save_processed_data(test_array, path, "test")


def save_processed_data(arrays: ak.Array, path: str, dataset: str) -> None:
    """
    Parameters:
        arrays: ak.Array
            The awkward array to be saved into file
        path: str
            The path of the output file

    Returns:
        None
    """
    output_path = path.replace("data/", f"preprocessed_data/{dataset}/")
    output_path = output_path.replace(".root", ".parquet")
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    io.save_array_to_file(data=arrays, output_path=output_path)


@hydra.main(config_path="../../config/datasets", config_name="FCC", version_base=None)
def prepare_inputs(cfg: DictConfig) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    cfg = cfg.FCC.preprocessing
    paths_to_process_wcp = os.path.join(cfg.raw_input_dir, "*")
    zero_start = time.time()
    for input_path in glob.glob(paths_to_process_wcp):
        start = time.time()
        process_root_file(input_path, cfg)
        end = time.time()
        print(f"Processing {input_path} took {end-start} seconds.")
    zero_end = time.time()
    print(f"Processing all samples took {zero_end-zero_start} seconds.")


if __name__ == '__main__':
    prepare_inputs()
