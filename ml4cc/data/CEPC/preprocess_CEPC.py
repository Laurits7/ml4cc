import os
import glob
import json
import time
import hydra
import ml4cc
import numpy as np
import awkward as ak
from ml4cc.tools.data import io
from ml4cc.tools.data import slurm_tools as st
from omegaconf import DictConfig, OmegaConf


def prepare_slurm_inputs(input_files: list, cfg: DictConfig):
    input_path_chunks = list(np.array_split(input_files, len(input_files) // cfg.files_per_job))
    print(f"From {len(input_files)} input files created {len(input_path_chunks)} chunks")
    st.multipath_slurm_processor(input_path_chunks=input_path_chunks, job_script=__file__)


def print_config(cfg: DictConfig) -> None:
    """ Prints the configuration used for the processing

    Parameters:
        cfg : DictConfig
            The configuration to be used

    Returns:
        None
    """
    print("Used configuration:")
    print(json.dumps(OmegaConf.to_container(cfg), indent=4))


def save_processed_data(arrays: ak.Array, path: str) -> None:
    """
    Parameters:
        arrays: ak.Array
            The awkward array to be saved into file
        path: str
            The path of the output file

    Returns:
        None
    """
    output_path = path.replace("CEPC/data/", "CEPC/preprocessed_data/")
    output_path = output_path.replace(".root", ".parquet")
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    io.save_record_to_file(data=arrays, output_path=output_path)


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
    primary_ionization_mask = indices_to_booleans(arrays['time'][arrays['tag'] == 1], arrays['wf_i'])
    secondary_ionization_mask = indices_to_booleans(arrays['time'][arrays['tag'] == 2], arrays['wf_i'])
    target = (primary_ionization_mask * 1) + (secondary_ionization_mask * 2)
    processed_array = ak.Array({
        "waveform": arrays['wf_i'],
        "target": target
    })
    save_processed_data(processed_array, path)


def prepare_inputs(cfg: DictConfig) -> None:
    all_paths_to_process = []
    for training_type in cfg.training_types:
        for dataset in ["test", "train"]:
            if (training_type == "clusterization") and (dataset == "test"):
                sample_dir = os.path.join(dataset, "*")
            else:
                sample_dir = dataset
            input_file_wcp = os.path.join(
                cfg.raw_input_dir,
                training_type,
                sample_dir,
                "*"
            )
            raw_data_input_paths = glob.glob(input_file_wcp)
            all_paths_to_process.extend(raw_data_input_paths)
    if cfg.slurm.use_it:
        prepare_slurm_inputs(input_files=all_paths_to_process, cfg=cfg.slurm)
    else:
        for path in all_paths_to_process:
            process_root_file(path, cfg)


def run_job(cfg: DictConfig) -> None:
    input_paths = []
    with open(cfg.slurm.input_path, 'rt') as inFile:
        for line in inFile:
            input_paths.append(line.strip('\n'))
    total_start_time = time.time()
    for path in input_paths:
        file_start_time = time.time()
        process_root_file(path, cfg)
        file_end_time = time.time()
        print(f"Processing {path} took {file_end_time - file_start_time:.2f} seconds")
    total_end_time = time.time()
    print(f"Processing {len(input_paths)} took{total_end_time - total_start_time:.2f} seconds")


@hydra.main(config_path="../../config/datasets", config_name="CEPC", version_base=None)
def main(cfg: DictConfig) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    print_config(cfg)
    cfg = cfg.CEPC

    if cfg.preprocessing.slurm.slurm_run:
        run_job(cfg.preprocessing)
    else:
        prepare_inputs(cfg.preprocessing)


if __name__ == '__main__':
    main()
