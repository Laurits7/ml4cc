import os
import glob
import time
import hydra
import numpy as np
from omegaconf import DictConfig
from ml4cc.tools import general as g
from ml4cc.tools.data import slurm_tools as st
from ml4cc.tools.data import preprocessing as pp


def prepare_slurm_inputs(input_files: list, cfg: DictConfig) -> None:
    input_path_chunks = list(np.array_split(input_files, len(input_files) // cfg.preprocessing.slurm.files_per_job))
    print(f"From {len(input_files)} input files created {len(input_path_chunks)} chunks")
    st.multipath_slurm_processor(input_path_chunks=input_path_chunks, job_script=__file__, cfg=cfg.host)


def prepare_fcc_inputs(cfg: DictConfig) -> list:
    wcp_path = os.path.join(cfg.raw_input_dir, "*")
    paths_to_process = list(glob.glob(wcp_path))
    return paths_to_process


def prepare_cepc_inputs(cfg: DictConfig) -> list:
    all_paths_to_process = []
    for dataset in ["test", "train"]:
        input_file_wcp = os.path.join(
            cfg.raw_input_dir,
            dataset,
            "*"
        )
        raw_data_input_paths = list(glob.glob(input_file_wcp))
        all_paths_to_process.extend(raw_data_input_paths)
    return all_paths_to_process


def prepare_inputs(cfg: DictConfig) -> None:
    experiment = cfg.dataset.name
    all_paths_to_process = []
    if experiment == "FCC":
        all_paths_to_process = prepare_fcc_inputs(cfg.dataset)
    elif experiment == "CEPC":
        all_paths_to_process = prepare_cepc_inputs(cfg.dataset)
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    if cfg.preprocessing.slurm.use_it:
        prepare_slurm_inputs(input_files=all_paths_to_process, cfg=cfg)
    else:
        process_files(input_files=all_paths_to_process, cfg=cfg)


def process_files(input_files: list, cfg: DictConfig) -> None:
    for path in input_files:
        file_start_time = time.time()
        if cfg.preprocessing.data_type == "two_step":
            pp.process_twostep_root_file(path, cfg)
        elif cfg.preprocessing.data_type == "one_step":
            pp.process_onestep_root_file(path, cfg)
        else:
            raise ValueError(f"Unknown data type: {cfg.preprocessing.data_type}")
        file_end_time = time.time()
        print(f"Processing {path} took {file_end_time - file_start_time:.2f} seconds")


def run_job(cfg: DictConfig) -> None:
    input_paths = []
    with open(cfg.preprocessing.slurm.input_path, 'rt', encoding="utf-8") as inFile:
        for line in inFile:
            input_paths.append(line.strip('\n'))
    total_start_time = time.time()
    process_files(input_files=input_paths, cfg=cfg)
    total_end_time = time.time()
    print(f"Processing {len(input_paths)} took {total_end_time - total_start_time:.2f} seconds")


@hydra.main(config_path="../config", config_name="main", version_base=None)
def main(cfg: DictConfig) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    g.print_config(cfg)

    if cfg.preprocessing.slurm.slurm_run:
        run_job(cfg)
    else:
        prepare_inputs(cfg)


if __name__ == '__main__':
    main() # pylint: disable=E1120