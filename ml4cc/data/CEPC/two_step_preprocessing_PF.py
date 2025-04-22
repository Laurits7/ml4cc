import os
import glob
import time
import hydra
import numpy as np
from omegaconf import DictConfig
from ml4cc.tools.data import slurm_tools as st
from ml4cc.tools.data import preprocessing as pp


def prepare_slurm_inputs(input_files: list, cfg: DictConfig, dataset: str) -> None:
    input_path_chunks = list(np.array_split(input_files, len(input_files) // cfg.files_per_job))
    print(f"From {len(input_files)} input files created {len(input_path_chunks)} chunks")
    st.multipath_slurm_processor(input_path_chunks=input_path_chunks, job_script=__file__, cfg=cfg, dataset=dataset)


def prepare_inputs(cfg: DictConfig, dataset) -> None:
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
        pp.prepare_slurm_inputs(input_files=all_paths_to_process, cfg=cfg.slurm, dataset=dataset)
    else:
        for path in all_paths_to_process:
            pp.process_twostep_root_file(path, cfg)


def run_job(cfg: DictConfig) -> None:
    input_paths = []
    with open(cfg.slurm.input_path, "rt") as inFile:
        for line in inFile:
            input_paths.append(line.strip("\n"))
    total_start_time = time.time()
    for path in input_paths:
        file_start_time = time.time()
        pp.process_twostep_root_file(path, cfg)
        file_end_time = time.time()
        print(f"Processing {path} took {file_end_time - file_start_time:.2f} seconds")
    total_end_time = time.time()
    print(f"Processing {len(input_paths)} took{total_end_time - total_start_time:.2f} seconds")


@hydra.main(config_path="../../config/datasets", config_name="CEPC", version_base=None)  # TODO: change according to new config structure
def main(cfg: DictConfig) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    cfg = cfg.CEPC.preprocessing
    pp.print_config(cfg)

    if cfg.slurm.slurm_run:
        run_job(cfg)
    else:
        prepare_inputs(cfg, dataset="CEPC")


if __name__ == "__main__":
    main()
