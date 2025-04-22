import os
import glob
import time
import hydra
import numpy as np
import awkward as ak
from omegaconf import DictConfig
from ml4cc.tools.data import preprocessing as pp


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
        pp.process_onestep_fcc_root_file(input_path, cfg)
        end = time.time()
        print(f"Processing {input_path} took {end-start:.2f} seconds.")
    zero_end = time.time()
    print(f"Processing all samples took {zero_end-zero_start:.2f} seconds.")


if __name__ == '__main__':
    prepare_inputs()
