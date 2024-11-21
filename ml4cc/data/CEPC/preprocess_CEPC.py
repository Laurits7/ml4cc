import os
import glob
import json
import hydra
import uproot
import numpy as np
import awkward as ak
from ml4cc.tools.data import io
from omegaconf import DictConfig, OmegaConf


def print_config(cfg: DictConfig):
    print("Used configuration:")
    print(json.dumps(OmegaConf.to_container(cfg), indent=4))


def save_processed_data(arrays: ak.Array, path: str):
    output_path = path.replace("CEPC/data/", "CEPC/preprocessed_data/")
    output_path = output_path.replace(".root", ".parquet")
    output_dir = os.path.dirname(output_path)
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    io.save_record_to_file(data=arrays, output_path=output_path)


def process_root_file(
        path: str,
        cfg: DictConfig,
        type: int = 0
):
    """ Processes the .root file into a more ML friendly format.

    Caveats: from data_gen_txt.C the 'id' field is always set to -1

    Parameters:
        path : str
            Path to the .root file to be processed
        cfg : DictConfig
            The configuration to be used for processing
        type : int
            TODO: Type of WHAT?? Possible values: [0, 1, 2]. {0: signal, 1: background}??
            TODO: Meanings OLD: {0: pri, 1: sec, 2: bkg}
            TODO: Meanings NEW: {0: bkg, 1: pri, 2: sec} -> C
            TODO: Currently it seems that tag=1 is primary peak, tag=2 secondary peak

    """
    arrays = io.load_root_file(path=path, tree_path=cfg.preprocessing.tree_path, branches=cfg.preprocessing.branches)
    branches_to_create = cfg.branches_to_create
    new_data = {
        "EventNo": np.arange(len(arrays)), # TODO: Is this actually needed?
        "ID": 1, # TODO: Map arrays['tag'] as {0: -1, 1: 1, 2: 0} for what exact reason? Meaning background is given id=-1, primary peak id=1 and secondary peak id=0??
        "Shift": 1,  # TODO
        "Sigma": 1,  # TODO
        "Time": 1,  # TODO
    }

    save_processed_data(arrays, path)


@hydra.main(config_path="../../config/datasets", config_name="CEPC", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = cfg.CEPC
    print_config(cfg)
    for training_type in cfg.preprocessing.training_types:
        for dataset in ["test", "train"]:
            if (training_type == "clusterization") and (dataset == "test"):
                sample_dir = os.path.join(dataset, "*")
            else:
                sample_dir = dataset
            input_file_wcp = os.path.join(
                cfg.preprocessing.raw_input_dir,
                training_type,
                sample_dir,
                "*"
            )
            print(input_file_wcp)
            paths = glob.glob(input_file_wcp)
            print("Len paths:", len(paths))
            for path in paths[:3]:
                process_root_file(path, cfg)


if __name__ == '__main__':
    main()
