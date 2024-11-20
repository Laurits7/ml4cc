import hydra
from omegaconf import DictConfig
import uproot
import awkward as ak


def load_root_file(path: str, tree_path: str = "sim", branches: list = None):
    """ Loads the CEPC dataset .root file.

    Parameters:
        path : str
            Path to the .root file
        tree_path : str
            Path in the tree in the .root file.
        branches : list
            [default: None] Branches to be loaded from the .root file. By default, all branches will be loaded.

    Returns:
        array : ak.Array
            Awkward array containing the .root file data
    """
    with uproot.open(path) as in_file:
        tree = in_file[tree_path]
        arrays = tree.arrays(branches)
    return arrays


def process_root_file(path: str, cfg: DictConfig):
    arrays = load_root_file(path=path, tree_path=cfg.preprocessing.tree_path, branches=cfg.preprocessing.branches)


@hydra.main(config_path="../config/datasets", config_name="CEPC", version_base=None)
def main():

    pass


if __name__ == '__main__':
    main()
