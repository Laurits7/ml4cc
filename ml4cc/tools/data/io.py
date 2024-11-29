import uproot
import awkward as ak


def load_root_file(path: str, tree_path: str = "sim", branches: list = None) -> ak.Array:
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


def save_record_to_file(data: ak.Array, output_path: str) -> None:
    print(f"Saving {len(data)} processed entries to {output_path}")
    ak.to_parquet(data, output_path, row_group_size=1024)
