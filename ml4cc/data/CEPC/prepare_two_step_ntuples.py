import os
import glob
import awkward as ak


def prepare_data(data: ak.Array, window_size: int = 15, stride: int = 15):
    waveform = ak.Array(data.waveform)
    target = ak.Array(data.target)
    all_waveforms = []
    all_targets = []
    for wf_idx, (wf, t) in enumerate(zip(waveform, target)):
        targets = []
        waveforms = []
        for start_idx in range(0, len(wf) - window_size + 1, stride):
            waveforms.append(wf[start_idx: start_idx + window_size])
            if 1 in t[start_idx: start_idx + window_size]:
                targets.append(1)
            else:
                targets.append(0)
        all_waveforms.append(waveforms)
        all_targets.append(targets)
    all_waveforms = ak.Array(all_waveforms)
    all_targets = ak.Array(all_targets)
    return all_waveforms, all_targets


def save_ntuple(data: ak.Array, output_path: str) -> None:
    print(f"Saving {len(data)} processed entries to {output_path}")
    ak.to_parquet(data, output_path, row_group_size=1024)


def process_single_file(input_path: str, output_dir: str):
    data = ak.from_parquet(input_path)
    all_waveforms, all_targets = prepare_data(data)
    output_data = {"waveform": all_waveforms, "target": all_targets}
    output_path = os.path.join(output_dir, os.path.basename(input_path))
    save_ntuple(output_data, output_path)


def process_ntuples(input_dir, output_dir):
    for input_path in glob.glob(os.path.join(input_dir, "*")):
        print(f"Processing {input_path}")
        process_single_file(input_path=input_path, output_dir=output_dir)


INPUT_DIR = "/scratch/project_465001293/ML4CC/data/CEPC/preprocessed_data/peakFinding/"
OUTPUT_DIR = "/scratch/project_465001293/ML4CC/data/CEPC/preprocessed_twoStep_data/peakFinding/"
if __name__ == '__main__':
    for dset in ['test', 'train']:
        output_dset_dir = os.path.join(OUTPUT_DIR, dset)
        input_dset_dir = os.path.join(INPUT_DIR, dset)
        os.makedirs(output_dset_dir, exist_ok=True)
        process_ntuples(input_dir=input_dset_dir, output_dir=output_dset_dir)