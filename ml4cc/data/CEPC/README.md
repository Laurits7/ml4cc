# CEPC simulation files [![DOI:10.57760/sciencedb.16322](http://img.shields.io/badge/DOI-10.57760/sciencedb.16322-f9f107.svg)](https://doi.org/10.57760/sciencedb.16322) [![arXiv](https://img.shields.io/badge/arXiv-2402.16493-b31b1b.svg)](https://doi.org/10.48550/arXiv.2402.16493)

This is the dataset is introduced in the paper
"[_Cluster Counting Algorithm for Drift Chamber using LSTM and DGCNN_](https://doi.org/10.48550/arXiv.2402.16493)"
by Guang Zhao et al., and is made available at
[https://doi.org/10.57760/sciencedb.16322](https://doi.org/10.57760/sciencedb.16322).

## Downloading the dataset

In order to download the dataset run the following script with the option **-d**. If you want to additionally download
also the software the authors of the dataset used in training, specify also the option **-s**:

```bash
./download_CEPC_files.sh -sd
```

The meaning of the branches in the downloaded simulation files are the following:

| branch        | meaning                                                                                                                                          |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| wf_i          | wave-form. Has in principle the same info as the 'time' and 'height' branches. In order to get the time value, use just np.arange(len(waveform)) |
| sampling_rate | sampling rate                                                                                                                                    |
| ncls          | number of primary clusters                                                                                                                       |
| mom           | momentum                                                                                                                                         |
| cluster_size  | cluster size                                                                                                                                     |
| amp           | Signal amplitude **(?)**                                                                                                                         |
| time          | time t for a given waveform height h                                                                                                             |
| height        | height h of the waveform at time t                                                                                                               |
| tag           | Possible values [0, 1, 2]. 0: background, 1: primary peak, 2: secondary peak.                                                                    |
| mother        | The originating particle responsible for secondary ionization (?)                                                                                |
| naval         | ??                                                                                                                                               |
| xe            | x-coordinate where the electron cluster is generated                                                                                             |
| ye            | y-coordinate where the electron cluster is generated                                                                                             |
| ze            | z-coordinate where the electron cluster is generated                                                                                             |
| amp_pri       | Pre-amplification (?) of **WHAT?**                                                                                                               |

## Misc

If height > 3sigma then it is defined as a peak (??)

## Dataset structure

After downloading the CEPC dataset, in the current version the structure of the dataset will be as follows:

```text
data/
├─ peakFinding/
│  ├─ train/
│  │  ├─ signal_noise05_PFtrain_*.root
│  ├─ test/
│  │  ├─ signal_noise05_PFtest_*.root
├─ clusterization/
│  ├─ test/
│  │  ├─ pion/
│  │  │  ├─ signal_pi_12.5_*.root
│  │  ├─ kaon/
│  │  │  ├─ signal_K_12.5_*.root
│  ├─ train/
│  │  ├─ signal_noise05_*.root
```

However, in order to have similar data structure, the new structure would be the following:

```text
data/

├─ train/
│  ├─ signal_noise05_*.root
├─ test/
│  ├─ signal_pi_{5.0 .. 20.0}_{0..49}.root
│  ├─ signal_K_{5.0 .. 20.0}_{0..49}.root
```
Under the "train", the file numbering follows the following rules from the old structure:
``` text
clusterization/train/signal_noise05_{0..49}  ---> train/signal_noise05_{0..49}
peakFinding/train/signal_noise05_{0..49}  ---> train/ignal_noise05_{50..99}
peakFinding/test/signal_noise05_{0..49}  ---> train/ignal_noise05_{100..149}
```

## Processing CEPC dataset

To make the input files a bit more ML training friendly, we will only the necessary data from the raw **.root** files
and will work with **.parquet** files for the ML training:
