# Hierarchical Ego Graph Neural Networks

This repository contains the code for the paper "Logical Expressiveness of Graph Neural Networks with Hierarchical Node Individualization".

If you use this code and require further clarification or documentation, please create an issue on the GitHub repository.

## License
This code is released under the MIT License. See ./LICENSE for details.

## Citation
If you use this code in your work, please cite our paper LINK TO ARXIV:
Arie Soeteman and Balder ten Cate
Logical Expressiveness of Graph Neural Networks with Hierarchical Node Individualization

Bibtex:
ADD BIBTEX

## Features
- Support for node and edge features
- Subgraph generation and chunked storage
- Experiments on ZINC and Strongly Regular Graphs

## Dependencies
- Python 3.10
- pytorch 2.5.1
- torchaudio 2.5.1
- torchtriton 3.1.0
- torchgeometric 2.6.1
- torch-scatter 2.1.2
- torchmetrics 1.6.3

These are in the environment file, but note that custom installations may be required for cuda compatibility.

## Installation

```bash
cd HEGNN
conda env create -f environment.yml
conda activate hegnn_experiments
```

## Usage

Zinc experiments

```bash
cd hegnn/zinc_experiments

# Simple GIN test
python gin_baseline.py --experiment_name ginbaseline -print_log

# Depth 1
python depth_1_gin_baseline.py -print_log --experiment_name d1_gin_experiment

# Depth 2
python depth_2_gin_baseline.py -print_log --experiment_name d2_gin_experiment --batch_size 5 --n_hops_iter 3 --run_index 0
```

SRG experiment

```bash
cd hegnn/srg
python srg_experiment.py --experiment_name srg -print_log --batch_size 5 --depth 2
```

All configurations can be adjusted via the command line.

## Folder Structure

- `all_data/` — Strongly regular graphs, downloaded from https://github.com/twitter-research/cwn. ZINC data is downloaded at runtime.
- `config/` — Example configuration file and parser
- `data_processing/` — Data loaders and preprocessing
- `models/` — Model definitions
- `srg/` — Strongly Regular Graph experiment scripts
- `tests/` — Unit tests
- `utils/` — Utility functions
- `zinc_experiments/` — ZINC experiment scripts

## Citation and License

Not yet applicable.