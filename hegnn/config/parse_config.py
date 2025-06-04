import argparse
import pdb

import yaml

from hegnn.globals import config_path

parser = argparse.ArgumentParser(description="Train a GNN on the ZINC dataset")

# DATA
parser.add_argument("--n_hops", type=int)
parser.add_argument("--storage_size", type=int)
parser.add_argument("--data_size", type=int)
parser.add_argument("--no_storage", action="store_true")
parser.add_argument("--overwrite", action="store_true")

# MODEL
parser.add_argument("--model", type=str)

# HEGNN options
parser.add_argument("--depth", type=int)
parser.add_argument("--hidden_dim", type=int)
parser.add_argument("--mode", type=str)  # one of splitbatch, original
parser.add_argument("--aggregators", nargs="+", type=str)
parser.add_argument("--scalers", nargs="+", type=str)
parser.add_argument("-different_layers", action="store_true")
parser.add_argument("--layer", type=str)
parser.add_argument("--dropout", type=float)
parser.add_argument("--no_batch_norm", action="store_true")
parser.add_argument("--num_layers", type=int)

parser.add_argument("--n_hops_iter", nargs="+", type=int)
parser.add_argument("--depth_iter", nargs="+", type=int)

# TRAINING PARAMS
parser.add_argument("--n_epochs", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("-print_log", action="store_true")
parser.add_argument("-wandb", action="store_true")

###############################3
parser.add_argument("--experiment_name", type=str)
parser.add_argument("-cycle_residual", action="store_true")
parser.add_argument("--run_index", type=int)

args = parser.parse_args()
args.stored_input = not args.no_storage
args.batch_norm = not args.no_batch_norm


def recursive_dict_update(config_dict, args):
    for k, v in config_dict.items():
        if isinstance(v, dict):
            config_dict[k] = recursive_dict_update(v, args)
        else:
            if hasattr(args, k) and getattr(args, k) is not None:
                config_dict[k] = getattr(args, k)
    return config_dict


class DotDict(dict):
    def __init__(self, data=None):
        super().__init__()
        if data is None:
            data = {}
        for key, value in data.items():
            self[key] = self._wrap(value)

    def _wrap(self, value):
        if isinstance(value, dict):
            return DotDict(value)
        elif isinstance(value, list):
            return [self._wrap(v) for v in value]
        return value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = self._wrap(value)

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")

    def to_dict(self):
        """
        Convert the DotDict to a regular dictionary.
        """
        return {
            k: (v.to_dict() if isinstance(v, DotDict) else v) for k, v in self.items()
        }


def load_config():
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return DotDict(recursive_dict_update(config_dict, args))


def load_parser():
    return args
