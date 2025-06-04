import copy
import json
import os
import pdb
import time
from itertools import product

from hegnn.config.parse_config import load_config, load_parser
from hegnn.globals import results_dir
from hegnn.run_zinc_store_data import run_zinc

args = load_parser()
config = load_config()

config.n_epochs = 1
config.wandb = True
config.data.data_size = "subset"

config.model_params.num_layers = 6
config.n_epochs = 1
config.print_log = True
config.model_params.layer = "ZINCGINConv"
config.data.stored_input = True

config.model_params.mode = "original"

config.data.overwrite = False
config.batch_size = 50

config.model_params.hidden_dim = 16
all_results = {}

if args.depth_iter is None:
    depth_iter = [1]
else:
    depth_iter = args.depth_iter

if args.n_hops_iter is None:
    n_hops_iter = [5, 4, 3, 2, 1]
else:
    n_hops_iter = args.n_hops_iter

print(
    f"Starting data generation for n_hops {n_hops_iter} and depth _{depth_iter}, overwrite={config.data.overwrite}"
)


all_results_storage = []
all_results = {}

grid = [depth_iter, n_hops_iter]
i = 0
for settings in product(*grid):
    result_dict = {}
    for j in range(1):
        depth, n_hops = settings
        cur_config = copy.deepcopy(config)
        if i == 0:
            i += 1
        cur_config.model_params.depth = depth
        cur_config.data.n_hops = n_hops

        cur_config.run_name = f"n_hops_{cur_config.data.n_hops}_depth_{depth}"

        start_time = time.time()
        val_loss, train_loss, test_loss = run_zinc(cur_config)
        cur_config.model_params.pna_params.degrees = None
        result_dict = {
            "config": cur_config.to_dict(),
            "run": j,
            "results": {
                "val_loss": float(val_loss),
                "train_loss": float(train_loss),
                "test_loss": float(test_loss),
                "time": int(time.time() - start_time),
            },
        }
        all_results_storage.append(result_dict)

        sorted_results = sorted(
            all_results_storage, key=lambda x: x["results"]["val_loss"]
        )
        for result_dict in sorted_results:
            print("\n")
            run_name = result_dict["config"]["run_name"]
            val_loss = result_dict["results"]["val_loss"]
            train_loss = result_dict["results"]["train_loss"]
            test_loss = result_dict["results"]["test_loss"]
            runtime = result_dict["results"]["time"]
            print(
                f"{run_name}: val_loss={val_loss}, train_loss={train_loss}, test_loss={test_loss}, time={int(runtime)}"
            )
