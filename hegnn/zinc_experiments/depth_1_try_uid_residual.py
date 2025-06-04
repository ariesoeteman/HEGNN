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
config.wandb = False
config.data.data_size = "subset"
config.n_epochs = 600

config.batch_size = 100
config.model_params.num_layers = 6
config.print_log = True

config.model_params.layer = "ZINCGINConv"
config.data.stored_input = True

config.model_params.depth = 0
config.data.overwrite = False


config.model_params.dropout = 0.0
config.model_params.hidden_dim = 128
config.learning_rate.factor = 0.5
config.weight_decay = 1e-3

pool_subgraphs_iter = [False, True]
uid_residual_iter = [True, False]
depth_merge_iter = ["concat"]

if args.n_hops_iter is None:
    n_hops_iter = [3]
else:
    n_hops_iter = args.n_hops_iter


all_results = {}

if args.experiment_name is None:
    raise ValueError("Please specify an experiment name.")
else:
    experiment_name = args.experiment_name

# experiment_name = "varying_n_hops_experiment"
all_results_storage = []
all_results = {}

grid = [uid_residual_iter, n_hops_iter, depth_merge_iter, pool_subgraphs_iter]
n_runs = 1
i = 0
for settings in product(*grid):
    result_dict = {}
    for j in range(n_runs):
        uid_residual, n_hops, depth_merge, pool_subgraph = settings
        cur_config = copy.deepcopy(config)

        cur_config.data.n_hops = n_hops
        cur_config.model_params.depth_merge = depth_merge
        cur_config.model_params.pool_subgraphs = pool_subgraph
        cur_config.model_params.uid_residual = uid_residual

        cur_config.data.input_dim = cur_config.model_params.hidden_dim
        cur_config.run_name = f"ZINCGIN_run_{j}_n_hops_{n_hops}_depth_{cur_config.model_params.depth}__merge_{depth_merge}_pool_subgraph_{pool_subgraph}_uid_residual_{uid_residual}"

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

        # Save results to a JSON file
        ########################
        results_file = os.path.join(results_dir, experiment_name + ".json")
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(all_results_storage, f, indent=4)
        print(f"Results saved to {results_file}")
        ###########################3
