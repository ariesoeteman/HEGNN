import copy
import json
import os
import pdb
import time
from itertools import product

from hegnn.config.parse_config import load_config
from hegnn.globals import results_dir
from hegnn.run_zinc_store_data import run_zinc

config = load_config()
config.wandb = True
config.data.data_size = "subset"
config.n_epochs = 500

config.batch_size = 50
config.model_params.num_layers = 6
config.print_log = True
config.model_params.layer = "ZINCGINConv"
config.data.stored_input = True
config.data.n_hops = 3
config.model_params.mode = "original"

weight_decay_iter = [0.0]
dropout_iter = [0.0]
hidden_dim_iter = [64]
learning_rate_factor_iter = [0.5]
gradient_fraction_iter = [0.5]
pool_subgraphs_iter = [False, True]
all_results = {}
depth_merge_iter = ["concat"]
n_hops_iter = [4, 3, 2, 1]
depth_iter = [2]

# experiment_name = "varying_n_hops_experiment"
experiment_name = "depth_2_n_hops_varying"
all_results_storage = []
all_results = {}

grid = [
    pool_subgraphs_iter,
    n_hops_iter,
    gradient_fraction_iter,
    depth_iter,
    dropout_iter,
    hidden_dim_iter,
    depth_merge_iter,
]
i = 0
for settings in product(*grid):
    result_dict = {}
    for j in range(1):
        (
            pool_subgraphs,
            n_hops,
            gradient_fraction,
            depth,
            dropout,
            hidden_dim,
            depth_merge,
        ) = settings
        cur_config = copy.deepcopy(config)
        if i == 0:
            cur_config.data.overwrite = True
            i += 1
        cur_config.data.n_hops = n_hops
        cur_config.model_params.depth_merge = depth_merge
        cur_config.model_params.pool_subgraphs = pool_subgraphs
        cur_config.model_params.depth = depth
        cur_config.model_params.dropout = dropout
        cur_config.model_params.hidden_dim = hidden_dim
        cur_config.model_params.gradient_fraction = gradient_fraction
        cur_config.data.input_dim = cur_config.model_params.hidden_dim

        cur_config.run_name = f"ZINCGIN_run_{j}"
        cur_config.run_name = f"n_hops_{cur_config.data.n_hops}_depth_{depth}_gradient_fraction_{gradient_fraction}_hidden_{hidden_dim}_dropout_{dropout}_merge_{depth_merge}_global_pool_{pool_subgraphs}"

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
        # results_file = os.path.join(results_dir, experiment_name + ".json")
        # os.makedirs(os.path.dirname(results_file), exist_ok=True)
        # with open(results_file, "w") as f:
        #     json.dump(all_results_storage, f, indent=4)
        # print(f"Results saved to {results_file}")
        ###########################3
