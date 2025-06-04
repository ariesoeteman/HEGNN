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
config.wandb = True
config.data.data_size = "subset"

config.n_epochs = 1000
config.batch_size = 1000
################# For less memory usage
# config.batch_size = 100
###############

config.model_params.hidden_dim = 256

config.model_params.num_layers = 6
config.model_params.layer = "ZINCGINConv"
config.data.stored_input = True

config.model_params.depth = 1


config.model_params.dropout = 0.1
config.learning_rate.factor = 0.75
config.weight_decay = 1e-3
config.model_params.uid_residual = True
config.cycles = [3, 4, 5, 6, 7, 8, 9, 10]

if args.n_hops_iter is None:
    n_hops_iter = [5, 4, 3, 2, 1]
else:
    n_hops_iter = args.n_hops_iter

all_results = {}

if args.experiment_name is None:
    raise ValueError("Please specify an experiment name.")
else:
    experiment_name = args.experiment_name

config.model_params.cycle_residual = args.cycle_residual

all_results_storage = []
all_results = {}

n_runs = 5
for n_hops in n_hops_iter:
    result_dict = {}
    for j in range(n_runs):
        cur_config = copy.deepcopy(config)
        cur_config.data.n_hops = n_hops

        cur_config.run_name = f"ZINCGIN_run_{j}_n_hops_{n_hops}_depth_{cur_config.model_params.depth}_cycle_residual_{cur_config.model_params.cycle_residual}"

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
