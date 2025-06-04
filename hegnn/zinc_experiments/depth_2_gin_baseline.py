import copy
import json
import os
import pdb
import time
from itertools import product

from hegnn.config.parse_config import load_config, load_parser
from hegnn.globals import results_dir
from hegnn.run_hegnn import run_hegnn

args = load_parser()
config = load_config()
config.wandb = True
config.data.data_size = "subset"


config.data.dataset = "zinc"
config.model_params.layer = "ZINCGINConv"
config.data.stored_input = True

config.model_params.depth = 2
config.model_params.hidden_dim = 16
config.learning_rate.factor = 0.6
config.learning_rate.patience = 50
config.weight_decay = 1e-3
config.model_params.uid_residual = True
config.model_params.pool_subgraphs = True

if args.n_hops_iter is None:
    raise ValueError("Please specify n_hops_iter.")
else:
    n_hops_iter = args.n_hops_iter

if args.experiment_name is None:
    raise ValueError("Please specify an experiment name.")
else:
    experiment_name = args.experiment_name

if args.run_index is None:
    raise ValueError("Please specify a run index.")
else:
    run_index = args.run_index

# experiment_name = "varying_n_hops_experiment"
all_results_storage = []
all_results = {}

for n_hops in n_hops_iter:
    result_dict = {}
    cur_config = copy.deepcopy(config)
    cur_config.data.n_hops = n_hops
    cur_config.run_name = experiment_name

    start_time = time.time()
    val_loss, train_loss, test_loss, _, _, _ = run_hegnn(cur_config)
    cur_config.model_params.pna_params.degrees = None
    result_dict = {
        "config": cur_config.to_dict(),
        "run": run_index,
        "results": {
            "val_loss": float(val_loss),
            "train_loss": float(train_loss),
            "test_loss": float(test_loss),
            "time": int(time.time() - start_time),
        },
    }
    all_results_storage.append(result_dict)

    sorted_results = sorted(all_results_storage, key=lambda x: x["results"]["val_loss"])
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
