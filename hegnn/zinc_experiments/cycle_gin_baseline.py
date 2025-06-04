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
config.wandb = False
config.data.data_size = "subset"
config.n_epochs = 600

config.cycle_residual = True

config.batch_size = 1000
config.model_params.num_layers = 6
# config.print_log = True
config.model_params.layer = "ZINCGINConv"
config.data.stored_input = True

config.model_params.depth = 0
config.data.overwrite = False

config.model_params.dropout = 0.0
config.learning_rate.factor = 0.5
config.weight_decay = 1e-3

dropout_iter = [0.0, 0.2]
# hidden_dim_iter = [256, 128]
######
hidden_dim_iter = [16]
config.data.data_size = "smallsubset"
#######
config.cycles = [3, 4, 5, 6, 7, 8, 9, 10]

all_results = {}

if args.experiment_name is None:
    raise ValueError("Please specify an experiment name.")
else:
    experiment_name = args.experiment_name

all_results_storage = []
all_results = {}

grid = [dropout_iter, hidden_dim_iter]

n_runs = 10
i = 0
for settings in product(*grid):
    result_dict = {}
    dropout, hidden_dim = settings
    for j in range(n_runs):
        cur_config = copy.deepcopy(config)
        cur_config.model_params.dropout = dropout
        cur_config.model_params.hidden_dim = hidden_dim

        cur_config.run_name = f"ZINCGIN_run_{j}_cycles_{cur_config.cycles}_dropout_{dropout}_hidden_{hidden_dim}"

        start_time = time.time()
        val_loss, train_loss, test_loss, _, _, _ = run_hegnn(cur_config)
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
        # ########################
        results_file = os.path.join(results_dir, experiment_name + ".json")
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(all_results_storage, f, indent=4)
        print(f"Results saved to {results_file}")
        # ###########################3
