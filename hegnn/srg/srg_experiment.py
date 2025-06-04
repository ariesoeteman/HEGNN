import copy
import json
import os
import pdb
import time
from itertools import product

from hegnn.config.parse_config import load_config, load_parser
from hegnn.globals import results_dir, srg_data_dir
from hegnn.run_hegnn import run_hegnn

if __name__ == "__main__":
    args = load_parser()
    config = load_config()
    config.data.dataset = "srg"
    # config.model_params.depth = 2
    config.wandb = False
    config.model_params.layer = "GINConv"

    ####################### tryout
    config.data.stored_input = True
    config.model_params.hidden_dim = 32
    config.n_epochs = 50
    config.learning_rate.patience = 15
    config.model_params.dropout = 0.0
    config.data.srg.n_isomorphisms = 30
    ########################

    config.cycles = [3, 4, 5, 6, 7, 8, 9, 10]
    config.model_params.cycle_residual = True

    if args.experiment_name is None:
        raise ValueError("Please specify an experiment name.")
    else:
        experiment_name = args.experiment_name

    config.model_params.num_layers = 4

    # n_hop_iter = [5, 4, 3, 2, 1, -1]
    n_hop_iter = [-1]

    n_runs = 10
    all_results_storage = []
    for n_hops in n_hop_iter:
        # stored_input, n_hops = settings
        if n_hops == -1:
            batch_size = 4
        else:
            batch_size = 50
        # for n_layers in n_layers_iter:
        for j in range(n_runs):
            # n_isomorphisms, n_layers, hidden_dim, dropout = settings
            cur_config = copy.deepcopy(config)
            cur_config.batch_size = batch_size
            cur_config.data.n_hops = n_hops
            cur_config.run_name = f"SRG_n_hops_{n_hops}_depth_2_hidden_{cur_config.model_params.hidden_dim}_n_isomorphisms_{cur_config.data.srg.n_isomorphisms}"

            start_time = time.time()
            val_loss, train_loss, test_loss, val_wrong, train_wrong, test_wrong = (
                run_hegnn(cur_config)
            )

            cur_config.model_params.pna_params.degrees = None
            result_dict = {
                "config": cur_config.to_dict(),
                "run": j,
                "results": {
                    "val_loss": float(val_loss),
                    "train_loss": float(train_loss),
                    "test_loss": float(test_loss),
                    "time": int(time.time() - start_time),
                    "val_wrong": val_wrong,
                    "train_wrong": train_wrong,
                    "test_wrong": test_wrong,
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
                val_wrong = result_dict["results"]["val_wrong"]
                train_wrong = result_dict["results"]["train_wrong"]
                test_wrong = result_dict["results"]["test_wrong"]
                runtime = result_dict["results"]["time"]
                print(
                    f"{run_name}: val_loss={val_loss}, train_loss={train_loss}, test_loss={test_loss}, time={int(runtime)}"
                )
                print(
                    f"val_wrong={val_wrong[0]}, train_wrong={train_wrong[0]}, test_wrong={test_wrong[0]}"
                )

            results_file = os.path.join(results_dir, experiment_name + ".json")
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            with open(results_file, "w") as f:
                json.dump(all_results_storage, f, indent=4)
            print(f"Results saved to {results_file}")
