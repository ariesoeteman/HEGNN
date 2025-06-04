import pdb
import time

import numpy as np
import torch
import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from hegnn.data_processing.data_structures import HierarchicalBatch
from hegnn.data_processing.hdf5_structures import (
    CustomLoader,
    HDF5Dataset,
    HierarchicalBatch,
)
from hegnn.data_processing.store_data_per_depth import load_data
from hegnn.data_processing.utils import ZincEncoder
from hegnn.models.HEGNN import HEGNN
from hegnn.utils.early_stopping import EarlyStopping
from hegnn.utils.memory_trackers import print_memory

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train(model, config, data_file, data_loader, optimizer, criterion):
    assert (
        data_loader is None or data_file is None
    ), "Either data_loader or data_file should be provided, not both."
    if data_loader is None:
        dataset = HDF5Dataset(data_file, device=device)
        data_loader = CustomLoader(dataset, batch_size=config.batch_size, shuffle=False)

    model.train()
    total_loss = 0
    data_size = 0
    num_wrong = 0
    for data in data_loader:
        if not isinstance(data, HierarchicalBatch):
            data = HierarchicalBatch.from_batch(data)
        data.to(device)
        data_size += data.batch_size

        optimizer.zero_grad()
        out = model(data, split="train")

        if config.data.dataset in ["csl", "srg"]:
            y = data.y
            total_loss += criterion(out, y)
            pred = out.argmax(dim=1)  # predicted class per graph
            num_wrong += (pred != y).sum().item()

        elif config.data.dataset == "zinc":
            y = data.y.view(-1, 1)
            total_loss += np.sum(
                np.abs(y.cpu().detach().numpy() - out.cpu().detach().numpy())
            )
            num_wrong += 0

        loss = criterion(out, y)  # Calculate loss
        loss.backward()  # Backpropagation

        optimizer.step()

    return total_loss / data_size, (f"{num_wrong}/{data_size}", num_wrong / data_size)


def evaluate(model, config, data_file, data_loader, criterion):
    assert (
        data_loader is None or data_file is None
    ), "Either data_loader or data_file should be provided, not both."

    if data_loader is None:
        dataset = HDF5Dataset(data_file, device=device)  # original data (depth 0)
        data_loader = CustomLoader(dataset, batch_size=config.batch_size)

    model.eval()
    total_loss = 0
    data_size = 0
    num_wrong = 0
    with torch.no_grad():
        for data in data_loader:
            if not isinstance(data, HierarchicalBatch):
                data = HierarchicalBatch.from_batch(data)
            data.to(device)
            data_size += data.batch_size
            out = model(data, split="val")

            if config.data.dataset in ["csl", "srg"]:
                y = data.y
                out = torch.clamp(out, min=-10, max=10)
                total_loss += criterion(out, y)
                pred = out.argmax(dim=1)  # predicted class per graph
                num_wrong += (pred != y).sum().item()

            elif config.data.dataset == "zinc":
                y = data.y.view(-1, 1)
                total_loss += np.sum(
                    np.abs(y.cpu().detach().numpy() - out.cpu().detach().numpy())
                )
                num_wrong += 0

    return total_loss / data_size, (f"{num_wrong}/{data_size}", num_wrong / data_size)


def test(model, config, data_file, data_loader, criterion):
    assert (
        data_loader is None or data_file is None
    ), "Either data_loader or data_file should be provided, not both."
    if data_loader is None:
        dataset = HDF5Dataset(data_file, device=device)  # original data (depth 0)
        data_loader = CustomLoader(dataset, batch_size=config.batch_size)

    model.eval()
    total_loss = 0
    data_size = 0
    num_wrong = 0
    with torch.no_grad():
        for data in data_loader:
            if not isinstance(data, HierarchicalBatch):
                data = HierarchicalBatch.from_batch(data)
            data.to(device)
            data_size += data.batch_size
            out = model(data, split="test")
            if config.data.dataset in ["csl", "srg"]:
                y = data.y
                out = torch.clamp(out, min=-10, max=10)
                total_loss += criterion(out, y)
                pred = out.argmax(dim=1)  # predicted class per graph
                num_wrong += (pred != y).sum().item()

            elif config.data.dataset == "zinc":
                y = data.y.view(-1, 1)
                total_loss += np.sum(
                    np.abs(y.cpu().detach().numpy() - out.cpu().detach().numpy())
                )
                num_wrong += 0
    return total_loss / data_size, (f"{num_wrong}/{data_size}", num_wrong / data_size)


def run_hegnn(config):
    config, dataloader_per_split, file_per_split = load_data(config)
    depth = config.model_params.depth
    if config.data.dataset == "zinc":
        feature_encoder = ZincEncoder(hidden_dim=config.model_params.hidden_dim)
    elif config.data.dataset in ["csl", "srg"]:
        feature_encoder = lambda x: x

    if config.model == "HEGNN":
        if config.data.dataset == "zinc":
            graph_output = True
            output_dim = 1
            if config.cycles is None:
                input_dim = config.model_params.hidden_dim
            else:
                input_dim = config.model_params.hidden_dim + len(config.cycles)
        elif config.data.dataset in ["csl", "srg"]:
            graph_output = True
            output_dim = None
            if config.cycles is None:
                input_dim = 1
            else:
                input_dim = 1 + len(config.cycles)
                input_dim = 1 + len(config.cycles)
        model = HEGNN(
            input_dim=input_dim,
            output_dim=output_dim,
            depth=depth,
            file_per_split=file_per_split,
            graph_output=graph_output,
            params=config,
            feature_encoder=feature_encoder,
        )  # Predicting a single value (regression task)
    else:
        raise NotImplementedError(f"Model {config.model} not implemented")

    print_log = config.print_log
    early_stopper = EarlyStopping(patience=100)
    wandb_mode = config.wandb
    if wandb_mode:
        wandb.init(
            project="hegnn",
            entity="awsoeteman",
            name=config.get("run_name", None),
            config=config,
        )

    model.to(device)

    if config.data.dataset in ["csl", "srg"]:
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    elif config.data.dataset == "zinc":
        criterion = torch.nn.L1Loss()

    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=config.weight_decay)

    factor, patience = config.learning_rate.factor, config.learning_rate.patience
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=factor, patience=patience
    )

    start_time = time.time()
    num_epochs = config.n_epochs
    current_epoch = 1

    best_val_loss = float("inf")
    final_test_loss = float("inf")
    final_train_loss = float("inf")
    final_train_wrong = ""
    final_val_wrong = ""
    final_test_wrong = ""
    while True:
        for epoch in range(current_epoch, num_epochs + 1):
            try:
                train_loss, num_wrong_train = train(
                    model,
                    config=config,
                    data_file=file_per_split["train"],
                    data_loader=dataloader_per_split["train"],
                    optimizer=optimizer,
                    criterion=criterion,
                )
                val_loss, num_wrong_val = evaluate(
                    model,
                    config=config,
                    data_file=file_per_split["val"],
                    data_loader=dataloader_per_split["val"],
                    criterion=criterion,
                )
                test_loss, num_wrong_test = test(
                    model,
                    config=config,
                    data_file=file_per_split["test"],
                    data_loader=dataloader_per_split["test"],
                    criterion=criterion,
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    final_test_loss = test_loss
                    final_train_loss = train_loss
                    final_train_wrong = num_wrong_train
                    final_val_wrong = num_wrong_val
                    final_test_wrong = num_wrong_test

                scheduler.step(val_loss)

                if print_log:
                    print(
                        f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f} time: {time.time() - start_time}"
                    )
                    if config.data.dataset in ["csl", "srg"]:
                        print(
                            f"Train wrong: {num_wrong_train[0]}, Val wrong: {num_wrong_val[0]}, Test wrong: {num_wrong_test[0]}"
                        )
                if wandb_mode:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "time_elapsed": time.time() - start_time,
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                    )

                from hegnn.models.HEGNN import TOTAL_DATA_TIME

                if print_log:
                    print(f"Total data time {TOTAL_DATA_TIME}")
                    print_memory()

                early_stopper(val_loss)
                if early_stopper.early_stop:
                    print(
                        f"Early stopping at epoch {epoch}, val loss: {val_loss:.4f}, test loss: {test_loss:.4f}"
                    )
                    break

                current_epoch += 1

            except KeyboardInterrupt:
                print("Training interrupted, cleaning up...")
                torch.cuda.empty_cache()
                exit()

        if __name__ == "__main__":
            user_input = input("Continue? (y/n): ").strip().lower()
            if user_input == "n":
                break

            user_input = input("how many epochs?").strip().lower()
            num_epochs += max(20, int(user_input))
        else:
            # If not in interactive mode, just break after training
            break

        if print_log:
            print(f"Training took {time.time() - start_time} seconds")

    if wandb_mode:
        wandb.finish()
    return (
        best_val_loss,
        final_train_loss,
        final_test_loss,
        final_val_wrong,
        final_train_wrong,
        final_test_wrong,
    )
