import os
import pdb
import time

import h5py
import numpy as np
import torch
from torch.optim import Adam
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import PNAConv

from hegnn.data_processing.cycle_counts import AddCycleCountsTransform
from hegnn.data_processing.data_preprocessing import square_batch
from hegnn.data_processing.data_structures import HierarchicalBatch
from hegnn.data_processing.hdf5_structures import HDF5Dataset
from hegnn.data_processing.k_hop_subgraphs import ego_network_batch
from hegnn.globals import custom_data_dir, data_dir, srg_data_dir
from hegnn.srg.sr import CustomSRGDataset


def create_folder(folder):
    """
    Create a folder if it doesn't exist.
    """
    if not os.path.exists(folder):
        print(f"Creating folder: {folder}")
        os.makedirs(folder, exist_ok=True)


def batch_to_dict(batch):
    """
    Convert a HierarchicalBatch to a dictionary of tensors, handling None values.
    """
    batch = batch.to(torch.device("cpu"))  # Move to CPU if necessary
    data_dict = {
        "x": batch.x.numpy(),
        "edge_index": batch.edge_index.numpy(),
        "edge_attr": None if batch.edge_attr is None else batch.edge_attr.numpy(),
        "y": None if batch.y is None else batch.y.numpy(),
        "batch": batch.batch.numpy(),
        "unique_node_id": (
            None if batch.unique_node_id is None else batch.unique_node_id.numpy()
        ),
        "next_depth_graph_id": (
            None
            if batch.next_depth_graph_id is None
            else batch.next_depth_graph_id.numpy()
        ),
    }
    return data_dict


def store_data_group(f, group_index, data_dict):
    """
    Store a group of data in the HDF5 file, handling None values properly.
    """
    n_graphs = np.unique(data_dict["batch"]).size

    group = f.create_group(f"group_{group_index}")
    for key, value in data_dict.items():
        if value is None:
            group.create_dataset(key, shape=(0,), dtype=np.float64)  # Empty dataset
            group.attrs[f"{key}_none"] = True  # Mark as None
        else:
            group.create_dataset(key, data=value)
            group.attrs[f"{key}_none"] = False  # Mark as valid
    group.attrs["n_graphs"] = n_graphs


def store_data_per_depth(
    dataset, folder, storage_size, total_depth, n_hops, overwrite=False
):
    # TODO: Maybe store a metadata file with:
    # - storage size
    # - total depth
    # - n hops
    # - dataset length

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    create_folder(folder)

    for d in range(total_depth + 1):
        start_time = time.time()
        file = f"{folder}/depth_{d}.h5"
        offset = 0

        if os.path.exists(file):
            if overwrite:
                print(f"File {file} already exists. Overwriting...")
            else:
                print(f"File {file} already exists. Skipping...")
                continue
        else:
            print(f"Creating file: {file}")

        with h5py.File(file, "w") as f:
            if d == 0:
                loader = DataLoader(
                    dataset,
                    batch_size=storage_size,
                    drop_last=False,
                    shuffle=False,
                )
                for group_index, data in enumerate(loader):
                    n_group_nodes = data.x.shape[0]

                    data = HierarchicalBatch(
                        x=data.x,
                        edge_index=data.edge_index,
                        edge_attr=data.edge_attr,
                        batch=data.batch,
                        y=data.y,
                        next_depth_graph_id=torch.arange(
                            offset, offset + n_group_nodes, dtype=torch.long
                        ),
                    )

                    data_dict = batch_to_dict(data)
                    offset += n_group_nodes
                    store_data_group(f, group_index, data_dict)

            if d > 0:
                prev_depth_dataset = HDF5Dataset(
                    f"{folder}/depth_{d-1}.h5", device=device
                )
                group_index = 0
                remainder = None

                for prev_depth_batch in prev_depth_dataset:
                    prev_depth_batch = prev_depth_batch.to(device)

                    if n_hops == -1:
                        expanded_batch = square_batch(prev_depth_batch)
                    else:
                        expanded_batch = ego_network_batch(prev_depth_batch, n_hops)

                    expanded_batch = expanded_batch.add_prefix(remainder)

                    n_groups = expanded_batch.batch_size // storage_size
                    if expanded_batch.batch_size % storage_size == 0:
                        remainder = None
                    else:
                        remainder = expanded_batch.extract_subbatch_between(
                            n_groups * storage_size, expanded_batch.batch_size - 1
                        )

                    for batch_group_index in range(n_groups):
                        # Standard case
                        data = expanded_batch.extract_subbatch_between(
                            batch_group_index * storage_size,
                            (batch_group_index + 1) * storage_size - 1,
                        )
                        next_depth_graph_id = torch.arange(
                            offset, offset + data.x.shape[0], dtype=torch.long
                        )
                        data = HierarchicalBatch.add_next_depth_graph_id(
                            data, next_depth_graph_id
                        )
                        offset += data.x.shape[0]

                        data_dict = batch_to_dict(data)
                        if (
                            "next_depth_graph_id" not in data_dict
                            or data_dict["next_depth_graph_id"] is None
                        ):
                            pdb.set_trace()

                        store_data_group(f, group_index, data_dict)
                        group_index += 1

                if remainder is not None:
                    # Handle remainder
                    next_depth_graph_id = torch.arange(
                        offset, offset + remainder.x.shape[0], dtype=torch.long
                    )
                    remainder = HierarchicalBatch.add_next_depth_graph_id(
                        remainder, next_depth_graph_id
                    )
                    data_dict = batch_to_dict(remainder)
                    if (
                        "next_depth_graph_id" not in data_dict
                        or data_dict["next_depth_graph_id"] is None
                    ):
                        pdb.set_trace()

                    store_data_group(f, group_index, data_dict)

            print(f"Stored depth {d} in {file}. Time: {time.time() - start_time:.2f}")


def check_data_config(config):
    depth, model = config.model_params.depth, config.model

    if model in ["HEGNN", "SingleNetHEGNN"]:
        assert depth >= 0
    elif model == "GNN":
        assert depth == 0, "Depth must be 0 for GNN"
    else:
        raise ValueError(f"Unknown model: {model}")

    assert depth is not None, "Depth must be specified"


def simple_srg_data(config):
    if config.cycles is None:
        dataset = CustomSRGDataset(
            root=srg_data_dir,
            name=config.data.srg.name,
            n_isomorphisms=config.data.srg.n_isomorphisms,
            train_fraction=0.8,
            val_fraction=0.1,
            test_fraction=0.1,
        )
    else:
        for i in config.cycles:
            cycle_string += f"{i}"
        dataset = CustomSRGDataset(
            root=srg_data_dir,
            name=config.data.srg.name,
            n_isomorphisms=config.data.srg.n_isomorphisms,
            train_fraction=0.8,
            val_fraction=0.1,
            test_fraction=0.1,
            transform=AddCycleCountsTransform(config.cycles),
        )

    train_dataset, val_dataset, test_dataset = dataset.split()
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False
    )
    dataloader_per_split = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

    degrees = PNAConv.get_degree_histogram(train_loader)
    config.model_params.pna_params.degrees = degrees
    config.data.n_classes = dataset.num_classes
    config.data.task = "classification"

    loader_per_split = {"train": train_loader, "val": val_loader, "test": test_loader}

    return config, loader_per_split


def stored_srg_data(config):
    check_data_config(config)
    data_config = config.data
    storage_size, n_hops = data_config.storage_size, data_config.n_hops
    depth = config.model_params.depth

    if config.cycles is None:
        dataset = CustomSRGDataset(
            root=srg_data_dir,
            name=config.data.srg.name,
            n_isomorphisms=config.data.srg.n_isomorphisms,
            train_fraction=0.8,
            val_fraction=0.1,
            test_fraction=0.1,
        )
        train_data, val_data, test_data = dataset.split()
    else:
        cycle_string = ""
        for i in config.cycles:
            cycle_string += f"{i}"
        dataset = CustomSRGDataset(
            root=srg_data_dir,
            name=config.data.srg.name,
            n_isomorphisms=config.data.srg.n_isomorphisms,
            train_fraction=0.8,
            val_fraction=0.1,
            test_fraction=0.1,
            transform=AddCycleCountsTransform(config.cycles),
        )
        train_data, val_data, test_data = dataset.split()
    config.data.n_classes = dataset.num_classes
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        drop_last=False,
        shuffle=False,
    )
    degrees = PNAConv.get_degree_histogram(train_loader)

    folders = {"train": None, "val": None, "test": None}
    for split, dataset in zip(
        ["train", "val", "test"], [train_data, val_data, test_data]
    ):
        if config.cycles is None:
            folder = f"{custom_data_dir}/{config.data.srg.name}/n_hops_{n_hops}/storage_size_{storage_size}/{split}"
        else:
            folder = f"{custom_data_dir}/{config.data.srg.name}_{cycle_string}/n_hops_{n_hops}/storage_size_{storage_size}/{split}"

        store_data_per_depth(
            dataset,
            folder,
            storage_size,
            depth,
            n_hops,
            overwrite=config.data.overwrite,
        )
        folders[split] = folder

    files = {k: get_data_file(v, 0) for k, v in folders.items()}
    config.model_params.pna_params.degrees = degrees
    config.data.task = "classification"
    return config, files


def load_data(config):
    if config.data.stored_input:
        dataloader_per_split = {"train": None, "val": None, "test": None}
        if config.data.dataset == "srg":
            config, file_per_split = stored_srg_data(config)
        elif config.data.dataset == "zinc":
            config, file_per_split = stored_zinc_data(config)
        else:
            raise ValueError(f"Unknown dataset: {config.data.dataset}")
    else:
        file_per_split = {"train": None, "val": None, "test": None}
        if config.data.dataset == "srg":
            config, dataloader_per_split = simple_srg_data(config)
        elif config.data.dataset == "zinc":
            config, dataloader_per_split = simple_zinc_data(config)
        else:
            raise ValueError(f"Unknown dataset: {config.data.dataset}")

    return config, dataloader_per_split, file_per_split


def stored_zinc_data(config):
    """
    Load the ZINC dataset and store data per depth.
    """
    check_data_config(config)
    data_config = config.data
    data_size, storage_size, n_hops, overwrite = (
        data_config.data_size,
        data_config.storage_size,
        data_config.n_hops,
        data_config.overwrite,
    )
    depth = config.model_params.depth

    if data_size in ["smallsubset", "subset"]:
        subset = True
    elif data_size == "full":
        subset = False
    else:
        raise ValueError("Invalid size. Choose 'smallsubset', 'subset', or 'full'.")

    if config.cycles is None:
        train_data = ZINC(root=f"{data_dir}/basic_ZINC", subset=subset, split="train")
        val_data = ZINC(root=f"{data_dir}/basic_ZINC", subset=subset, split="val")
        test_data = ZINC(root=f"{data_dir}/basic_ZINC", subset=subset, split="test")
    else:
        cycle_string = ""
        for i in config.cycles:
            cycle_string += f"{i}"
        train_data = ZINC(
            root=f"{data_dir}/basic_ZINC" + cycle_string,
            subset=subset,
            split="train",
            transform=AddCycleCountsTransform(config.cycles),
        )
        val_data = ZINC(
            root=f"{data_dir}/basic_ZINC" + cycle_string,
            subset=subset,
            split="val",
            transform=AddCycleCountsTransform(config.cycles),
        )
        test_data = ZINC(
            root=f"{data_dir}/basic_ZINC" + cycle_string,
            subset=subset,
            split="test",
            transform=AddCycleCountsTransform(config.cycles),
        )

    if data_size == "smallsubset":
        train_data = train_data[:5000]

    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        drop_last=False,
        shuffle=False,
    )
    degrees = PNAConv.get_degree_histogram(train_loader)

    folders = {"train": None, "val": None, "test": None}
    for split, dataset in zip(
        ["train", "val", "test"], [train_data, val_data, test_data]
    ):

        if config.cycles is None:
            folder = f"{custom_data_dir}/zinc_{data_size}/n_hops_{n_hops}/storage_size_{storage_size}/{split}"
        else:
            folder = f"{custom_data_dir}/zinc_{data_size}_{cycle_string}/n_hops_{n_hops}/storage_size_{storage_size}/{split}"

        store_data_per_depth(dataset, folder, storage_size, depth, n_hops, overwrite)
        folders[split] = folder

    files = {k: get_data_file(v, 0) for k, v in folders.items()}

    config.model_params.pna_params.degrees = degrees
    config.data.task = "regression"
    return config, files


def simple_zinc_data(config):
    """
    Load the ZINC dataset and store data per depth.
    """
    check_data_config(config)
    data_config = config.data
    data_size, storage_size, n_hops, overwrite = (
        data_config.data_size,
        data_config.storage_size,
        data_config.n_hops,
        data_config.overwrite,
    )
    depth = config.model_params.depth

    if data_size in ["smallsubset", "subset"]:
        subset = True
    elif data_size == "full":
        subset = False
    else:
        raise ValueError("Invalid size. Choose 'smallsubset', 'subset', or 'full'.")

    if config.cycles is None:
        train_data = ZINC(root=f"{data_dir}/basic_ZINC", subset=subset, split="train")
        val_data = ZINC(root=f"{data_dir}/basic_ZINC", subset=subset, split="val")
        test_data = ZINC(root=f"{data_dir}/basic_ZINC", subset=subset, split="test")
    else:
        raise NotImplementedError("Cycle counts not implemented for this configuration")

    if data_size == "smallsubset":
        train_data = train_data[:5000]

    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        drop_last=False,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        drop_last=False,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=config.batch_size,
        drop_last=False,
        shuffle=False,
    )

    degrees = PNAConv.get_degree_histogram(train_loader)
    config.model_params.pna_params.degrees = degrees

    loader_per_split = {"train": train_loader, "val": val_loader, "test": test_loader}
    config.data.task = "regression"
    return config, loader_per_split


def get_data_file(folder, depth):
    """
    Get the file path for a specific depth.
    """
    return f"{folder}/depth_{depth}.h5"


def next_depth_file(file):
    """
    Get the next depth file path.
    """
    if file is None:
        return None

    folder = os.path.dirname(file)
    depth = int(os.path.basename(file).split("_")[1].split(".")[0])
    return f"{folder}/depth_{depth+1}.h5"
