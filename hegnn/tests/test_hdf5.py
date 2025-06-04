import pdb

import torch
from torch_geometric.datasets import ZINC

from hegnn.config.parse_config import load_config
from hegnn.data_processing.data_structures import HierarchicalBatch, merge_batch_list
from hegnn.data_processing.hdf5_structures import HDF5Dataset
from hegnn.data_processing.store_data_per_depth import (
    get_data_file,
    store_data_per_depth,
)
from hegnn.globals import custom_data_dir, data_dir


def test_HDF5_storage():
    train_dataset = ZINC(root=data_dir + "/basic_ZINC", subset=True, split="train")
    train_dataset = train_dataset[:2010]
    storage_size = 100
    total_depth = 1
    n_hops = -1

    # Store data per depth
    folder = f"{data_dir}/testing/zinc_smallsubset/n_hops_{n_hops}/storage_size_{storage_size}/train"
    store_data_per_depth(
        train_dataset, folder, storage_size, total_depth, n_hops, overwrite=True
    )

    file_nul = f"{folder}/depth_0.h5"
    hdf5_dataset = HDF5Dataset(file_nul, device="cpu")

    # Extracting individual graphs
    offset = 0
    batches = []
    for i, data in enumerate(train_dataset):
        batch = HierarchicalBatch(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=torch.zeros(
                data.x.shape[0], device=data.x.device
            ),  # this construction doesn't add batch
            y=data.y,
            next_depth_graph_id=torch.arange(
                offset, offset + data.x.shape[0], dtype=torch.int
            ),
            unique_node_id=None,
        )
        offset += data.x.shape[0]
        batches.append(batch)

        # Check if the batch is equal to the one in the HDF5 file
        hdf5_batch = hdf5_dataset.get_graph(i)

        try:
            assert batch == hdf5_batch, pdb.set_trace()
        except:
            print("ASSERT FAILED")
            pdb.set_trace()

    print("SINGLE GRAPH TEST PASSED")

    # Extracting subbatches
    start_end = (100, 150)
    merged_batch = merge_batch_list(batches[start_end[0] : start_end[1]])
    hdf5_merged_batch = hdf5_dataset.get_graphs_between(start_end[0], start_end[1] - 1)

    assert merged_batch == hdf5_merged_batch, pdb.set_trace()

    print("SUBBATCH QUERY TEST PASSES")


def test_HDF5_depth_one_size():
    train_dataset = ZINC(root=data_dir + "/basic_ZINC", subset=True, split="train")
    train_dataset = train_dataset[:200]

    # GRAPH/batch index in depth d is subset index in depth d+1
    storage_size = 100
    total_depth = 1
    n_hops = -1

    # Store data per depth
    folder = f"{data_dir}/testing/zinc_smallsubset/n_hops_{n_hops}/storage_size_{storage_size}/train"
    store_data_per_depth(
        train_dataset, folder, storage_size, total_depth, n_hops, overwrite=True
    )

    file_one = f"{folder}/depth_1.h5"
    hdf5_dataset_one = HDF5Dataset(file_one)

    assert train_dataset.x.shape[0] == hdf5_dataset_one.n_graphs, pdb.set_trace()

    print("Depth 1 TEST PASSED")


def test_next_depth_graph_id():
    train_dataset = ZINC(root=data_dir + "/basic_ZINC", subset=True, split="train")
    train_dataset = train_dataset[:200]

    # GRAPH/batch index in depth d is subset index in depth d+1
    storage_size = 100
    n_hops = -1

    config = load_config()
    config.model_params.depth = 2
    config.data.data_size = "smallsubset"
    config.n_epochs = 1
    config.data.stored_input = True
    config.data.n_hops = 2
    config.data.overwrite = True

    data_size = "smallsubset"
    depth = 2
    overwrite = True

    train_data = ZINC(root=f"{data_dir}/basic_ZINC", subset=True, split="train")
    train_data = train_data[:200]

    folder = f"{custom_data_dir}_test/zinc_{data_size}/n_hops_{n_hops}/storage_size_{storage_size}/train"
    store_data_per_depth(train_data, folder, storage_size, depth, n_hops, overwrite)

    depth_nul_file = get_data_file(folder, 0)
    depth_one_file = get_data_file(folder, 1)
    depth_two_file = get_data_file(folder, 2)

    depth_nul_data = HDF5Dataset(depth_nul_file)
    depth_one_data = HDF5Dataset(depth_one_file)
    depth_two_data = HDF5Dataset(depth_two_file)

    depth_one_batch = depth_one_data.get_graphs_between(0, 10)
    depth_one_batch_two = depth_one_data.get_graphs_between(11, 20)

    graph_id = torch.cat(
        [depth_one_batch.next_depth_graph_id, depth_one_batch_two.next_depth_graph_id]
    )

    assert torch.equal(
        graph_id, torch.arange(len(graph_id), device=graph_id.device)
    ), "graph_id is not a tensor [0, 1, ..., len(graph_id)-1]"

    print("NEXT_DEPTH_GRAPH_ID TEST PASSED")


if __name__ == "__main__":
    test_next_depth_graph_id()
    test_HDF5_storage()
    test_HDF5_depth_one_size()
    print("All tests passed.")
