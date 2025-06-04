import pdb
import time

import torch
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader

from hegnn.data_processing.data_preprocessing import square_batch
from hegnn.data_processing.data_structures import HierarchicalBatch
from hegnn.data_processing.k_hop_subgraphs import ego_networks_for_nodes


def value_index(tensor, value):
    return ((tensor == value).nonzero(as_tuple=True)[0]).squeeze()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = ZINC(
    root="../data/basic_ZINC", subset=True, split="train"
)  # Train split


def single_k_hop_from_batch():
    n_hops = 2
    miniset = train_dataset[:100]
    loader = DataLoader(miniset, batch_size=1, shuffle=True)
    for i, data in enumerate(loader):
        center_node = 10
        data = HierarchicalBatch(x=data.x, edge_index=data.edge_index, batch=data.batch)
        data.to(device)

        en_batch = ego_networks_for_nodes(data, en_centers=center_node, num_hops=n_hops)

        # Sanity checks
        assert len(en_batch) == 1

        # 2-hop subgraph of 2-hop subgraph is the same
        new_center_node = en_batch.unique_node_id[0]
        second_en_batch = ego_networks_for_nodes(
            en_batch, en_centers=new_center_node, num_hops=2
        )

        assert en_batch == second_en_batch

        # Number of nodes and edges is correct
        node_mask = data.x.new_empty(data.x.shape[0], dtype=torch.bool)
        node_mask.fill_(False)
        node_mask[center_node] = True
        for _ in range(n_hops):
            edges_to_keep = (
                node_mask[data.edge_index[0, :]] | node_mask[data.edge_index[1, :]]
            )
            node_mask[data.edge_index[0, edges_to_keep]] = True
            node_mask[data.edge_index[1, edges_to_keep]] = True

        assert len(node_mask.nonzero()) == len(en_batch.x), pdb.set_trace()

        edge_mask = node_mask[data.edge_index[0, :]] & node_mask[data.edge_index[1, :]]
        assert len(edge_mask.nonzero()) == len(en_batch.edge_index[0]), pdb.set_trace()

        data_nodes = torch.arange(data.x.shape[0], device=device)[
            node_mask.nonzero(as_tuple=True)[0]
        ]
        en_nodes = torch.arange(en_batch.x.shape[0], device=device)

        # Same edge structure
        for test_row, test_col, fun_row, fun_col in zip(
            data.edge_index[0, edge_mask],
            data.edge_index[1, edge_mask],
            en_batch.edge_index[0],
            en_batch.edge_index[1],
        ):
            assert value_index(data_nodes, test_row) == value_index(
                en_nodes, fun_row
            ), pdb.set_trace()
            assert value_index(data_nodes, test_col) == value_index(
                en_nodes, fun_col
            ), pdb.set_trace()

    print("SANITY CHECKS PASSED")


import random


def multiple_k_hop_vs_single():
    batch_size = 50
    mediumset = train_dataset[:1000]
    loader = DataLoader(mediumset, batch_size=batch_size, shuffle=False)
    n_hops = 2

    for i, data in enumerate(loader):
        data = HierarchicalBatch(x=data.x, edge_index=data.edge_index, batch=data.batch)
        data.to(device)

        en_centers = random.sample(range(0, data.x.shape[0]), 10)
        en_batch = ego_networks_for_nodes(data, en_centers=en_centers, num_hops=n_hops)

        for i, center_node in enumerate(en_centers):
            en_single = ego_networks_for_nodes(
                data, en_centers=center_node, num_hops=n_hops
            )
            assert en_single.extract_graph(0) == en_batch.extract_graph(
                i
            ), pdb.set_trace()

    print("Multiple k_hops matches single k_hop")


def time_k_hop():
    # Extract multiple subgraphs at once
    batch_size = 50
    data_size = 1000
    mediumset = train_dataset[:data_size]
    loader = DataLoader(mediumset, batch_size=batch_size, shuffle=False)
    n_hops = 2

    start_time = time.time()
    for data in loader:
        data = HierarchicalBatch(x=data.x, edge_index=data.edge_index, batch=data.batch)
        data.to(device)

        en_centers = list(range(0, data.x.shape[0]))
        _ = ego_networks_for_nodes(data, en_centers=en_centers, num_hops=n_hops)

    print(
        f"Time taken for {data_size} graphs, expended to {data_size*22} EGO NETWORKS: {time.time() - start_time:.2f} s"
    )

    start_time = time.time()
    for data in loader:
        data = HierarchicalBatch(x=data.x, edge_index=data.edge_index, batch=data.batch)
        data.to(device)

        _ = square_batch(data)

    print(
        f"Time taken for {data_size} graphs, expended to {data_size*22} graphs: {time.time() - start_time:.2f} s"
    )


single_k_hop_from_batch()
multiple_k_hop_vs_single()
time_k_hop()
