import pdb
import time
from typing import List, Union

import torch
from torch import Tensor

from hegnn.data_processing.data_structures import HierarchicalBatch


def ego_network_batch(batch: HierarchicalBatch, num_hops: int) -> HierarchicalBatch:
    assert num_hops >= 0, "num_hops should be a non-negative integer."
    en_centers = list(range(0, batch.x.shape[0]))
    return ego_networks_for_nodes(batch, en_centers, num_hops)


def ego_networks_for_nodes(
    batch: HierarchicalBatch,
    en_centers: Union[int, List[int], Tensor],
    num_hops: int,
    relabel_nodes: bool = True,
) -> HierarchicalBatch:

    x, edge_index = batch.x, batch.edge_index
    row, col = edge_index

    # We try to generate a distinct batch for each k-hop subgraph
    if isinstance(en_centers, (int, list, tuple)):
        en_centers = torch.tensor([en_centers], device=row.device).flatten()
    else:
        assert en_centers.dim() == 1, "en_centers should be a 1D tensor."
        en_centers = en_centers.to(row.device)

    num_nodes = x.shape[0]
    num_subgraphs = en_centers.shape[0]

    node_mask = row.new_empty(num_nodes, dtype=torch.bool, device=row.device)
    batch_masks = torch.zeros(
        num_subgraphs, num_nodes, dtype=torch.bool, device=row.device
    )

    for i in range(num_subgraphs):
        batch_masks[i, en_centers[i]] = True

    node_mask[en_centers] = True
    subsets = [en_centers]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True

        # All edges with 'row' in node_mask
        edge_mask = node_mask[row]

        # Next hop, contains duplicates if two kept edges reach the same node
        source, targets = edge_index[:, edge_mask]
        subsets.append(targets)

        for i in range(num_subgraphs):
            batch_subset = targets[batch_masks[i][source]]
            batch_masks[i, batch_subset] = True

    # Nodes for each batch concatenated
    x_list = []
    edge_index_list = []
    if batch.edge_attr is None:
        edge_attr = None
    else:
        edge_attr_list = []
    batch_list = []
    unique_node_id_list = []

    init_time = 0
    relabel_time = 0
    append_time = 0

    offset = 0
    for i, en_center in enumerate(en_centers):
        batch_x = x[batch_masks[i]]
        batch_size = batch_x.size(0)
        batch_batch = row.new_full((batch_size,), i)

        batch_edge_mask = batch_masks[i][row] & batch_masks[i][col]
        batch_edge_index = edge_index[:, batch_edge_mask]

        if batch.edge_attr is not None:
            batch_edge_attr = batch.edge_attr[batch_edge_mask]
        unique_node_id = en_centers[i]
        old_to_new_node_indices = row.new_full((num_nodes,), -1)

        # Indices of all the old nodes in the batch
        batch_subset = batch_masks[i].nonzero(as_tuple=False).flatten()
        new_node_indices = torch.arange(offset, offset + batch_size, device=row.device)

        # Now all the old nodes in my batch subset are mapped to the new indices
        old_to_new_node_indices[batch_subset] = new_node_indices

        # Now no edge index should have -1 anymore
        batch_edge_index = old_to_new_node_indices[batch_edge_index]

        # Add unique id features
        unique_node_id = old_to_new_node_indices[unique_node_id]

        new_en_center = old_to_new_node_indices[en_center] - offset
        unique_id_features = torch.zeros(
            (batch_size,), device=row.device, dtype=batch_x.dtype
        )
        unique_id_features[new_en_center] = 1
        batch_x = torch.cat([batch_x, unique_id_features.unsqueeze(1)], dim=1)

        relabel_time += time.time() - start
        start = time.time()

        offset += batch_size

        x_list.append(batch_x)
        edge_index_list.append(batch_edge_index)

        if batch.edge_attr is not None:
            edge_attr_list.append(batch_edge_attr)
        batch_list.append(batch_batch)
        unique_node_id_list.append(torch.tensor([unique_node_id], device=row.device))

        append_time += time.time() - start

    if batch.edge_attr is None:
        edge_attr = None
    else:
        edge_attr = torch.cat(edge_attr_list, dim=0)
    subgraph_batch = HierarchicalBatch(
        x=torch.cat(x_list, dim=0),
        y=None,
        edge_index=torch.cat(edge_index_list, dim=1),
        edge_attr=edge_attr,
        batch=torch.cat(batch_list, dim=0),
        unique_node_id=torch.cat(unique_node_id_list, dim=0).unsqueeze(1),
        next_depth_graph_id=None,
    )
    return subgraph_batch
