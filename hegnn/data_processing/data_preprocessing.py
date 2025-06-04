import pdb
import time

import numpy as np
import torch
from torch_geometric.data import Batch

from .data_structures import HierarchicalBatch


def square_x(x, batch_sizes, device):
    starts = torch.cumsum(
        torch.cat([torch.tensor([0], device=device), batch_sizes[:-1]]), dim=0
    )
    ends = torch.cumsum(batch_sizes, dim=0)
    squared_x = torch.cat(
        [
            x[start:end].repeat(batch_size, 1)
            for start, end, batch_size in zip(starts, ends, batch_sizes)
        ]
    )
    return squared_x


def square_edge_index(batch, batch_sizes, device):
    edge_sizes = batch.get_edge_counts()

    squared_batch_sizes = torch.square(batch_sizes)
    node_offsets = squared_batch_sizes - batch_sizes

    node_offset_per_batch = torch.cumsum(
        torch.cat([torch.tensor([0], device=device), node_offsets[:-1]]), dim=0
    )

    # Working approach, more memory efficient and slightly faster
    edge_index_tuple = torch.split(batch.edge_index, list(edge_sizes), dim=1)

    extended_edge_index = torch.cat(
        [
            edge_index.repeat(1, batch_size)
            + torch.arange(batch_size, device=device).repeat_interleave(edge_size)
            * batch_size
            + node_offset
            for edge_index, edge_size, batch_size, node_offset in zip(
                edge_index_tuple, edge_sizes, batch_sizes, node_offset_per_batch
            )
        ],
        dim=1,
    )

    if batch.edge_attr is None:
        extended_edge_attr = None
    else:
        edge_attr_tuple = torch.split(batch.edge_attr, list(edge_sizes), dim=0)
        extended_edge_attr = torch.cat(
            [
                edge_attr.repeat(batch_size)
                for edge_attr, batch_size in zip(edge_attr_tuple, batch_sizes)
            ]
        )

    return extended_edge_index, extended_edge_attr


def square_batch(batch: HierarchicalBatch) -> HierarchicalBatch:
    device = batch.x.device
    batch_sizes = batch.get_batch_sizes()

    squared_x = square_x(batch.x, batch_sizes, device)
    squared_edge_index, squared_edge_attr = square_edge_index(
        batch, batch_sizes, device
    )

    size_per_squared_x = batch_sizes.repeat_interleave(batch_sizes)
    squared_batch_index = torch.arange(
        size_per_squared_x.shape[0], device=device
    ).repeat_interleave(size_per_squared_x)

    ### Compute en_center ids, and add them to the squared_x
    relative_unique_node_id = torch.cat(
        [torch.arange(N, device=device) for N in batch_sizes]
    )
    node_offset = torch.cumsum(
        torch.cat([torch.tensor([0], device=device), size_per_squared_x]), dim=0
    )[:-1]

    absolute_uids = relative_unique_node_id + node_offset
    absolute_uids = absolute_uids.unsqueeze(1)

    ### Add identifiers to x
    total_nodes = size_per_squared_x.sum()

    uid_features = torch.zeros(total_nodes, 1, device=device)
    ones = torch.ones_like(absolute_uids, dtype=uid_features.dtype)
    uid_features.scatter_(0, absolute_uids, ones)

    squared_x = torch.cat([squared_x, uid_features], dim=1)

    output_batch = HierarchicalBatch(
        x=squared_x,
        y=None,
        edge_index=squared_edge_index,
        edge_attr=squared_edge_attr,
        batch=squared_batch_index,
        unique_node_id=absolute_uids,
    )
    return output_batch


def power_batch(batch, power):
    assert isinstance(batch, Batch)
    assert power >= 0

    batch = HierarchicalBatch(
        x=batch.x,
        edge_index=batch.edge_index,
        edge_attr=batch.edge_attr,
        batch=batch.batch,
        y=batch.y,
    )

    batch_per_depth = [None for _ in range(power + 1)]
    batch_per_depth[0] = batch

    if power == 0:
        return batch_per_depth

    cur_batch = batch
    for i in range(1, power + 1):
        cur_batch = square_batch(cur_batch)
        batch_per_depth[i] = cur_batch

    batch_per_depth.reverse()

    return batch_per_depth


def add_zero_columns(x, n):
    return torch.cat([x, torch.zeros(x.shape[0], n, device=x.device)], dim=1)
