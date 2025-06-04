import pdb

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj


class AddCycleCountsTransform:
    def __init__(self, cycles=[4]):
        self.cycles = cycles

    def __call__(self, data: Data):
        cycle_counts = count_cycles(data.edge_index, self.cycles)
        try:
            if len(self.cycles) == 1:
                cycle_features = cycle_counts[self.cycles[0]].unsqueeze(1)
            else:

                cycle_features = torch.cat(
                    [cycle_counts[l].unsqueeze(1) for l in self.cycles], dim=1
                )
        except:
            print(cycle_counts)
            print(self.cycles)
            pdb.set_trace()
        data.x = torch.cat([data.x, cycle_features], dim=1)

        return data


def count_cycles(edge_index, cycles):
    # Convert edge_index to an adjacency matrix
    adj = to_dense_adj(edge_index)[0]  # Shape: (num_nodes, num_nodes)
    power_adj = adj.clone()

    power_adj
    cycle_counts = {}

    max_cycle_length = max(cycles)
    for cycle_length in range(2, max_cycle_length + 1):
        # Compute the adjacency matrix raised to the power of cycle_length
        power_adj = torch.matmul(power_adj, adj)
        cycle_count = torch.diagonal(power_adj)
        cycle_counts[cycle_length] = cycle_count

    return cycle_counts
