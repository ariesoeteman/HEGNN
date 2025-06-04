import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool

from hegnn.data_processing.data_preprocessing import add_zero_columns, square_batch
from hegnn.data_processing.data_structures import HierarchicalBatch
from hegnn.data_processing.hdf5_structures import HDF5Dataset
from hegnn.data_processing.k_hop_subgraphs import ego_network_batch
from hegnn.data_processing.store_data_per_depth import next_depth_file
from hegnn.models.layers import GNN

TOTAL_DATA_TIME = 0


class HEGNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth: int,
        file_per_split: dict,
        graph_output: bool,
        params: dict,
        feature_encoder=lambda x: x,
    ):
        super(HEGNN, self).__init__()
        model_params = params.model_params
        self.depth = depth
        self.graph_output = graph_output
        self.hidden_dim = model_params.hidden_dim
        self.mode = model_params.mode
        self.layers_above = model_params.depth - self.depth
        self.n_hops = params.data.n_hops
        self.gradient_fraction = params.model_params.gradient_fraction
        self.depth_merge = model_params.depth_merge
        self.stored_input = params.data.stored_input
        self.feature_encoder = feature_encoder

        next_file_per_split = None
        self.next_data_per_split = None

        if depth == 0:
            self.he_gnn = None
            self.gnn = GNN(
                input_dim=input_dim,
                output_dim=output_dim,
                graph_output=graph_output,
                params=params,
                feature_encoder=feature_encoder,
                layers_above=self.layers_above,
            )
        else:
            if self.stored_input:
                next_file_per_split = {
                    k: next_depth_file(v) for k, v in file_per_split.items()
                }
                self.next_data_per_split = {
                    k: HDF5Dataset(v) for k, v in next_file_per_split.items()
                }
            else:
                next_file_per_split = {k: None for k, v in file_per_split.items()}
                self.next_data_per_split = {k: None for k, v in file_per_split.items()}

            if model_params.depth_merge == "concat":
                self.he_gnn = HEGNN(
                    input_dim=input_dim + 1,
                    output_dim=self.hidden_dim,
                    depth=depth - 1,
                    graph_output=model_params.pool_subgraphs,
                    file_per_split=next_file_per_split,
                    params=params,
                    feature_encoder=feature_encoder,
                )
                self.gnn = GNN(
                    input_dim=input_dim + self.hidden_dim,
                    output_dim=output_dim,
                    graph_output=graph_output,
                    params=params,
                    feature_encoder=feature_encoder,
                    layers_above=self.layers_above,
                )

            elif model_params.depth_merge == "sum":
                self.he_gnn = HEGNN(
                    input_dim=input_dim + 1,
                    output_dim=input_dim,
                    depth=depth - 1,
                    graph_output=model_params.pool_subgraphs,
                    file_per_split=next_file_per_split,
                    params=params,
                    feature_encoder=feature_encoder,
                )
                self.gnn = GNN(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    graph_output=graph_output,
                    params=params,
                    feature_encoder=feature_encoder,
                    layers_above=self.layers_above,
                )
            else:
                raise ValueError(
                    f"depth_merge must be either 'concat' or 'sum', found {model_params.depth_merge}"
                )

    def forward(self, batch, split=None):
        if self.layers_above == 0:
            batch.x = self.feature_encoder(batch.x)

        if self.depth == 0:
            return self.gnn(batch)  # Forward pass

        global TOTAL_DATA_TIME
        t = time.time()

        if self.stored_input:
            assert split is not None, "split must be specified when using stored data"
            next_depth_data = self.next_data_per_split[split]
            next_depth_batch = next_depth_data.get_graphs(batch.next_depth_graph_id)

            # Again encode the x
            next_depth_batch.x = self.feature_encoder(next_depth_batch.x)

        else:
            if self.n_hops == -1:
                next_depth_batch = square_batch(batch)
            else:
                next_depth_batch = ego_network_batch(batch, num_hops=self.n_hops)

        TOTAL_DATA_TIME += time.time() - t

        if self.mode == "original":
            deeper_output = self.he_gnn(next_depth_batch, split=split)  # Forward pass

        elif self.mode == "splitbatch":
            new_size = len(next_depth_batch)
            old_size = len(batch)
            growth = (new_size + old_size // 2) // old_size
            if len(next_depth_batch) > 22 and growth > 1:
                split_batch = next_depth_batch.split_into_parts(growth)
            else:
                split_batch = [next_depth_batch]

            gradient_branches = random.sample(
                range(0, len(split_batch)),
                int(len(split_batch) * self.gradient_fraction),
            )
            deeper_output = torch.cat(
                [
                    (
                        self.he_gnn(sub_batch, split=split)
                        if i in gradient_branches
                        else self.he_gnn(sub_batch, split=split).detach()
                    )
                    for i, sub_batch in enumerate(split_batch)
                ],
                dim=0,
            )
        else:
            raise ValueError("Mode not recognized")

        if self.depth_merge == "concat":
            x = torch.cat([batch.x, deeper_output], dim=1)
        elif self.depth_merge == "sum":
            x = batch.x + deeper_output
        new_data = HierarchicalBatch(
            x=x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch,
            unique_node_id=batch.unique_node_id,
        )
        return self.gnn(new_data)  # Forward pass

    def update_model(self, grads, params, lr=0.001):
        """Manually apply gradients to model parameters"""
        with torch.no_grad():  # Disable gradient tracking for manual update
            for param, grad in zip(params, grads):
                if param.grad is not None:
                    param.grad.zero_()  # Clear existing gradients
                param -= lr * grad  # Apply gradient update manually

    def custom_update_params(self, loss, optimizer):
        """Compute final loss and manually propagate gradients"""
        params = list(self.parameters())  # Get all parameters
        grads = torch.autograd.grad(
            loss, params, create_graph=False
        )  # Compute gradients

        for param, grad in zip(params, grads):
            param.grad = grad  # Manually set gradients

        optimizer.step()
        optimizer.zero_grad()

    def check_used_params(self, loss):
        """Check which parameters are used in the computation of the loss"""

        used_params = set()
        loss.backward(retain_graph=True)  # Standard backprop
        for name, param in self.named_parameters():
            if param.grad is not None:
                used_params.add(name)

        for name, param in self.named_parameters():
            print(name, param.requires_grad)

        unused_params = set(name for name, _ in self.named_parameters()) - used_params
        print("Unused parameters:", unused_params)

        used_params = list(used_params)
        print("\n USED PARAMS", used_params)

        assert len(unused_params) == 0, "Some gradients are not computed properly!!"
        print("All gradients computed")


class SingleNetHEGNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        graph_output=False,
        mode="original",
        num_layers=10,
        pna_params=None,
    ):
        super(SingleNetHEGNN, self).__init__()
        self.mode = mode
        self.graph_output = graph_output
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [
                GCNConv(
                    input_dim + depth + hidden_dim if i == 0 else hidden_dim, hidden_dim
                )
                for i in range(num_layers)
            ]
        )

        if self.graph_output:
            self.output_graph_regressor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),  # Output single value per graph
            )
        else:
            self.output_node_regressor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        if self.depth > 0:
            self.intermediate_node_regressor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def forward(self, batch, depth=None):
        if depth is None:
            depth = self.depth

        global TOTAL_DATA_TIME
        t = time.time()

        if depth == 0:
            x = add_zero_columns(batch.x, self.hidden_dim)
        else:
            start_time = time.time()
            uid_batch = square_batch(batch)
            TOTAL_DATA_TIME += time.time() - start_time

            if self.mode == "original":
                deeper_output = self.forward(uid_batch, depth=depth - 1)  # Forward pass

            elif self.mode == "splitbatch":
                start_time = time.time()
                new_size = len(uid_batch)
                old_size = len(batch)
                growth = (new_size + old_size // 2) // old_size
                if len(uid_batch) > 22 and growth > 1:
                    split_batch = uid_batch.split_into_parts(growth)
                else:
                    split_batch = [uid_batch]

                TOTAL_DATA_TIME += time.time() - start_time

                gradient_branches = random.sample(range(0, len(split_batch)), 2)
                deeper_output = torch.cat(
                    [
                        (
                            self.forward(sub_batch, depth - 1)
                            if i in gradient_branches
                            else self.forward(sub_batch, depth - 1).detach()
                        )
                        for i, sub_batch in enumerate(split_batch)
                    ],
                    dim=0,
                )
            else:
                raise ValueError("Mode not recognized")

            x = torch.cat([batch.x, deeper_output], dim=1)
            x = add_zero_columns(x, depth)

        for i in range(self.num_layers):
            x = self.convs[i](x, batch.edge_index)
            x = torch.relu(x)

        ## GRAPH OUTPUT
        if depth == self.depth:
            if self.graph_output:
                graph_feature = global_add_pool(x, batch.batch)
                return self.output_graph_regressor(graph_feature)
            else:
                out = self.output_node_regressor(x)

        else:
            output_nodes = batch.unique_node_id
            assert (
                output_nodes is not None
            ), "Output nodes must be specified for intermediate layers"

            output_features = x[output_nodes.squeeze(), :]
            out = self.intermediate_node_regressor(output_features)

        if out.ndim == 1:
            out = out.unsqueeze(0)

        return out

    def check_used_params(self, loss):
        """Check which parameters are used in the computation of the loss"""

        used_params = set()
        loss.backward(retain_graph=True)  # Standard backprop
        for name, param in self.named_parameters():
            if param.grad is not None:
                used_params.add(name)

        for name, param in self.named_parameters():
            print(name, param.requires_grad)

        unused_params = set(name for name, _ in self.named_parameters()) - used_params
        print("Unused parameters:", unused_params)

        used_params = list(used_params)
        print("\n USED PARAMS", used_params)

        assert len(unused_params) == 0, "Some gradients are not computed properly!!"

        print("All gradients computed")
