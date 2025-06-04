import pdb

import torch
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import BondEncoder
from torch import nn
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import degree

from hegnn.models.pnaconv import PNAConv


class GIN(MessagePassing):
    def __init__(self, in_dim, hidden_dim):
        super(GIN, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 2 * hidden_dim),
            torch.nn.BatchNorm1d(2 * hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim=in_dim)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is None:
            edge_embedding = None
        else:
            edge_embedding = self.bond_encoder(edge_attr)

        out = self.mlp(
            (1 + self.eps) * x
            + self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        )

        return out

    def message(self, x_j, edge_attr):
        if edge_attr is None:
            return F.relu(x_j)

        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class ZINCGIN(MessagePassing):
    def __init__(self, in_dim, hidden_dim):
        super(ZINCGIN, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = torch.nn.Embedding(4, in_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr.squeeze())
        out = self.mlp(
            (1 + self.eps) * x
            + self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        )
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GCN(MessagePassing):
    def __init__(self, in_dim, hidden_dim):
        super(GCN, self).__init__(aggr="add")

        self.linear = torch.nn.Linear(in_dim, hidden_dim)
        self.root_emb = torch.nn.Embedding(1, hidden_dim)
        self.bond_encoder = BondEncoder(emb_dim=hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        if edge_attr is None:
            edge_embedding = None
        else:
            edge_embedding = self.bond_encoder(edge_attr.unsqueeze(1))

        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(
            edge_index, x=x, edge_attr=edge_embedding, norm=norm
        ) + F.relu(x + self.root_emb.weight) * 1.0 / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):

        if edge_attr is None:
            return norm.view(-1, 1) * F.relu(x_j)
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GNN to generate node embedding
class GNN(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        graph_output=False,
        params=None,
        feature_encoder=lambda x: x,
        layers_above=None,
    ):
        super(GNN, self).__init__()
        model_params = params.model_params
        self.graph_output = graph_output
        self.num_layers = model_params.num_layers
        self.residual = model_params.residual
        self.uid_residual = model_params.uid_residual
        self.cycle_residual = model_params.cycle_residual
        hidden_dim = model_params.hidden_dim
        self.dropout = model_params.dropout
        self.gnn_type = model_params.layer
        self.layers_above = layers_above
        self.n_cycle_features = len(params.cycles) if params.cycles is not None else 0

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = feature_encoder
        self.convs = torch.nn.ModuleList()
        if model_params.batch_norm:
            self.batch_norms = torch.nn.ModuleList()
        else:
            self.batch_norms = None

        for layer in range(self.num_layers):
            if self.gnn_type == "GCNConv":
                self.convs.append(
                    GCN(hidden_dim if layer != 0 else input_dim, hidden_dim)
                )
            elif self.gnn_type == "ZINCGINConv":
                self.convs.append(
                    ZINCGIN(hidden_dim if layer != 0 else input_dim, hidden_dim)
                )
            elif self.gnn_type == "GINConv":
                self.convs.append(
                    GIN(hidden_dim if layer != 0 else input_dim, hidden_dim)
                )
            elif self.gnn_type == "PNAConv":
                pna_params = model_params.pna_params
                degrees, aggregators, scalers = (
                    pna_params.degrees,
                    pna_params.aggregators,
                    pna_params.scalers,
                )
                self.convs.append(
                    PNAConv(
                        hidden_dim if layer != 0 else input_dim,
                        hidden_dim,
                        aggregators=aggregators,
                        scalers=scalers,
                        deg=degrees,
                        edge_dim=1,
                        divide_input=True,
                    )
                ),
            else:
                raise ValueError("Undefined GNN type called {}".format(self.gnn_type))

            if model_params.batch_norm:
                self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
                if self.n_cycle_features > 0:
                    self.cycle_norm = torch.nn.BatchNorm1d(self.n_cycle_features)

        if params.data.task == "classification" and self.layers_above == 0:
            self.output_generator = nn.Sequential(
                nn.Linear(hidden_dim, params.data.n_classes)  # One output per class
            )
        else:
            self.output_generator = nn.Sequential(
                nn.Linear(hidden_dim, output_dim)  # Output single value per graph
            )

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch, output_nodes = (
            batched_data.x,
            batched_data.edge_index,
            batched_data.edge_attr,
            batched_data.batch,
            batched_data.unique_node_id,
        )

        # Ensure x is in float format in a differentiable way
        if not torch.is_floating_point(x):
            x = x.float()

        h_list = [x]

        for layer in range(self.num_layers):
            # Get features for residual connections
            if self.layers_above > 0:
                try:
                    unique_node_features = h_list[layer][:, -self.layers_above :]
                except Exception as e:
                    pdb.set_trace()
            if self.n_cycle_features > 0:
                if self.layers_above > 0:
                    cycle_features = x[
                        :,
                        -self.n_cycle_features - self.layers_above : -self.layers_above,
                    ].clone()
                else:
                    cycle_features = x[:, -self.n_cycle_features :].clone()

                if self.batch_norms:
                    cycle_features = self.cycle_norm(cycle_features)

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            if self.batch_norms:
                h = self.batch_norms[layer](h)

            if self.gnn_type == "ZINCGINConv":
                h = F.relu(h)  # remove last relu for ogb

            if self.dropout > 0.0:
                h = F.dropout(h, self.dropout, training=self.training)

            ### Residual connections for extra features
            if self.residual:
                h += h_list[layer]
            else:
                if self.uid_residual and self.layers_above > 0:
                    h_updated = h[:, -self.layers_above :] + unique_node_features
                    h = torch.cat((h[:, : -self.layers_above], h_updated), dim=1)
                if self.cycle_residual and self.n_cycle_features > 0:

                    if self.layers_above > 0:
                        cycle_columns = list(
                            range(
                                h.shape[1] - self.n_cycle_features - self.layers_above,
                                h.shape[1] - self.layers_above,
                            )
                        )
                        h_updated = h[:, cycle_columns] + cycle_features
                        h = torch.cat(
                            (
                                h[:, : -self.n_cycle_features - self.layers_above],
                                h_updated,
                                h[:, -self.layers_above :],
                            ),
                            dim=1,
                        )
                    else:
                        h_updated = h[:, -self.n_cycle_features :] + cycle_features
                        h = torch.cat(
                            (h[:, : -self.n_cycle_features], h_updated), dim=1
                        )
            h_list.append(h)

        x = h_list[-1]

        if self.graph_output:
            graph_feature = global_add_pool(
                x, batch
            )  # Aggregate node embeddings into graph representation
            return self.output_generator(graph_feature)  # Predict graph-level value

        if output_nodes is not None:
            output_features = x[output_nodes.squeeze(), :]
            out = self.output_generator(output_features)
        else:
            out = self.output_generator(x)
        return out
