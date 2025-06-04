import errno
import os
import os.path as osp
import pickle

import networkx as nx
import torch
from torch.utils.data import Subset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected

from hegnn.globals import all_data_dir


def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


def load_sr_dataset(path):
    """Load the Strongly Regular Graph Dataset from the supplied path."""
    nx_graphs = nx.read_graph6(path)
    graphs = list()
    for nx_graph in nx_graphs:
        n = nx_graph.number_of_nodes()
        edge_index = to_undirected(
            torch.tensor(list(nx_graph.edges()), dtype=torch.long).transpose(1, 0)
        )
        graphs.append((edge_index, n))

    return graphs


def load_sr_graph_dataset(name, root=all_data_dir, prefer_pkl=False):
    raw_dir = os.path.join(root, "SR_graphs", "raw")
    load_from = os.path.join(raw_dir, "{}.g6".format(name))
    load_from_pkl = os.path.join(raw_dir, "{}.pkl".format(name))
    if prefer_pkl and osp.exists(load_from_pkl):
        print(f"Loading SR graph {name} from pickle dump...")
        with open(load_from_pkl, "rb") as handle:
            data = pickle.load(handle)
    else:
        data = load_sr_dataset(load_from)
    graphs = list()
    for datum in data:
        edge_index, num_nodes = datum
        x = torch.ones(num_nodes, 1, dtype=torch.float32)
        graph = Data(
            x=x, edge_index=edge_index, y=None, edge_attr=None, num_nodes=num_nodes
        )
        graphs.append(graph)
    train_ids = list(range(len(graphs)))
    val_ids = list(range(len(graphs)))
    test_ids = list(range(len(graphs)))
    return graphs, train_ids, val_ids, test_ids


def generate_isomorphisms(data: InMemoryDataset, num_permutations: int, graph_id: int):
    isomorphs = []
    num_nodes = data.num_nodes

    for _ in range(num_permutations):
        perm = torch.randperm(num_nodes)

        # Permute features and edge indices
        x_perm = data.x[perm]
        edge_index = data.edge_index.clone()
        edge_index[0] = perm[edge_index[0]]
        edge_index[1] = perm[edge_index[1]]

        iso_data = Data(x=x_perm, edge_index=edge_index, num_nodes=num_nodes)

        # Add origin ID as label
        iso_data.y = torch.tensor([graph_id], dtype=torch.long)

        isomorphs.append(iso_data)

    return isomorphs


class CustomSRGDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        name,
        n_isomorphisms,
        transform=None,
        pre_transform=None,
        train_fraction=0.8,
        val_fraction=0.1,
        test_fraction=0.1,
    ):
        self.name = name
        self.n_isos = n_isomorphisms

        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction

        super().__init__(root, transform, pre_transform)
        self.data, self.slices, self.split_dict = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["dummy"]

    def download(self):
        pass

    @property
    def processed_file_names(self):
        return [f"{self.name}_{self.n_isos}_isos.pt"]

    def process(self):
        # Load original graphs
        graphs, _, _, _ = load_sr_graph_dataset(self.name, prefer_pkl=True)
        all_isomorphs = []
        for graph_id, graph in enumerate(graphs):
            isos = generate_isomorphisms(
                graph, num_permutations=self.n_isos, graph_id=graph_id
            )
            all_isomorphs.extend(isos)

        # Pre-transform if needed
        if self.pre_transform is not None:
            all_isomorphs = [self.pre_transform(g) for g in all_isomorphs]

        # Collate and save
        data, slices = InMemoryDataset.collate(all_isomorphs)

        # Determine split sizes
        total_size = len(all_isomorphs)
        print(f"Total size of dataset: {total_size}")

        train_size = int(self.train_fraction * total_size)
        val_size = int(self.val_fraction * total_size)

        # Deterministic shuffle:
        # generator = torch.Generator().manual_seed(42)
        # indices = torch.randperm(total_size, generator=generator).tolist()

        indices = torch.randperm(total_size).tolist()
        split_dict = {
            "train": indices[:train_size],
            "val": indices[train_size : train_size + val_size],
            "test": indices[train_size + val_size :],
        }

        # Save everything
        torch.save((data, slices, split_dict), self.processed_paths[0])

    def split(self):
        return (
            Subset(self, self.split_dict["train"]),
            Subset(self, self.split_dict["val"]),
            Subset(self, self.split_dict["test"]),
        )

    @property
    def n_classes(self):
        return len(set(self.y.tolist()))


def print_dataloader_stats(dataloader):
    num_batches = len(dataloader)
    total_nodes = 0
    total_edges = 0
    avg_batch_size = 0

    for batch in dataloader:
        num_nodes = batch.num_nodes
        num_edges = batch.edge_index.size(1)  # size[1] gives the number of edges

        total_nodes += num_nodes
        total_edges += num_edges
        avg_batch_size += len(batch)

    avg_batch_size /= num_batches
    print(f"Total Batches: {num_batches}")
    print(f"Average Batch Size: {avg_batch_size:.2f}")
    print(f"Total Nodes: {total_nodes}")
    print(f"Total Edges: {total_edges}")
    print(f"Average Nodes per Batch: {total_nodes / num_batches:.2f}")
    print(f"Average Edges per Batch: {total_edges / num_batches:.2f}")
