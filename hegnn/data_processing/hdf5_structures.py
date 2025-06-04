import pdb
import time

import h5py
import torch
from torch.utils.data import Dataset

from hegnn.data_processing.data_structures import HierarchicalBatch, merge_batch_list


def to_numpy(group, key, device=None):
    """
    Convert a numpy array to a PyTorch tensor.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if group.attrs.get(f"{key}_none", False):
        return None
    else:
        if key in ["y"]:
            return group[key][:]
        elif key in ["x"]:
            return group[key][:]
        elif key in [
            "edge_index",
            "edge_attr",
            "batch",
            "unique_node_id",
            "next_depth_graph_id",
        ]:
            return group[key][:]
        else:
            raise ValueError(f"Unknown key: {key}")


def to_tensor(group, key, device=None):
    """
    Convert a numpy array to a PyTorch tensor.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if group.attrs.get(f"{key}_none", False):
        return None
    else:
        if key in ["y"]:
            return torch.tensor(group[key][:], dtype=torch.float, device=device)
        elif key in ["x"]:
            return torch.tensor(group[key][:], dtype=torch.long, device=device)
        elif key in [
            "edge_index",
            "edge_attr",
            "batch",
            "unique_node_id",
            "next_depth_graph_id",
        ]:
            return torch.tensor(group[key][:], dtype=torch.long, device=device)
        else:
            raise ValueError(f"Unknown key: {key}")


def array_list_to_tensor_list(array_list, dim=1):

    if all(x is None for x in array_list):
        return array_list
    elif any(x is None for x in array_list):
        raise ValueError(f"array_list contains None and other values: {array_list}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_list = [torch.from_numpy(x) for x in array_list]  # shape: [N_i]

    # Step 2: Pad to max length
    if dim == 1:
        lengths = [t.size(0) for t in tensor_list]
        padded_tensor = torch.nn.utils.rnn.pad_sequence(
            tensor_list, batch_first=True, padding_value=0.0
        )  # shape: (num_arrays, max_length)
        padded_tensor = padded_tensor.to(device)
        unpadded_list = [padded_tensor[i, :l] for i, l in enumerate(lengths)]

    elif dim == 2:
        lengths = [t.shape[1] for t in tensor_list]
        max_len = max(t.shape[1] for t in tensor_list)

        # Pad each to shape (2, max_len)
        padded_tensor = torch.stack(
            [torch.nn.functional.pad(t, (0, max_len - t.shape[1])) for t in tensor_list]
        )  # shape: [batch_size, 2, max_len]
        padded_tensor = padded_tensor.to(device)
        unpadded_list = [padded_tensor[i, :, :l] for i, l in enumerate(lengths)]

    else:
        raise ValueError(f"Unsupported dimension: {dim}")

    return unpadded_list


class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path, device=None):
        """
        Custom dataset for data stored as HDF5.
        """
        with h5py.File(hdf5_path, "r") as f:
            n_groups = len(f.keys())
        if n_groups == 0:
            raise ValueError(f"No groups found in the HDF5 file: {hdf5_path}")

        self.hdf5_path = hdf5_path
        self.n_groups = n_groups
        self.keys = [f"group_{group_id}" for group_id in range(n_groups)]
        self.index = 0
        self.device = device

        with h5py.File(hdf5_path, "r") as f:
            self.storage_size = f["group_0"].attrs.get("n_graphs")
            self.n_graphs = sum(f[key].attrs.get("n_graphs") for key in self.keys)

    def __len__(self):
        return len(self.keys)

    def __iter__(self):
        """
        Iterate over the dataset.
        """
        self.index = 0
        return self

    def __next__(self):
        """
        Get the next item in the dataset.
        """
        if self.index >= len(self):
            raise StopIteration
        else:
            item = self[self.index]
            self.index += 1
            return item

    def __getitem__(self, index):
        """
        Loads only the requested subset from the HDF5 file.
        """
        group_key = self.keys[index]
        with h5py.File(self.hdf5_path, "r") as f:
            group = f[group_key]
            data = HierarchicalBatch(
                x=to_tensor(group, "x", self.device),
                edge_index=to_tensor(group, "edge_index", self.device),
                edge_attr=to_tensor(group, "edge_attr", self.device),
                y=to_tensor(group, "y", self.device),
                batch=to_tensor(group, "batch", self.device),
                unique_node_id=to_tensor(group, "unique_node_id", self.device),
                next_depth_graph_id=to_tensor(
                    group, "next_depth_graph_id", self.device
                ),
            )
        return data

    def get_batch_list(self, indices):
        with h5py.File(self.hdf5_path, "r") as f:
            groups = [f[self.keys[i]] for i in indices]

            xlist = [to_numpy(group, "x") for group in groups]
            ylist = [to_numpy(group, "y") for group in groups]
            edge_index_list = [to_numpy(group, "edge_index") for group in groups]
            edge_attr_list = [to_numpy(group, "edge_attr") for group in groups]
            batch_list = [to_numpy(group, "batch") for group in groups]
            unique_node_id_list = [
                to_numpy(group, "unique_node_id") for group in groups
            ]
            next_depth_graph_id_list = [
                to_numpy(group, "next_depth_graph_id") for group in groups
            ]

        tensor_dict_list = {
            "x": array_list_to_tensor_list(xlist),
            "edge_index": array_list_to_tensor_list(edge_index_list, dim=2),
            "edge_attr": array_list_to_tensor_list(edge_attr_list),
            "y": array_list_to_tensor_list(ylist),
            "batch": array_list_to_tensor_list(batch_list),
            "unique_node_id": array_list_to_tensor_list(unique_node_id_list),
            "next_depth_graph_id": array_list_to_tensor_list(next_depth_graph_id_list),
        }

        batch_list = []
        for i in range(len(tensor_dict_list["x"])):
            batch = HierarchicalBatch(
                x=tensor_dict_list["x"][i],
                edge_index=tensor_dict_list["edge_index"][i],
                edge_attr=tensor_dict_list["edge_attr"][i],
                y=tensor_dict_list["y"][i],
                batch=tensor_dict_list["batch"][i],
                unique_node_id=tensor_dict_list["unique_node_id"][i],
                next_depth_graph_id=tensor_dict_list["next_depth_graph_id"][i],
            )
            batch_list.append(batch)

        return batch_list

    # TODO not tested
    def graph_by_index(self, index):
        """
        Get the group for a given index.
        """
        if index < 0 or index >= self.n_graphs:
            raise ValueError(f"index: {index} out of range")

        group = index // self.storage_size
        index_in_group = index % self.storage_size
        batch = self[group]
        return batch.extract_graph(index_in_group)

    def get_graphs_by_indices(self, indices):
        """
        Get all graphs between start_graph and end_graph INCLUDING the end graph
        Much slower than get_graphs_between
        """
        if len(indices) == 0:
            raise ValueError(f"Indices cannot be empty")
        graph_list = [self.graph_by_index(i) for i in indices]
        batch_list = [graph.to_batch() for graph in graph_list]
        merged_batch = merge_batch_list(batch_list)
        return merged_batch

    def get_graphs_between(self, start_graph, end_graph):
        """
        Get all graphs between start_graph and end_graph INCLUDING the end graph
        """
        t = time.time()

        if start_graph < 0 or end_graph >= self.n_graphs or start_graph > end_graph:
            raise ValueError(
                f"start_graph: {start_graph}, end_graph: {end_graph} out of range. Allowed: 0-{self.n_graphs-1}"
            )

        start_group = start_graph // self.storage_size
        end_group = end_graph // self.storage_size

        start_graph = start_graph % self.storage_size
        end_graph = end_graph % self.storage_size

        batches = self.get_batch_list(list(range(start_group, end_group + 1)))

        subbatches = []
        for i, batch in enumerate(batches):
            if len(batches) == 1:
                subbatch = batch.extract_subbatch_between(start_graph, end_graph)
            else:
                if i == 0:
                    subbatch = batch.extract_subbatch_between(
                        start_graph, self.storage_size - 1
                    )
                elif i == len(batches) - 1:
                    subbatch = batch.extract_subbatch_between(0, end_graph)
                else:
                    subbatch = batch
            subbatches.append(subbatch)

        merged_batch = merge_batch_list(subbatches)
        return merged_batch

    def get_graphs(self, graph_ids):
        assert list(graph_ids) == list(
            range(graph_ids[0], graph_ids[-1] + 1)
        ), f"graph_ids must be continuous, got {graph_ids}"
        return self.get_graphs_between(graph_ids[0], graph_ids[-1])

    def get_graph(self, i):
        """
        Get a single graph by its index.
        """
        if i < 0 or i >= self.n_graphs:
            raise ValueError(f"i: {i} out of range")

        return self.get_graphs_between(i, i)

    def full_data(self):
        """
        Returns all data in the dataset as a single HierarchicalBatch object.
        """
        batches = [self[i] for i in range(self.n_groups)]
        merged_batch = merge_batch_list(batches)
        return merged_batch


class CustomLoader(object):
    """
    Custom loader for the HDF5Dataset.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        assert isinstance(
            dataset, HDF5Dataset
        ), f"dataset must be of type HDF5Dataset, got {type(dataset)}"

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        if self.shuffle:
            self.graph_ids = torch.randperm(self.dataset.n_graphs)
            self.graph_ids_per_batch = [
                self.graph_ids[i : min(i + self.batch_size - 1, dataset.n_graphs - 1)]
                for i in range(0, len(self.graph_ids), self.batch_size)
            ]

    def __len__(self):
        return self.dataset.n_graphs // self.batch_size + (
            1 if self.dataset.n_graphs % self.batch_size > 0 else 0
        )

    def __iter__(self):
        self.index = 0
        return self

    def __getitem__(self, index):
        assert 0 <= index < len(self), f"index: {index} out of range"

        if not self.shuffle:
            batch_start = index * self.batch_size
            batch_end = min(
                (index + 1) * self.batch_size - 1, self.dataset.n_graphs - 1
            )
            return self.dataset.get_graphs_between(batch_start, batch_end)
        else:
            graph_indices = self.graph_ids_per_batch[index]
            return self.dataset.get_graphs_by_indices(graph_indices)

    def __next__(self):
        if self.index >= len(self):
            raise StopIteration
        else:
            item = self[self.index]
            self.index += 1
            return item
