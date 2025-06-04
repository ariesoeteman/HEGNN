import pdb

import torch
from torch_geometric.data import Batch, Data


class HierarchicalData(Data):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "unique_node_id" in kwargs:
            self._unique_node_id = kwargs["unique_node_id"]
        else:
            self._unique_node_id = None

        if "next_depth_graph_id" in kwargs:
            self._next_depth_graph_id = kwargs["next_depth_graph_id"]
        else:
            self._next_depth_graph_id = None

    @property
    def unique_node_id(self):
        return getattr(self, "_unique_node_id", None)

    @property
    def next_depth_graph_id(self):
        return getattr(self, "_next_depth_graph_id", None)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "unique_node_id":
            return None  # Prevents concatenation, stores as a list per graph
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == "unique_node_id":
            return self.x.size(0)  # Increment by the number of nodes in this graph
        # if key == "next_depth_graph_id":
        #     return 0
        return super().__inc__(key, value, *args, **kwargs)

    def __eq__(self, other):
        if not isinstance(other, HierarchicalData):
            return False

        return (
            torch.equal(self.x, other.x)
            and torch.equal(self.edge_index, other.edge_index)
            and (
                self.unique_node_id is None
                and other.unique_node_id is None
                or torch.equal(self.unique_node_id, other.unique_node_id)
            )
            and (
                self.next_depth_graph_id is None
                and other.next_depth_graph_id is None
                or torch.equal(self.next_depth_graph_id, other.next_depth_graph_id)
            )
            and (self.y is None and other.y is None or torch.equal(self.y, other.y))
            and (
                self.edge_attr is None
                and other.edge_attr is None
                or torch.equal(self.edge_attr, other.edge_attr)
            )
        )

    def to_batch(self):
        """Converts the HierarchicalData to a HierarchicalBatch object"""
        if self.y is None:
            batch_y = None
        else:
            batch_y = torch.tensor([self.y], device=self.x.device)
        batch = HierarchicalBatch(
            x=self.x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            y=batch_y,
            unique_node_id=self.unique_node_id,
            next_depth_graph_id=self.next_depth_graph_id,
            batch=torch.zeros(
                self.x.size(0), dtype=torch.long, device=self.x.device
            ),  # Create a dummy batch tensor
        )
        return batch


class HierarchicalBatch(Batch):
    def __init__(self, unique_node_id=None, next_depth_graph_id=None, **kwargs):
        super().__init__(**kwargs)
        self._unique_node_id = unique_node_id
        self._next_depth_graph_id = next_depth_graph_id

        if self.batch is not None:
            self.batch_size = self.batch[-1].item() + 1
        else:
            self.batch_size = None

    def __eq__(self, other):
        if not isinstance(other, HierarchicalBatch):
            return False

        return (
            (self.y is None and other.y is None or torch.equal(self.y, other.y))
            and torch.equal(self.x, other.x)
            and torch.equal(self.edge_index, other.edge_index)
            and torch.equal(self.batch, other.batch)
            and (
                self.unique_node_id is None
                and other.unique_node_id is None
                or torch.equal(self.unique_node_id, other.unique_node_id)
            )
            and (
                self.next_depth_graph_id is None
                and other.next_depth_graph_id is None
                or torch.equal(self.next_depth_graph_id, other.next_depth_graph_id)
            )
            and (
                self.edge_attr is None
                and other.edge_attr is None
                or torch.equal(self.edge_attr, other.edge_attr)
            )
        )

    # Returns torch tensor
    def get_batch_sizes(self):
        """Returns the number of nodes in each graph within a batch."""
        unique, counts = torch.unique(self.batch, return_counts=True)
        return counts.to(self.x.device)  # Ensure the counts are on the same device as x

    # Returns torch tensor
    def get_edge_counts(batch):
        """Returns the number of edges in each graph within a batch (supports GPU)."""
        edge_batch = batch.batch[
            batch.edge_index[0]
        ]  # Assign edges to batches based on source nodes
        unique, counts = torch.unique(edge_batch, return_counts=True)
        return counts

    @classmethod
    def from_batch(cls, batch):
        if not isinstance(batch, Batch):
            raise ValueError(f"Expected a Batch object, got {type(batch)}")
        if batch.x is None:
            x = torch.zeros(
                (batch.batch.size(0), 1), device=batch.edge_index.device
            )  # Create a dummy x tensor if not provided
        else:
            x = batch.x
        return cls(
            x=x,
            edge_index=batch.edge_index,
            batch=batch.batch,
            edge_attr=batch.edge_attr,
            y=batch.y,
        )

    @classmethod
    def add_next_depth_graph_id(cls, batch, value):
        if not isinstance(batch, HierarchicalBatch):
            raise ValueError(f"Expected a HierarchicalBatch object, got {type(batch)}")
        new_batch = HierarchicalBatch(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            y=batch.y,
            unique_node_id=batch.unique_node_id,
            next_depth_graph_id=value,
            batch=batch.batch,
        )
        return new_batch

    @property
    def unique_node_id(self):
        return getattr(self, "_unique_node_id", None)

    @property
    def next_depth_graph_id(self):
        return getattr(self, "_next_depth_graph_id", None)

    def to(self, device):
        """
        Move the batch to the specified device.
        """
        super().to(device)
        if self.unique_node_id is not None:
            self._unique_node_id = self.unique_node_id.to(device)
        if self.next_depth_graph_id is not None:
            self._next_depth_graph_id = self.next_depth_graph_id.to(device)
        return self

    def extract_graph(self, idx):
        if self.batch_size is None or idx >= self.batch_size:
            raise ValueError(f"Size: {self.batch_size}, idx: {idx} in 'extract_graph'")

        mask = (self.batch == idx).to(
            self.edge_index.device
        )  # Find nodes belonging to graph `idx`
        y = (
            self.y[idx] if self.y is not None else None
        )  # Extract the target value for the selected graph
        x = self.x[mask]

        edge_mask = mask[self.edge_index[0]]
        edge_index = self.edge_index[:, edge_mask]
        if self.edge_attr is None:
            edge_attr = None
        else:
            edge_attr = self.edge_attr[edge_mask]

        first_batch_node_idx = torch.nonzero(mask, as_tuple=True)[0][0].item()

        # Rescale to start from 0
        edge_index = edge_index - first_batch_node_idx

        if self.unique_node_id is None:
            unique_node_id = None
        else:
            # Rescale to start from 0
            unique_node_id = self.unique_node_id[idx, :] - first_batch_node_idx

        if self.next_depth_graph_id is None:
            next_depth_graph_id = None
        else:
            next_depth_graph_id = self.next_depth_graph_id[mask]
        return HierarchicalData(
            x=x,
            y=y,
            edge_index=edge_index,
            edge_attr=edge_attr,
            unique_node_id=unique_node_id,
            next_depth_graph_id=next_depth_graph_id,
        )

    def extract_subbatch_between(self, start_idx, end_idx):
        """Returns a HierarchicalBatch with the concatenated data, including start and end"""
        if (
            self.batch_size is None
            or start_idx >= self.batch_size
            or end_idx >= self.batch_size
            or start_idx > end_idx
        ):
            raise ValueError(
                f"Size: {self.batch_size}, start_idx: {start_idx}, end_idx: {end_idx} in 'extract_graphs_between'"
            )

        # Create a mask for all nodes belonging to graphs in the range [start_idx, end_idx]
        mask = (self.batch >= start_idx) & (self.batch <= end_idx)
        y = (
            self.y[start_idx : end_idx + 1] if self.y is not None else None
        )  # Extract the target value for the selected graphs
        x = self.x[mask]  # Extract nodes belonging to the selected graphs

        batch = self.batch[mask] - start_idx  # Shift batch indices to start from 0

        # Create a mask for edges, ensuring edges belong to the selected graphs
        edge_mask = mask[self.edge_index[0]]
        edge_index = self.edge_index[:, edge_mask]
        if self.edge_attr is None:
            edge_attr = None
        else:
            edge_attr = self.edge_attr[edge_mask]

        # Shift edge indices for the whole batch
        first_batch_node_idx = torch.nonzero(mask, as_tuple=True)[0][0].item()
        edge_index = edge_index - first_batch_node_idx

        # Handle unique node IDs
        if self.unique_node_id is None:
            unique_node_id = None
        else:
            unique_node_id = (
                self.unique_node_id[start_idx : end_idx + 1, :] - first_batch_node_idx
            )
        if self.next_depth_graph_id is None:
            next_depth_graph_id = None
        else:
            next_depth_graph_id = self.next_depth_graph_id[mask]

        return HierarchicalBatch(
            x=x,
            y=y,
            edge_index=edge_index,
            edge_attr=edge_attr,
            unique_node_id=unique_node_id,
            batch=batch,
            next_depth_graph_id=next_depth_graph_id,
        )

    def split_into_parts(self, num_parts):
        """Divides the batch into N parts and returns a list of subbatches"""
        if num_parts < 1:
            raise ValueError(f"num_parts must be at least 1, got {num_parts}")
        assert type(num_parts) == int, f"num_parts must be an integer, got {num_parts}"

        part_size = self.batch_size // (num_parts)
        remainder = self.batch_size % (num_parts)

        subbatches = []
        start_idx = 0

        for i in range(num_parts):
            end_idx = start_idx + part_size + (1 if i < remainder else 0)
            subbatch = self.extract_subbatch_between(start_idx, end_idx - 1)
            subbatches.append(subbatch)
            start_idx = end_idx

        return subbatches

    def add_prefix(self, prefix):
        if prefix is None:
            return self
        assert isinstance(
            prefix, HierarchicalBatch
        ), f"Prefix must be a HierarchicalBatch, got {type(prefix)}"

        # Concatenate the prefix data with the current batch data
        n_prefix_nodes = prefix.x.size(0)
        y = torch.cat([prefix.y, self.y], dim=0) if self.y is not None else None
        x = torch.cat([prefix.x, self.x], dim=0)
        edge_index = torch.cat(
            [prefix.edge_index, self.edge_index + n_prefix_nodes], dim=1
        )

        batch = torch.cat([prefix.batch, self.batch + prefix.batch_size], dim=0)

        if (self.edge_attr is None) != (prefix.edge_attr is None):
            raise ValueError(
                "Both batches must have edge_attr None or not None to concatenate them."
            )
        if self.edge_attr is None:
            edge_attr = None
        else:
            edge_attr = torch.cat([prefix.edge_attr, self.edge_attr], dim=0)

        if (self.unique_node_id is None) != (prefix.unique_node_id is None):
            raise ValueError(
                "Both batches must have unique_node_id None or not None to concatenate them."
            )
        if self.unique_node_id is None:
            unique_node_id = None
        else:
            unique_node_id = torch.cat(
                [prefix.unique_node_id, self.unique_node_id + n_prefix_nodes], dim=0
            )

        if (self.next_depth_graph_id is None) != (prefix.next_depth_graph_id is None):
            raise ValueError(
                "Both batches must have next_depth_graph_id None or not None to concatenate them."
            )
        if self.next_depth_graph_id is None:
            next_depth_graph_id = None
        else:
            raise ValueError(
                "Add prefix with existing next_graph_index. I think something is wrong here"
            )

        return HierarchicalBatch(
            x=x,
            y=y,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            unique_node_id=unique_node_id,
            next_depth_graph_id=next_depth_graph_id,
        )

    def to_list(self):
        return [self.extract_graph(i) for i in range(self.batch_size)]


def merge_batch_list(batch_list):
    """
    Merges a list of HierarchicalBatch objects into a single HierarchicalBatch.
    """
    assert isinstance(
        batch_list, list
    ), f"batch_list must be a list, got {type(batch_list)}"
    assert all(
        isinstance(b, HierarchicalBatch) for b in batch_list
    ), f"All elements in batch_list must be HierarchicalBatch, got {[type(b) for b in batch_list]}"

    x = torch.cat([b.x for b in batch_list], dim=0)
    edge_index = torch.cat(
        [
            b.edge_index + sum([bb.x.size(0) for bb in batch_list[:i]])
            for i, b in enumerate(batch_list)
        ],
        dim=1,
    )

    batch_sizes = torch.tensor([b.batch_size for b in batch_list], dtype=torch.long)
    batch_sizes = torch.cat([torch.tensor([0], dtype=torch.long), batch_sizes], dim=0)
    cum_batch_sizes = torch.cumsum(batch_sizes, dim=0)
    batch = torch.cat(
        [b.batch + cum_batch_sizes[i] for i, b in enumerate(batch_list)], dim=0
    )

    assert all(b.edge_attr is None for b in batch_list) or all(
        b.edge_attr is not None for b in batch_list
    ), "edge_attr must be all None or all not None to merge batches"
    assert all(b.y is None for b in batch_list) or all(
        b.y is not None for b in batch_list
    ), "y must be all None or all not None to merge batches"
    assert all(b.unique_node_id is None for b in batch_list) or all(
        b.unique_node_id is not None for b in batch_list
    ), "unique_node_id must be all None or not None to merge batches"
    assert all(b.next_depth_graph_id is None for b in batch_list) or all(
        b.next_depth_graph_id is not None for b in batch_list
    ), "unique_node_id must be all None or not None to merge batches"
    y = None if batch_list[0].y is None else torch.cat([b.y for b in batch_list], dim=0)

    if all(b.edge_attr is None for b in batch_list):
        edge_attr = None
    else:
        edge_attr = torch.cat([b.edge_attr for b in batch_list], dim=0)

    if all(b.unique_node_id is None for b in batch_list):
        unique_node_id = None
    else:
        unique_node_id = torch.cat(
            [
                b.unique_node_id + sum([bb.x.size(0) for bb in batch_list[:i]])
                for i, b in enumerate(batch_list)
            ],
            dim=0,
        )

    if all(b.next_depth_graph_id is None for b in batch_list):
        next_depth_graph_id = None
    else:
        next_depth_graph_id = torch.cat(
            [b.next_depth_graph_id for b in batch_list], dim=0
        )
    merged_batch = HierarchicalBatch(
        x=x,
        y=y,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
        unique_node_id=unique_node_id,
        next_depth_graph_id=next_depth_graph_id,
    )

    return merged_batch
