import pdb

from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from .data_preprocessing import power_batch


class MultiLevelBatchLoader:
    def __init__(self, dataset, batch_size, depth):
        pdb.set_trace()

        self.loader = DataLoader(
            list(dataset),  # Ensure it's a list of Data objects
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,  # Ensure this is used
        )
        self.depth = depth  # The number of batch levels per input batch

    def collate_fn(self, batch):
        """Transform a batch into a sequence of batches."""
        base_batch = Batch.from_data_list(batch)
        batch_sequence = power_batch(base_batch, self.depth)
        return batch_sequence  # Return sequence of batches for a single dataset batch

    def __iter__(self):
        for batch_sequence in self.loader:
            yield batch_sequence  # Yield all transformed batches for one input batch

    def __len__(self):
        return len(self.loader)  # Matches dataset batch count
