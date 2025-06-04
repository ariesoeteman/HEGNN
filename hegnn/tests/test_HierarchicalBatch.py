import pdb

import numpy as np
import torch
from torch_geometric.data import DataLoader
from torch_geometric.datasets import ZINC

from hegnn.data_processing.data_preprocessing import square_batch
from hegnn.data_processing.data_structures import HierarchicalBatch, merge_batch_list

torch.cuda.empty_cache()  # Clear cache after training or during cancellation
# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


train_dataset = ZINC(
    root="../data/basic_ZINC", subset=True, split="train"
)  # Train split
batch_size = 50
num_epochs = 20
depth = 2


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    drop_last=True,
    shuffle=True,
)


def test_split_extract_subbatch():
    N = 10
    for i, data in enumerate(train_loader):
        if i >= N:
            break

        data = HierarchicalBatch(
            x=data.x, edge_index=data.edge_index, batch=data.batch, y=data.y
        )
        uid_batch = square_batch(data)

        for _ in range(N):
            i_ind = np.random.randint(0, len(uid_batch))
            graph_one = uid_batch.extract_graph(i_ind)

            j_ind = np.random.randint(0, i_ind + 1)
            k_ind = np.random.randint(0, len(uid_batch) - i_ind)

            subbatch = uid_batch.extract_subbatch_between(j_ind, i_ind + k_ind)

            try:
                graph_two = subbatch.extract_graph(i_ind - j_ind)
                assert graph_one == graph_two, "Subbatch failed"
            except:
                pdb.set_trace()

        split_batch = uid_batch.split_into_parts(22)
        assert len(split_batch) == 22

        assert sum(len(b) for b in split_batch) == len(uid_batch)

        counter = 0
        for i in range(22):
            for j in range(len(split_batch[i])):
                graph = split_batch[i].extract_graph(j)
                try:
                    assert graph == uid_batch.extract_graph(
                        counter
                    ), "Split batch failed"
                except:
                    pdb.set_trace()
                counter += 1

    print("TEST SPLIT EXTRACT PASSED")


def test_unique_node_id():
    N = 10
    for i, data in enumerate(train_loader):
        if i >= N:
            break

        data = HierarchicalBatch(
            x=data.x, edge_index=data.edge_index, batch=data.batch, y=data.y
        )
        uid_batch = square_batch(data)

        assert len(uid_batch.unique_node_id) == data.x.shape[0]

        for _ in range(100):
            ind = np.random.randint(0, len(uid_batch))
            graph = uid_batch.extract_graph(ind)
            en_center = torch.where(graph.x[:, 1] == 1)[0][0].item()

            try:
                assert graph.unique_node_id[0] == en_center
            except:
                pdb.set_trace()

        for _ in range(100):
            i_ind = np.random.randint(0, len(uid_batch))
            j_ind = np.random.randint(0, i_ind + 1)
            k_ind = np.random.randint(0, len(uid_batch) - i_ind)

            subbatch = uid_batch.extract_subbatch_between(j_ind, i_ind + k_ind)

            graph_one = uid_batch.extract_graph(i_ind)
            en_center = torch.where(graph_one.x[:, 1] == 1)[0][0].item()
            try:
                graph_two = subbatch.extract_graph(i_ind - j_ind)
                assert graph_two.unique_node_id[0] == en_center
            except:
                pdb.set_trace()

    print("TEST UNIQUE NODE IDS PASSED")


def test_merge_batches():
    N = 10
    batches = []
    for i, data in enumerate(train_loader):
        if i >= N:
            break

        data = HierarchicalBatch(
            x=data.x, edge_index=data.edge_index, batch=data.batch, y=data.y
        )
        data = data.to(device)
        batches.append(data)

    merged_batch = merge_batch_list(batches)

    rev = batches[::-1]
    indirect_merged_batch = rev[0]
    for i in range(1, len(rev)):
        indirect_merged_batch = indirect_merged_batch.add_prefix(rev[i])

    assert (
        indirect_merged_batch == merged_batch
    ), "Merge from list and merge from prefix give different results"

    size_per_batch = merged_batch.get_batch_sizes()
    batch_indices = torch.arange(
        0, merged_batch.batch_size, device=device
    ).repeat_interleave(size_per_batch)

    assert torch.all(
        batch_indices == merged_batch.batch
    ), "Batch indices of merged batch are wrong"

    print("TEST MERGE BATCHES PASSED")


if __name__ == "__main__":
    test_split_extract_subbatch()
    test_unique_node_id()
    test_merge_batches()
    print("All tests passed")
