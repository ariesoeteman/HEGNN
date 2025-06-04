import copy
import pdb

import torch

from hegnn.config.parse_config import load_config
from hegnn.data_processing.data_preprocessing import square_batch
from hegnn.data_processing.data_structures import HierarchicalBatch
from hegnn.data_processing.hdf5_structures import CustomLoader, HDF5Dataset
from hegnn.data_processing.store_data_per_depth import load_data, next_depth_file


def test_loaded_vs_online_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()
    config.data.dataset = "srg"
    config.model_params.depth = 2
    config.batch_size = 1
    config.print_log = True
    config.wandb = False
    config.model_params.layer = "GINConv"

    ####################### tryout
    config.data.stored_input = True
    config.model_params.hidden_dim = 32
    config.n_epochs = 50
    config.learning_rate.patience = 15
    config.model_params.dropout = 0.0
    config.data.srg.n_isomorphisms = 30
    ########################
    config.batch_size = 6

    # TODO: set n_layers
    config.model_params.num_layers = 4
    config.data.overwrite = False

    store_data_config = copy.deepcopy(config)
    online_data_config = copy.deepcopy(config)
    store_data_config.data.stored_input = True
    online_data_config.data.stored_input = False

    store_data_config, _, data_file_per_split = load_data(store_data_config)
    online_data_config, dataloader_per_split, _ = load_data(online_data_config)

    train_dataset = HDF5Dataset(
        data_file_per_split["train"], device=device
    )  # original data (depth 0)
    stored_data_loader = CustomLoader(train_dataset, batch_size=config.batch_size)

    online_data_loader = dataloader_per_split["train"]

    for stored_data, online_data in zip(stored_data_loader, online_data_loader):

        if not isinstance(online_data, HierarchicalBatch):
            online_data = HierarchicalBatch.from_batch(online_data)
        stored_data.to(device)
        online_data.to(device)

        # Compare the data
        assert torch.equal(
            stored_data.x, online_data.x
        ), "Stored and online data do not match!"
        assert torch.equal(
            stored_data.edge_index, online_data.edge_index
        ), "Stored and online data do not match!"
        assert torch.equal(
            stored_data.y, online_data.y
        ), "Stored and online data do not match!"
        assert torch.equal(
            stored_data.batch, online_data.batch
        ), "Stored and online data do not match!"

        online_next_batch = square_batch(online_data)

        stored_next_data = HDF5Dataset(next_depth_file(data_file_per_split["train"]))
        stored_next_batch = stored_next_data.get_graphs(stored_data.next_depth_graph_id)

        assert torch.equal(
            stored_next_batch.x, online_next_batch.x
        ), "Stored and online data do not match!"
        assert torch.equal(
            stored_next_batch.edge_index, online_next_batch.edge_index
        ), "Stored and online data do not match!"
        assert (
            stored_next_batch.y is None and online_next_batch.y is None
        ), "Stored and online data do not match!"
        assert torch.equal(
            stored_next_batch.batch, online_next_batch.batch
        ), "Stored and online data do not match!"
        assert torch.equal(
            stored_next_batch.unique_node_id, online_next_batch.unique_node_id
        ), "Stored and online data do not match!"

    print("All data matches!")


test_loaded_vs_online_data()
