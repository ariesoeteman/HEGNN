import pdb

import torch


class ZincEncoder(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(ZincEncoder, self).__init__()
        self.enc = torch.nn.Embedding(21, hidden_dim)

    # NOTE can try different variants here
    def forward(self, x):
        first_column = x[:, 0].long()
        encoded = torch.hstack([self.enc(first_column), x[:, 1:]])
        return encoded
