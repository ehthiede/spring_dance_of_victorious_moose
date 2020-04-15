import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv
import math
from .utils import _calc_padding, _unpack_from_convolution, _pack_for_convolution
from binding_prediction.dataset import build_dgl_graph_batch


class GraphAndConvParent(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=1, intermediate_channels=None):
        super().__init__()
        if intermediate_channels is None:
            intermediate_channels = out_channels
        self.lin = nn.Linear(2*in_channels, intermediate_channels)
        padding = _calc_padding(1, conv_kernel_size)
        self.conv = nn.Conv1d(intermediate_channels, out_channels, conv_kernel_size, padding=padding)

        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        self.out_channels = out_channels

class GraphAndConv(GraphAndConvParent):
    def __init__(self, in_channels, out_channels, conv_kernel_size=1, intermediate_channels=None):
        super().__init__(in_channels, out_channels, conv_kernel_size, intermediate_channels)

    def forward(self, adj, inputs):
        batch_size = inputs.shape[0]
        mask = adj.sum(dim=2).bool()
        x = torch.einsum('bilc,bij->bjlc', inputs, adj)
        x = torch.cat((x, inputs), dim=-1)
        x = self.lin(x)
        x = _pack_for_convolution(x)
        x = self.conv(x)
        x = _unpack_from_convolution(x, batch_size)
        x[~mask] = 0.
        return x

class GraphAndConvDGL(GraphAndConvParent):
    def __init__(self, in_channels, out_channels, conv_kernel_size=1, intermediate_channels=None):
        super().__init__(in_channels, out_channels, conv_kernel_size, intermediate_channels)
        self.gnn = GraphConv(in_channels, in_channels)

    def forward(self, adj, inputs):
        batch_size = inputs.shape[0]
        g = build_dgl_graph_batch(inputs, adj)
        inputs = g.ndata['features']
        mask = adj.sum(dim=2).bool()

        x = self.gnn(g, inputs)
        x = torch.cat((x, inputs), dim=-1)
        x = self.lin(x)
        x = torch.transpose(x, 1, 2)
        x = self.conv(x)
        x = _unpack_from_convolution(x, batch_size)
        x[~mask] = 0.
        return x

class MergeSnE1(nn.Module):
    def __init__(self):
        super(MergeSnE1, self).__init__()

    def forward(self, features, embedding):
        N_resid = embedding.shape[1]
        N_nodes = features.shape[1]
        nodes_expanded = torch.stack([features] * N_resid, dim=2)
        embed_expanded = torch.stack([embedding] * N_nodes, dim=1)
        full_features = torch.cat((nodes_expanded, embed_expanded), dim=3)
        return full_features


class RankingLayer(nn.Module):
    def __init__(self, input_size, emb_dim):
        """ Initialize model parameters for Siamese network.
        Parameters
        ----------
        input_size: int
            Input dimension size
        emb_dim: int
            Embedding dimension for both datasets
        Note
        ----
        This implicitly assumes that the embedding dimension for
        both datasets are the same.
        """
        # See here: https://adoni.github.io/2017/11/08/word2vec-pytorch/
        super(RankingLayer, self).__init__()
        self.input_size = input_size
        self.emb_dimension = emb_dim
        self.output = nn.Linear(input_size, emb_dim)
        self.init_emb()

    def init_emb(self):
        initstd = 1 / math.sqrt(self.emb_dimension)
        self.output.weight.data.normal_(0, initstd)

    def forward(self, pos, neg):
        """
        Parameters
        ----------
        pos : torch.Tensor
           Positive shared representation vector
        neg : torch.Tensor
           Negative shared representation vector(s).
           There can be multiple negative examples (~5 according to NCE).
        """
        losses = 0
        pos_out = self.output(pos)
        neg_out = self.output(neg)
        diff = pos_out - neg_out
        return diff
