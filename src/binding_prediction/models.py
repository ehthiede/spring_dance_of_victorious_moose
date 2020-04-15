import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from .layers import GraphAndConv, MergeSnE1, RankingLayer
from dgl.nn.pytorch.conv import GraphConv
from binding_prediction.dataset import build_dgl_graph_batch


class BindingModel(torch.nn.Module):
    """
    Model for predicting weather or not we achieve binding.

    Parameters
    ----------
    in_channels : int
        The size of the channel index of the input tensor
    hidden_channel_list : iterable of ints
        Number of channels in every hidden layer of the encoder
        Length corresponds to the number of hidden layers.
    out_channels : int
        number of output channels.
    conv_kernel_sizes : iterable of ints, optional
        Number of channels in every hidden layer of the encoder
        Length corresponds to the number of layers.
        If not provided, defaults to 1 for every layer
    """
    def __init__(self, in_channels_graph, in_channels_prot,
                 merge_channels_graph, merge_channels_prot,
                 hidden_channel_list, out_channels, final_channels=1,
                 conv_kernel_sizes=None, nonlinearity=None, layer_cls=GraphAndConv):
        super(BindingModel, self).__init__()
        self.in_graph = nn.Linear(in_channels_graph, merge_channels_graph)
        self.in_prot = nn.Linear(in_channels_prot, merge_channels_prot)
        self.gcs_stack = GraphAndConvStack(merge_channels_graph + merge_channels_prot,
                                           hidden_channel_list,
                                           out_channels, conv_kernel_sizes,
                                           nonlinearity, layer_cls)
        self.merge_graph_w_sequences = MergeSnE1()
        total_number_inputs = sum(hidden_channel_list) + out_channels
        self.final_mix = nn.Linear(total_number_inputs, final_channels, bias=False)
        self.lm = None

    def forward(self, adj, x, prot_sequences):
        if self.lm is None:
            raise ValueError('Language model is not initialized!')
        prot_embeddings = [self.lm(p_i) for p_i in prot_sequences]
        embeddings = pad_sequence(prot_embeddings, batch_first=True, padding_value=0)

        x_node = self.in_graph(x)
        x_prot = self.in_prot(embeddings)
        y = self.merge_graph_w_sequences(x_node, x_prot)
        x_all = self.gcs_stack(adj, y)
        x_all = torch.cat(x_all, dim=-1)
        x_out = self.final_mix(x_all)
        x_out = torch.sum(torch.sum(x_out, dim=2), dim=1)
        return x_out

    def load_language_model(self, cls, path, device='cuda'):
        """
        Parameters
        ----------
        cls : Module name
            Name of the Language model.
            (i.e. binding_prediction.language_model.Elmo)
        path : filepath
            Filepath of the pretrained model.
        """
        self.lm = cls(path, device=device)


class PosBindingModel(nn.Module):

    def __init__(self, input_size, emb_dim, **kwargs):
        super(PosBindingModel, self).__init__()

        self.bm = BindingModel(**kwargs)
        self.ranking = RankingLayer(input_size, emb_dim)

    def predict(self, pos_adj, pos_x, neg_adj, neg_x, prot_sequences):
        pos_out = self.bm.forward(pos_adj, pos_x, prot_sequences)
        neg_out = self.bm.forward(neg_adj, neg_x, prot_sequences)
        score = self.ranking.forward(pos_out, neg_out)
        return score

    def forward(self, *args, **kwargs):
        score = self.predict(*args, **kwargs)
        loss = sum(sum(F.logsigmoid(score)))
        return -1 * loss

    def load_language_model(self, cls, path, device='cuda'):
        """
        Parameters
        ----------
        cls : Module name
            Name of the Language model.
            (i.e. binding_prediction.language_model.Elmo)
        path : filepath
            Filepath of the pretrained model.
        """
        self.bm.lm = cls(path, device=device)


class GraphAndConvStack(nn.Module):
    """
    Stack of graph and E(1) convolutional layers.

    Parameters
    ----------
    in_channels : int
        The size of the channel index of the input tensor
    hidden_channel_list : iterable of ints
        Number of channels in every hidden layer of the encoder
        Length corresponds to the number of hidden layers.
    out_channels : int
        number of output channels.
    conv_kernel_sizes : iterable of ints, optional
        Number of channels in every hidden layer of the encoder
        Length corresponds to the number of layers.
        If not provided, defaults to 1 for every layer
    """

    def __init__(self, in_channels, hidden_channel_list, out_channels,
                 conv_kernel_sizes=None, nonlinearity=None, layer_cls=GraphAndConv):
        super(GraphAndConvStack, self).__init__()
        if nonlinearity is None:
            nonlinearity = nn.ReLU
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [1] * (len(hidden_channel_list) + 1)
        self.nonlinearity = nonlinearity()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channel_list = hidden_channel_list

        self.conv_layers = nn.ModuleList()

        in_channel_i = in_channels
        for i, out_channel_i in enumerate(hidden_channel_list):
            conv_i = layer_cls(in_channel_i, out_channel_i,
                                  conv_kernel_size=conv_kernel_sizes[i])
            self.conv_layers.append(conv_i)
            in_channel_i = out_channel_i
        conv_i = layer_cls(in_channel_i, out_channels,
                              conv_kernel_size=conv_kernel_sizes[-1])
        self.conv_layers.append(conv_i)

    def forward(self, adj, x):
        """
        Runs the net.

        Parameters
        ----------
        x : pytorch tensor
            Input to the convolution, of shape B x N x L x C_in
            Here B is the batch size, C_in is the number of channels,
            N is the size of the largest graph, and L is the size of the E(1)
            equivariant string.

        Returns
        -------
        x_all : list of pytorch tensor
            The output of the net at every layer of the net.
        """
        x = self.conv_layers[0](adj, x)
        x_all = [x]

        for module in self.conv_layers[1:]:
            x = self.nonlinearity(x)
            x = module(adj, x)
            x_all.append(x)

        return x_all


class DecomposableAttentionModel(nn.Module):
    def __init__(self, node_dim, residue_dim, merge_channels_graph, merge_channels_prot, num_gnn_steps):
        super().__init__()
        self.residue_dim = residue_dim
        self.merge_channels_graph = merge_channels_graph
        self.merge_channels_prot = merge_channels_prot
        self.node_compression_layer = nn.Linear(node_dim, merge_channels_graph)
        self.residue_compression_layer = nn.Linear(self.residue_dim, merge_channels_prot)
        self.gnn = GraphConv(merge_channels_graph, merge_channels_graph)
        self.num_gnn_steps = num_gnn_steps
        self.attn_weight_layer = nn.Sequential(
            nn.Linear(merge_channels_graph+merge_channels_prot, 1),
            nn.Softmax(dim=-1))
        self.interaction_layer = nn.Sequential(
            nn.Linear(merge_channels_graph+merge_channels_prot,
                        merge_channels_graph + int(merge_channels_prot/2)),
            nn.ReLU(),
            nn.Linear(merge_channels_graph + int(merge_channels_prot/2), merge_channels_graph)
        )
        self.output_layer = nn.Linear(merge_channels_graph, 1)

        self.merge_graph_w_sequences = MergeSnE1()
        self.lm = None

    def load_language_model(self, cls, path, device='cuda'):
        self.lm = cls(path, device=device)


    def forward(self, adj_mats, nodes, protein_sequences):
        if self.lm is None:
            raise ValueError('Language model is not initialized!')
        protein_sequences = [self.lm(p_i) for p_i in protein_sequences]
        batch_size, max_nodes, node_dim = nodes.shape
        batch_graph = build_dgl_graph_batch(nodes, adj_mats)
        nodes = batch_graph.ndata.pop('features')
        nodes = self.node_compression_layer(nodes)
        for i in range(self.num_gnn_steps):
            nodes = self.gnn(batch_graph, nodes)
            activation = torch.relu if i < self.num_gnn_steps - 1 else torch.tanh
            nodes = activation(nodes)
        nodes = nodes.reshape(batch_size, max_nodes, -1)
        protein_sequences = pad_sequence(protein_sequences, batch_first=True, padding_value=0)
        protein_sequences = self.residue_compression_layer(protein_sequences)
        node_residue_cat = self.merge_graph_w_sequences(nodes, protein_sequences)
        seq_length = node_residue_cat.shape[2]
        node_residue_cat = node_residue_cat.reshape(batch_size, max_nodes * seq_length,
                            self.merge_channels_graph + self.merge_channels_prot)
        attn_weights = self.attn_weight_layer(node_residue_cat)
        node_residue_interactions = self.interaction_layer(node_residue_cat)
        weighted_sum = (attn_weights * node_residue_interactions).sum(dim=1)
        score = self.output_layer(weighted_sum)
        return score

MODELS_DICT = {'BindingModel': BindingModel,
               'DecomposableAttentionModel':DecomposableAttentionModel}
