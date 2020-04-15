import torch
import pytest
from binding_prediction.models import GraphAndConvStack, BindingModel, DecomposableAttentionModel
from binding_prediction import pretrained_language_models
from binding_prediction.layers import GraphAndConv, GraphAndConvDGL


def _permute_tensors(features, adj_mats, p_indices):
    permuted_adj_mats = []
    permuted_feature_mats = []
    for i, pidx in enumerate(p_indices):
        permuted_adj_mats.append(adj_mats[i][pidx][:, pidx])
        permuted_feature_mats.append(features[i][pidx])
    permuted_adj_mats = torch.stack(permuted_adj_mats, dim=0)
    permuted_feature_mats = torch.stack(permuted_feature_mats, dim=0)
    return permuted_feature_mats, permuted_adj_mats


class TestGraphAndConvStack(object):
    @pytest.mark.parametrize('num_intermediate', [None, 4])
    @pytest.mark.parametrize('hidden_channel_list', [[3], [2, 2], []])
    @pytest.mark.parametrize('explicit_conv_kernel_sizes', [True, False])
    @pytest.mark.parametrize('layer_cls', [GraphAndConv, GraphAndConvDGL])
    def test_permutation_equivariance(self, sample_batch, num_intermediate,
                                      hidden_channel_list, explicit_conv_kernel_sizes, layer_cls):
        features, adj_mats = sample_batch
        B, N, __ = adj_mats.shape
        p_indices = [torch.randperm(N) for i in range(B)]

        feature_perm, adj_mats_perm = _permute_tensors(features, adj_mats, p_indices)

        if explicit_conv_kernel_sizes:
            conv_kernel_sizes = list(range(1, 2 * (len(hidden_channel_list) + 2), 2))
        else:
            conv_kernel_sizes = None
        gconv = GraphAndConvStack(3, hidden_channel_list, 1,
                                  conv_kernel_sizes=conv_kernel_sizes, layer_cls=layer_cls)
        output = gconv(adj_mats, features)[-1]
        permed_output = _permute_tensors(output, adj_mats, p_indices)[0]
        output_from_perm = gconv(adj_mats_perm, feature_perm)[-1]
        assert(torch.norm(permed_output - output_from_perm) < 1e-4)

    @pytest.mark.parametrize('hidden_channel_list', [[3], [2, 2], []])
    @pytest.mark.parametrize('layer_cls', [GraphAndConv, GraphAndConvDGL])
    def test_translational_equivariance(self, sample_batch, hidden_channel_list, layer_cls):
        features, adj_mats = sample_batch
        features[:, :, -1] = 0.
        trans_features = torch.zeros(features.shape)
        trans_features[:, :, 1:] = features[:, :, :-1]

        gconv = GraphAndConvStack(3, hidden_channel_list, 1, layer_cls=layer_cls)
        output = gconv(adj_mats, features)[-1]
        output_from_translation = gconv(adj_mats, trans_features)[-1]
        assert(torch.norm(output[:, :, 4:-5] - output_from_translation[:, :, 5:-4]) < 1e-4)


class TestBindingModel(object):
    def test_permutation_invariance(self, sample_sequences):
        node_features = torch.randn(3, 13, 4)

        adj_mats = torch.randint(2, size=(3, 13, 13)).float()

        B, N, __ = adj_mats.shape
        p_indices = [torch.randperm(N) for i in range(B)]

        feature_perm, adj_mats_perm = _permute_tensors(node_features, adj_mats, p_indices)

        model = BindingModel(4, 512, 3, 3, [3], 3)
        cls, path = pretrained_language_models['elmo']
        model.load_language_model(cls, path, device='cpu')
        output = model(adj_mats, node_features, sample_sequences)
        output_from_perm = model(adj_mats_perm, feature_perm, sample_sequences)
        assert(torch.norm(output - output_from_perm) < 1e-3)

class TestPosBindingModel(object):
    def test_forward(self):
        # torch.Size([20, 50, 86]) torch.Size([20, 50, 50])
        # torch.Size([20, 31, 86]) torch.Size([20, 31, 31])
        # 20
        torch.manual_seed(0)
        pos_adj = torch.rand((20, 50, 86))
        pos_x = torch.rand((20, 50, 50))
        neg_adj = torch.rand((20, 31, 86))
        neg_x = torch.rand((20, 31, 31))
        lengths = [477, 777, 80, 232, 406, 433, 1291,
                   533, 858, 724, 360, 348, 298, 816,
                   719, 417, 680, 527, 506, 542]
        prots = list(map(lambda x: x*'A', lengths))
        out_channels=10
        in_channels_nodes = 20
        in_channels_seq = 512
        merge_molecule_channels = 10
        merge_prot_channels = 10
        hidden_channels = 10
        model = PosBindingModel(
            out_channels, out_channels,
            in_channels_graph=in_channels_nodes, in_channels_prot=in_channels_seq,
            merge_channels_graph=args.merge_molecule_channels,
            merge_channels_prot=args.merge_prot_channels,
            hidden_channel_list=args.hidden_channels,
            out_channels=out_channels, final_channels=out_channels)
        out = model(pos_adj, pos_x, neg_adj, neg_x, prots)
        self.assertTrue(0)

class TestAttentionModel(object):
    def test_permutation_invariance(self, sample_sequences):
        batch_size, node_dim, max_nodes = 3, 5, 4
        node_features = torch.randn(batch_size, max_nodes, node_dim)

        adj_mats = torch.randint(2, size=(batch_size, max_nodes, max_nodes)).float()

        p_indices = [torch.randperm(max_nodes) for i in range(batch_size)]

        feature_perm, adj_mats_perm = _permute_tensors(node_features, adj_mats, p_indices)

        cls, path = pretrained_language_models['elmo']
        model = DecomposableAttentionModel(node_dim, 512, 2, 2, num_gnn_steps=3)
        model.load_language_model(cls, path, device='cpu')
        output = model(adj_mats, node_features, sample_sequences)
        output_from_perm = model(adj_mats_perm, feature_perm, sample_sequences)
        assert(torch.norm(output - output_from_perm) < 1e-2)
