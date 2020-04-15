import torch


def run_model_on_batch(model, batch, device='cuda'):
    adj_mat = batch['adj_mat'].to(device=device)
    features = batch['node_features'].to(device=device)
    sequences = batch['protein']
    out_features = model(adj_mat, features, sequences)
    return out_features


def run_model_on_mixed_batch(model, batch, device='cuda'):
    pos_nodes = batch['pos_node_features'].to(device=device)
    pos_adj = batch['pos_adj_mat'].to(device=device)
    neg_nodes = batch['neg_node_features'].to(device=device)
    neg_adj = batch['neg_adj_mat'].to(device=device)
    sequences = batch['protein']
    return pos_adj, pos_nodes, neg_adj, neg_nodes, sequences

def run_bpr_on_batch(model, batch, device='cuda'):
    res = run_model_on_mixed_batch(model, batch, device='cuda')
    loss = model(*res)
    return loss


def get_targets(batch, device):
    return batch['is_true'].to(device=device).float()
