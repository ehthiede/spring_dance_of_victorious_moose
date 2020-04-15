import numpy as np
import pandas as pd
import torch
from binding_prediction.model_utils import (run_model_on_batch,
                                            run_model_on_mixed_batch, get_targets)
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from itertools import product


# Global evaluation metrics
# these are used mainly for testing evaluation
def mrr(model, dataloader):
    """ Mean reciprocial ranking.

    Parameters
    ----------
    model : popular.model
       Model to be evaluated
    dataloader : torch.DataLoader
       Pytorch dataloader.

    Returns
    -------
    float : mean reciprocial ranking
    """
    pass

def roc_auc(binding_model, dataloader, name, it, writer, device='cuda'):
    """ ROC AUC

    Parameters
    ----------
    model : popular.model
       Model to be evaluated
    dataloader : torch.DataLoader
       Pytorch dataloader for validation data

    Returns
    -------
    float : Area under the curve
    """
    outs, tars = [], []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            output = run_model_on_batch(binding_model, batch, device=device).squeeze(-1)
            targets = get_targets(batch, device)
            out = output.cpu().detach().numpy().ravel()
            tar = targets.cpu().detach().numpy().ravel()
            outs += list(out)
            tars += list(tar)

    fpr, tpr, thresholds = roc_curve(tars, outs)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.set_xlabel('fpr')
    ax.set_ylabel('tpr')
    writer.add_figure(f'{name}/AUC', fig, it)
    auc = roc_auc_score(tars, outs)
    return auc


def pairwise_auc(binding_model,
                 dataloader, name, it, writer):
    """ Pairwise AUC comparison

    Parameters
    ----------
    binding_model : binding model
       Binding prediction model.
    dataloader : dataloader
       Dataset iterator for test/validation ppi dataset.
    name : str
       Name of the database used in dataloader.
    it : int
       Iteration number.
    writer : SummaryWriter
       Tensorboard writer.
    device : str
       Device name to transfer model data to.

    Returns
    -------
    float : average AUC

    Notes
    -----
    This assumes that the dataloader can return positive / negative samples.
    """
    with torch.no_grad():
        rank_counts = 0
        total = 0
        for j, batch in enumerate(dataloader):
            res = run_model_on_mixed_batch(binding_model, batch)
            pa, pn, na, nn, s = res
            v = binding_model.predict(pa, pn, na, nn, s)
            score = torch.sum(v > 0).item()
            rank_counts += score
            total += len(v)

        tpr = rank_counts / total
        writer.add_scalar('pairwise_auc', tpr, it)

    return tpr
