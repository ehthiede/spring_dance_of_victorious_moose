import torch
import argparse
import os
import pickle
import datetime
from torch import nn
from binding_prediction.models import BindingModel
from binding_prediction.dataset import DrugProteinDataset, collate_fn
from binding_prediction.summary import initialize_logging
from binding_prediction.model_utils import run_model_on_batch, get_targets
from binding_prediction import pretrained_language_models
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from binding_prediction.models import MODELS_DICT
from binding_prediction.evaluate import roc_auc
import gc

def _parse_args():
    parser = argparse.ArgumentParser(description='Train on a binding database.')
    parser.add_argument('--num-epoch', '-e', type=int, default=256,
                        help='Number of epochs to train (default: 255)')
    parser.add_argument('--dir', '-d', type=str, default='.', help='Directory to which to save the model.')
    parser.add_argument('--batch-size', '-b', type=int, default=20,
                        help='Mini-batch size (default: 25)')
    parser.add_argument('--train_dataset', '-t', type=str, default='data/molecules_train_qm9.json',
                        help='Training dataset')
    parser.add_argument('--valid_dataset', '-v', type=str, default='data/molecules_valid_qm9.json',
                        help='Validation dataset')
    parser.add_argument('--model_name', choices=['DecomposableAttentionModel', 'BindingModel'],
                        default='BindingModel')
    parser.add_argument('--merge_molecule_channels', '-m', type=int, default=10,
                        help='Number of channels to use in the hidden layers')
    parser.add_argument('--merge_prot_channels', '-p', type=int, default=10,
                        help='Number of channels to use in the hidden layers')
    parser.add_argument('--hidden_channels', '-c', nargs='*', type=int, default=[10],
                        help='Number of channels to use in the hidden layers')
    parser.add_argument('--conv_kernel_sizes', '-k', nargs='*', type=int, default=None,
                        help='Number of channels to use in the hidden layers')
    parser.add_argument('--num_gnn_steps', '-n', type=int, default=3,
                        help='Number of times to pass graph through GNN')
    parser.add_argument('--learning_rate', '-l', type=float, default=1e-3,
                        help='Learning Rate')
    parser.add_argument('--lmarch', '-a', type=str, default='elmo',
                        help='Language Model Architecture')
    parser.add_argument('--cuda', dest='cuda', action='store_true',
                        help='Use CUDA (default)')
    parser.add_argument('--no-cuda', '--cpu', dest='cuda', action='store_false',
                        help='Use CPU')
    parser.set_defaults(cuda=True)
    args = parser.parse_args()
    return args


def main():
    args = _parse_args()

    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)
    writer = initialize_logging(args.dir, '/logs/')

    # Save the construction arguments for future reference.
    with open(args.dir + '/training_args.pkl', 'wb') as f:
        pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)

    lm, path = pretrained_language_models[args.lmarch]

    train_dataset = DrugProteinDataset(args.train_dataset, prob_fake=0.5, precompute=False)
    valid_dataset = DrugProteinDataset(args.valid_dataset, prob_fake=0.5, precompute=False)

    cfxn = lambda x: collate_fn(x, prots_are_sequences=True)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=cfxn)
    valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=True, collate_fn=cfxn)

    loss_fxn = nn.BCEWithLogitsLoss()

    in_channels_nodes = train_dataset[0]['node_features'].shape[-1]
    in_channels_seq = 512
    out_channels = 1
    model_cls = MODELS_DICT.get(args.model_name)
    if args.model_name == 'BindingModel':
        model = BindingModel(in_channels_nodes, in_channels_seq, args.merge_molecule_channels,
                             args.merge_prot_channels, args.hidden_channels, out_channels)
    elif args.model_name == 'DecomposableAttentionModel':
        model = model_cls(in_channels_nodes, in_channels_seq, args.merge_molecule_channels,
                          args.merge_prot_channels, args.num_gnn_steps)
    model = model.to(device=device)
    model.load_language_model(lm, path)
    writer.add_text("Log", "Initialized Model.")

    if os.path.isfile(args.dir + '/model_best.pt'):
        writer.add_text("Log", "Previous Model found.  Attempting to load previous best model...")
        model_param_dict = torch.load(args.dir + '/model_best.pt')
        model.load_state_dict(model_param_dict)
        writer.add_text("Log", "Succesfully loaded previous model")

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_valid_loss = 1E8

    for n in range(args.num_epoch):
        model.train()
        total_train_loss = 0
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = run_model_on_batch(model, batch, device=device).squeeze(-1)
            targets = get_targets(batch, device)
            loss = loss_fxn(output, targets)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                l = loss.item()
                print("Batch {}/{}.  Batch loss: {}".format(i, len(train_dataloader), l))
                total_train_loss += l
                torch.cuda.empty_cache()

        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(valid_dataloader):
                output = run_model_on_batch(model, batch, device=device).squeeze(-1)
                targets = get_targets(batch, device)
                loss = loss_fxn(output, targets)
                total_valid_loss += loss.item()
        auc = roc_auc(model, valid_dataloader, 'nce', n, writer)

        avg_train_loss = total_train_loss / len(train_dataset)
        avg_valid_loss = total_valid_loss / len(valid_dataset)
        print("Epoch {} Complete. Train loss: {}.  Valid loss: {}. AUC: {}".format(
            n, avg_train_loss, avg_valid_loss, auc))
        writer.add_scalar('training_loss', avg_train_loss, n)
        writer.add_scalar('validation_loss', avg_valid_loss, n)

        torch.save(model.state_dict(), args.dir + '/model_current.pt')
        if avg_valid_loss < best_valid_loss:
            writer.add_text("Log", "Best validation loss achieved at %d." % n)
            torch.save(model.state_dict(), args.dir + '/model_best.pt')
            best_valid_loss = avg_valid_loss


if __name__ == "__main__":
    main()
