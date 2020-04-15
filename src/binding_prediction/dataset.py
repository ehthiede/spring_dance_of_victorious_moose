import dgl
from torch.utils.data import Dataset
from rdkit.Chem import rdmolops
from rdkit import Chem
from .utils import onehot
import math
import torch
import numpy as np
import scipy.sparse as sps
import pandas as pd


all_elements = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na",
                "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti",
                "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
                "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru",
                "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs",
                "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
                "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os",
                "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"]


def _load_datafile(datafile):
    """
    Parameters
    ----------
    datafile : str
       File path.  Assumes that the file is only two columns
          1. Drug INCHI string
          2. Protein sequence
    """
    df = pd.read_table(datafile, header=None, delim_whitespace=True)
    drug_inchi = np.array(df[0])  # first column is inchi key
    protein_seqs = np.array(df[1])  # second column is sequence
    return drug_inchi, protein_seqs


class DrugProteinDataset(Dataset):
    """
    Database containing drugs, protein IDs, and whether or not they bind.
    """

    def __init__(self, datafile, multiple_bond_types=False,
                 precompute=True, transform=None, prob_fake=0.0, fake_dist=None):
        """
        Args:
            datafile (string) : Data file that has the protein sequences, smiles strings,
            and the binding type (FIX?)
        protein_embedding_folder (string) : Template for file containing embeddings.
        """
        super(DrugProteinDataset, self).__init__()
        self.all_drugs, self.all_prots = _load_datafile(datafile)
        self.unique_prots, prot_invs = np.unique(self.all_prots, return_inverse=True)
        self.unique_drugs, drug_invs = np.unique(self.all_drugs, return_inverse=True)
        self.multiple_bond_types = multiple_bond_types
        self.num_edge_features = 3
        self.prob_fake = prob_fake

        if fake_dist is None:
            def fake_dist():
                smiles_idx = np.random.randint(len(self.all_drugs))
                prot_idx = np.random.randint(len(self.all_prots))
                return smiles_idx, prot_idx
        self.fake_dist = fake_dist

        # Build interaction matrix
        # self._build_interaction_matrix(drug_invs, prot_invs)
        self._build_dataset_info()
        self.precompute = precompute
        if precompute:
            self.compute_graphs()

        self.transform = transform

    def __len__(self):
        return len(self.all_drugs)

    def _preprocess_molecule(self, smiles):
        if self.precompute:
            nodes, edges = self.drug_graphs[smiles]
        else:
            nodes, edges, mol = self._build_drug_graph(smiles)
        adj = self._graph_to_adj_mat(edges, len(nodes))
        if not self.multiple_bond_types:
            adj = torch.sum(adj, dim=2)
        return nodes, adj

    def __getitem__(self, idx):
        is_true = (np.random.rand() < (1. - self.prob_fake))
        if is_true:
            drug_idx = prot_idx = idx
        else:
            drug_idx, prot_idx = self.fake_dist()
        smiles = self.all_drugs[drug_idx]
        prot = self.all_prots[prot_idx]

        nodes, adj_mat = self._preprocess_molecule(smiles)

        sample = {'node_features': nodes, 'adj_mat': adj_mat,
                  'protein': prot, 'is_true': int(is_true)}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _build_interaction_matrix(self, drug_invs, prot_invs):
        M = np.max(prot_invs) + 1
        N = np.max(drug_invs) + 1
        self.interaction_matrix = sps.csr(np.ones(drug_invs.shape),
                                          (prot_invs, drug_invs), shape=(M, N))

    def compute_graphs(self):
        self.drug_graphs = {}
        for drug_smiles in self.unique_drugs:
            nodes, edges, __ = self._build_drug_graph(drug_smiles)
            self.drug_graphs[drug_smiles] = (nodes, edges)

    def _build_drug_graph(self, smiles):
        """
        Builds a molecular graph form a smiles string.  Taken from [FIND SOURCE!]
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError('Molecule construction failed on Smiles %s' % smiles)
        # Kekulize it
        if self.need_kekulize(mol):
            rdmolops.Kekulize(mol)
            if mol is None:
                return None, None
        # remove stereo information, such as inward and outward edges
        Chem.RemoveStereochemistry(mol)

        edges = []
        nodes = []
        for bond in mol.GetBonds():
            edges.append((bond.GetBeginAtomIdx(), self.bond_dict[str(bond.GetBondType())],
                          bond.GetEndAtomIdx()))
            assert self.bond_dict[str(bond.GetBondType())] != 3
        for atom in mol.GetAtoms():
            # This could probably be spead up....
            nodes.append(onehot(self.dataset_info['atom_types'].index(atom.GetSymbol()),
                                len(self.dataset_info['atom_types'])))

        nodes = torch.tensor(nodes).float()
        edges = torch.tensor(edges)

        return nodes, edges, mol

    def _build_dataset_info(self):
        # self.dataset_info = {'atom_types': ["H", "C", "N", "O", "S", "P", "F",
        #     "Cl", "Br", "I", "Fe", "B", "Ce", "As", "Se", "K", "Mg", "Ca", "Mo", "Hg", "Na", "Pt", "Cd", "Sn", "Co", "Re", "Si", "Al"]
        #                      }
        self.dataset_info = {'atom_types': all_elements}

        self.bond_dict = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, "AROMATIC": 3}

    def _graph_to_adj_mat(self, edge_features, num_atoms):
        adj_mat = torch.zeros(num_atoms, num_atoms, self.num_edge_features)
        if len(edge_features) > 0:
            edge_features = torch.LongTensor(edge_features)
            adj_mat[edge_features[:, 0], edge_features[:, 2], edge_features[:, 1]] = 1.
            adj_mat = adj_mat.float()
        return adj_mat

    def need_kekulize(self, mol):
        """
        Check if a molecule needs kekulization.  Taken from [FIND SOURCE!]
        """
        for bond in mol.GetBonds():
            if self.bond_dict[str(bond.GetBondType())] >= 3:
                return True
        return False


class PosDrugProteinDataset(DrugProteinDataset):
    """ Performs positive only sampling strategy.

    Notes
    -----
    Bayesian Personalized Ranking:
       https://arxiv.org/pdf/1205.2618.pdf
    """
    def __init__(self, num_neg=1, **kwargs):
        super(PosDrugProteinDataset, self).__init__(**kwargs)
        self.num_neg = num_neg

    def __getitem__(self, idx):

        drug_pos = prot_idx = idx
        smiles_neg = np.random.choice(self.all_drugs)
        smiles_pos = self.all_drugs[drug_pos]
        prot = self.all_prots[prot_idx]
        pos_nodes, pos_adj = self._preprocess_molecule(smiles_pos)
        neg_nodes, neg_adj = self._preprocess_molecule(smiles_neg)

        sample = {'pos_node_features': pos_nodes, 'pos_adj_mat': pos_adj,
                  'neg_node_features': neg_nodes, 'neg_adj_mat': neg_adj,
                  'protein': prot}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __iter__(self):
        """ Need this for multi-GPU support"""
        worker_info = torch.utils.data.get_worker_info()
        start = 0
        end = len(self.pairs)

        if worker_info is None:  # single-process data loading
            for i in range(end):
                for _ in range(self.num_neg):
                    yield self.__getitem__(i)
        else:
            worker_id = worker_info.id
            w = float(worker_info.num_workers)
            t = (end - start)
            w = float(worker_info.num_workers)
            per_worker = int(math.ceil(t / w))
            worker_id = worker_info.id
            iter_start = start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, end)
            for i in range(iter_start, iter_end):
                for _ in range(self.num_neg):
                    yield self.__getitem__(i)

class ComparisonDrugProteinDataset(DrugProteinDataset):
    def __getitem__(self, idx):
        # true sample
        smiles = self.all_drugs[idx]
        prot = self.all_prots[idx]

        nodes, adj_mat = self._preprocess_molecule(smiles)

        sample = {'node_features': nodes, 'adj_mat': adj_mat, 'smiles': smiles,
                  'protein': prot, 'is_true': 1}

        if self.transform:
            sample = self.transform(sample)

        # fake sample 1
        smiles = self.all_drugs[idx]
        interacting_proteins = self.all_prots[np.where(self.all_drugs == smiles)]
        non_interacting_proteins = np.setdiff1d(self.unique_prots, interacting_proteins)
        prot = np.random.choice(non_interacting_proteins, 1)[0]

        same_drug_other_prot_sample = {'node_features': nodes, 'adj_mat': adj_mat,
                                       'smiles': smiles, 'protein': prot, 'is_true': 0}

        # fake sample 2
        prot = self.all_prots[idx]
        interacting_drugs = self.all_drugs[np.where(self.all_prots == prot)]
        non_interacting_drugs = np.setdiff1d(self.unique_drugs, interacting_drugs)
        smiles = np.random.choice(non_interacting_drugs, 1)[0]
        nodes, adj_mat = self._preprocess_molecule(smiles)

        same_prot_other_drug_sample = {'node_features': nodes, 'adj_mat': adj_mat,
                                       'smiles': smiles, 'protein': prot, 'is_true': 0}

        return (sample, same_drug_other_prot_sample, same_prot_other_drug_sample)


def collate_fn(batch, prots_are_sequences=False):
    collated_batch = {}
    for prop in batch[0].keys():
        if isinstance(batch[0][prop], str):
            sequence_list = [mol[prop] for mol in batch]
            collated_batch[prop] = sequence_list
        else:
            is_adj_mat = ('adj_mat' in prop)
            collated_batch[prop] = _batch_stack([mol[prop] for mol in batch], edge_mat=is_adj_mat)

    return collated_batch

def collate_fn_triplet(triplet_batch, prots_are_sequences=False):
    transposed_triplet_batch = list(zip(*triplet_batch))
    real_batch = collate_fn(transposed_triplet_batch[0], prots_are_sequences)
    fake_batch_1 = collate_fn(transposed_triplet_batch[1], prots_are_sequences)
    fake_batch_2 = collate_fn(transposed_triplet_batch[2], prots_are_sequences)
    return (real_batch, fake_batch_1, fake_batch_2)


def _batch_stack(props, edge_mat=False):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.  Adapted from the cormorant library, and
    initially written by Brandon Anderson.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack
    edge_mat : bool
        The included tensor refers to edge properties, and therefore needs
        to be stacked/padded along two axes instead of just one.

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        props = [torch.tensor(pi) for pi in props]
    if props[0].dim() == 0:
        return torch.stack(props)
    elif not edge_mat:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)
    else:
        max_atoms = max([len(p) for p in props])
        max_shape = (len(props), max_atoms, max_atoms) + props[0].shape[2:]
        padded_tensor = torch.zeros(max_shape, dtype=props[0].dtype, device=props[0].device)

        for idx, prop in enumerate(props):
            this_atoms = len(prop)
            padded_tensor[idx, :this_atoms, :this_atoms] = prop

        return padded_tensor


def build_dgl_graph(features, adj_mat):
    g = dgl.DGLGraph()
    N_nodes = features.shape[0]
    g.add_nodes(N_nodes, {'features': features})
    nonzero_coords = torch.nonzero(adj_mat)
    u, v = nonzero_coords[:, 1], nonzero_coords[:, 0]
    g.add_edges(u, v)
    return g


def build_dgl_graph_batch(batch_features, batch_adj_mat):
    graphs = []
    for i in range(len(batch_adj_mat)):
        graphs.append(build_dgl_graph(batch_features[i], batch_adj_mat[i]))
    graphs_batch = dgl.batch(graphs)
    return graphs_batch
