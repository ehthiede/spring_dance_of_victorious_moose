from torch.utils.data import Dataset
from rdkit.Chem import rdmolops
from rdkit import Chem
import torch
import numpy as np
import scipy.sparse as sps
import pandas as pd


class DrugProteinDataset(Dataset):
    """
    Database containing drugs, protein IDs, and whether or not they bind.
    """

    def __init__(self, datafile, protein_embedding_template, multiple_bond_types=False,
                 precompute=True, transform=None):
        """
        Args:
            datafile (string) : Data file that has the uniprot ids, smiles strings,
            and the binding type (FIX?)
        protein_embedding_folder (string) : Template for file containing embeddings.
        """
        super(DrugProteinDataset, self).__init__()
        self.all_drugs, self.all_uniprot_ids = self._load_datafile(datafile)
        self.protein_embedding_template = protein_embedding_template
        self.unique_uniprot_ids, prot_invs = np.unique(self.all_uniprot_ids, return_inverse=True)
        self.unique_drugs, drug_invs = np.unique(self.all_drugs, return_inverse=True)
        self.multiple_bond_types = multiple_bond_types
        self.num_edge_features = 3

        # Build interaction matrix
        # self._build_interaction_matrix(drug_invs, prot_invs)
        self._build_dataset_info()
        self.precompute = precompute
        if precompute:
            self.compute_graphs()
            self.load_embeddings()

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.all_drugs[idx]
        prot_id = self.all_uniprot_ids[idx]
        if self.precompute:
            nodes, edges = self.drug_graphs[smiles]
            embedding = self.prot_embeddings[prot_id]
        else:
            nodes, edges, __ = self._build_drug_graph(smiles)
            embedding = self.get_prot_embedding(prot_id)
        adj_mat = self._graph_to_adj_mat(edges)
        if not self.multiple_bond_types:
            adj_mat = torch.sum(adj_mat, dim=2)

        sample = {'node_features': nodes, 'adj_mat': adj_mat, 'prot_embedding': embedding}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _load_datafile(self, datafile):
        """
        Parameters
        ----------
        datafile : str
           File path.  Assumes that the file is only two columns
              1. Uniprot id
              2. Drug SMILES string
        """
        df = pd.read_table('drugbank_pairs.txt', header=None)
        protein_uniprots = np.array(df[0])
        drug_smiles = np.array(df[1])
        return drug_smiles, protein_uniprots

    def _build_interaction_matrix(self, drug_invs, prot_invs):
        M = np.max(prot_invs) + 1
        N = np.max(drug_invs) + 1
        self.interaction_matrix = sps.csr(np.ones(drug_invs.shape), (prot_invs, drug_invs), shape=(M, N))

    def compute_graphs(self):
        self.drug_graphs = {}
        for drug_smiles in self.unique_drugs:
            nodes, edges, __ = self._build_drug_graph(drug_smiles)
            self.drug_graphs[drug_smiles] = (nodes, edges)

    def load_embeddings(self):
        self.prot_embeddings = {}
        for prot_id in self.unique_uniprot_ids:
            self.prot_embeddings[prot_id] = self.get_prot_embedding(prot_id)

    def get_prot_embedding(self, prot_id):
        return np.loadtxt(self.protein_embedding_template % prot_id)

    def _build_drug_graph(self, smiles):
        """
        Builds a molecular graph form a smiles string.  Taken from [FIND SOURCE!]
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [], []
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
            edges.append((bond.GetBeginAtomIdx(), self.bond_dict[str(bond.GetBondType())], bond.GetEndAtomIdx()))
            assert self.bond_dict[str(bond.GetBondType())] != 3
        for atom in mol.GetAtoms():
            nodes.append(onehot(self.dataset_info['atom_types'].index(atom.GetSymbol()), len(self.dataset_info['atom_types'])))

        nodes = torch.tensor(nodes)
        edges = torch.tensor(edges)

        return nodes, edges, mol

    def _build_dataset_info(self):
        self.dataset_info = {'atom_types': ["H", "C", "N", "O", "F", "S"]
                             }

        self.bond_dict = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, "AROMATIC": 3}

    def _graph_to_adj_mat(self, edge_features):
        max_size = max(torch.max(edge_features[:, 0]), torch.max(edge_features[:, 2])) + 1
        print(max_size, 'maxsize')
        adj_mat = torch.zeros(max_size, max_size, self.num_edge_features)
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


def onehot(idx, len):
    idx = np.array(idx)  # make sure this is an array
    z = np.array([0 for _ in range(len)])
    z[idx] = 1
    return z


class MergeSnE1(object):
    def __init__(self):
        super(MergeSnE1, self).__init__()

    def __call__(self, sample):
        embedding = sample['prot_embedding']
        nodes = sample['node_features']
        N_resid = embedding.shape[0]
        N_nodes = nodes.shape[0]
        nodes_expanded = torch.stack([nodes] * N_resid, dim=1)
        embed_expanded = torch.stack([embedding] * N_nodes, dim=0)
        full_features = torch.cat((nodes_expanded, embed_expanded), dim=2)

        new_sample = {'features': full_features, 'adj_mat': sample['adj_mat']}
        return new_sample
