from argparse import Namespace
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
import numpy as np
from torch.nn import Linear

from torch_geometric.data import Data
import random
from torch.utils.data.dataset import Dataset
from collections import defaultdict
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch

allowable_features = {
    'possible_atomic_num_list': list(range(0, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

smile_changed = {}
def get_atom_poses(mol, conf):
    """tbd"""
    atom_poses = []
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 0:
            return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())
        pos = conf.GetAtomPosition(i)
        atom_poses.append([pos.x, pos.y, pos.z])
    return atom_poses

def get_MMFF_atom_poses(mol, numConfs=None, return_energy=False):
    """the atoms of mol will be changed in some cases."""
    try:
        new_mol = Chem.AddHs(mol)
        res = AllChem.EmbedMultipleConfs(new_mol, numConfs=numConfs)
        ### MMFF generates multiple conformations
        res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
        new_mol = Chem.RemoveHs(new_mol)
        index = np.argmin([x[1] for x in res])
        #energy = res[index][1]
        conf = new_mol.GetConformer(id=int(index))
    except:
        new_mol = mol
        AllChem.Compute2DCoords(new_mol)
        #energy = 0
        conf = new_mol.GetConformer()

    atom_poses = get_atom_poses(new_mol, conf)

    return atom_poses


def get_two_graph(mol,args):
    atom_feature_list = []
    atom_size = mol.GetNumAtoms()
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
                                         'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_feature_list.append(atom_feature)
    x = torch.tensor(np.array(atom_feature_list), dtype=torch.long)

    row, col = [], []
    for bond in mol.GetBonds():
        i,j = bond.GetBeginAtomIdx(),  bond.GetEndAtomIdx()
        row += [i, j]
        col += [j, i]
    # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
    edge_index = torch.tensor(([row, col]), dtype=torch.long)

    return x, atom_size, edge_index

def get_three_graph(mol,args):
    atom_feature_list = []
    edges_list = []
    atom_size = mol.GetNumAtoms()
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
                                         'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_feature_list.append(atom_feature)
    x = torch.tensor(np.array(atom_feature_list), dtype=torch.long)

    if len(mol.GetBonds()) > 0:
        for i in range(atom_size):
            for j in range(atom_size):
                if i != j:
                    edges_list.append((i, j))

                # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list), dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.zeros((0, 2), dtype=torch.long)
    atom_3dcoords = get_MMFF_atom_poses(mol, numConfs=None, return_energy=False)
    pos = torch.tensor(np.array(atom_3dcoords), dtype=torch.float)

    return x, atom_size, edge_index, pos
