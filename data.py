from argparse import Namespace
from rdkit import Chem
import torch
from torch.utils.data.dataset import Dataset
from collections import defaultdict
import random
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedKFold

class MoleData:
    def __init__(self,line,args):
        self.args = args
        self.smile = line[0]
        self.mol = Chem.MolFromSmiles(self.smile)
        self.label = [float(x) if x != '' else None for x in line[1:]]
        
    def task_num(self):

        return len(self.label)
    
    def change_label(self,label):
        self.label = label


class MoleDataSet(Dataset):
    def __init__(self,data):
        self.data = data
        if len(self.data) > 0:
            self.args = self.data[0].args
        else:
            self.args = None
        self.scaler = None
    
    def smile(self):
        smile_list = []
        for one in self.data:
            smile_list.append(one.smile)
        return smile_list
    
    def mol(self):
        mol_list = []
        for one in self.data:
            mol_list.append(one.mol)
        return mol_list
    
    def label(self):
        label_list = []
        for one in self.data:
            label_list.append(one.label)
        return label_list
    
    def task_num(self):
        if len(self.data) > 0:
            return self.data[0].task_num()
        else:
            return None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,key):
        return self.data[key]
    
    def random_data(self,seed):
        random.seed(seed)
        random.shuffle(self.data)
    
    def change_label(self,label):
        assert len(self.data) == len(label)
        for i in range(len(label)):
            self.data[i].change_label(label[i])


def generate_scaffold(mol, include_chirality=False):
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    return scaffold


def scaffold_to_smiles(mol, use_indices=False):
    scaffolds = defaultdict(set)
    for i, one in enumerate(mol):
        scaffold = generate_scaffold(one)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(one)

    return scaffolds


def scaffold_split(data, size, seed, log):
    assert sum(size) == 1

    # Split
    train_size, val_size, test_size = size[0] * len(data), size[1] * len(data), size[2] * len(data)
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the data
    scaffold_to_indices = scaffold_to_smiles(data.mol(), use_indices=True)

    index_sets = list(scaffold_to_indices.values())
    big_index_sets = []
    small_index_sets = []
    for index_set in index_sets:
        if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)
    random.seed(seed)
    random.shuffle(big_index_sets)
    random.shuffle(small_index_sets)
    index_sets = big_index_sets + small_index_sets

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1

    log.debug(f'Total scaffolds = {len(scaffold_to_indices):,} | '
              f'train scaffolds = {train_scaffold_count:,} | '
              f'val scaffolds = {val_scaffold_count:,} | '
              f'test scaffolds = {test_scaffold_count:,}')

    # Map from indices to data
    train = [data[i] for i in train]
    val = [data[i] for i in val]
    test = [data[i] for i in test]

    return MoleDataSet(train), MoleDataSet(val), MoleDataSet(test)

def scaffold_split_balanced(data,size,seed,log):
    """
    Split a dataset by scaffold so that no molecules sharing a scaffold are in the same split.

    :param data: A MoleculeDataset.
    :param sizes: A length-3 tuple with the proportions of data in the
    train, validation, and test sets.
    :param balanced: Try to balance sizes of scaffolds in each set, rather than just putting smallest in test set.
    :param seed: Seed for shuffling when doing balanced splitting.
    :param logger: A logger.
    :return: A tuple containing the train, validation, and test splits of the data.
    """

    assert sum(size) == 1

    # Split
    train_size, val_size, test_size = size[0] * len(data), size[1] * len(data), size[2] * len(data)
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the smiles
    scaffold_to_indices = scaffold_to_smiles(data.mol(), use_indices=True)

    #if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
    index_sets = list(scaffold_to_indices.values())
    big_index_sets = []
    small_index_sets = []
    for index_set in index_sets:
        if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)
    random.seed(seed)
    random.shuffle(big_index_sets)
    random.shuffle(small_index_sets)
    index_sets = big_index_sets + small_index_sets
    #else:  # Sort from largest to smallest scaffold sets
        #index_sets = sorted(list(scaffold_to_indices.values()), key=lambda index_set: len(index_set), reverse=True)

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1
    log.debug(f'Total scaffolds = {len(scaffold_to_indices):,} | '
              f'train scaffolds = {train_scaffold_count:,} | '
              f'val scaffolds = {val_scaffold_count:,} | '
              f'test scaffolds = {test_scaffold_count:,}')

    # Map from indices to data
    train = [data[i] for i in train]
    val = [data[i] for i in val]
    test = [data[i] for i in test]

    return MoleDataSet(train), MoleDataSet(val), MoleDataSet(test)

