from collections import defaultdict
from torchdrug import datasets, data, utils
import os
import glob
import random
from rdkit import Chem
from tqdm import tqdm
import pickle


class EnzymeCommissionDataset(datasets.EnzymeCommission):

    url = "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/data/EnzymeCommission.tar.gz"
    md5 = "728e0625d1eb513fa9b7626e4d3bcf4d"
    processed_file = "enzyme_commission_toy.pkl.gz"

    # test_cutoffs = [0.3, 0.4, 0.5, 0.7, 0.95]

    def __init__(self, local_rank: int, **kwargs):
        super(EnzymeCommissionDataset, self).__init__(**kwargs)
        self.local_rank = local_rank

    def get_item(self, index):
        item = super(EnzymeCommissionDataset, self).get_item(index)
        item["sequence"] = self.sequences[index]
        return item


def convert_unnatural_aa(text):
    natural_aa = ["G", "A", "S", "P", "V", "T", "C", "I", "L", "N", "D", "Q", "K", "E", "M", "H", "F", "R", "Y", "W", "X"]
    res = list(text)
    for i, t in enumerate(res):
        if t not in natural_aa:
            res[i] = 'X'
    return ''.join(res)


class PDBPeptide(data.ProteinDataset):
    """
    A set of peptides with their 3D structures in PDB database

    Statistics:
        - num: 10,718

    Parameters:
        path (str): the path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """
    processed_file = 'pdb_clean.plk.gz'

    def __init__(self, local_rank, path, split_indices=None, verbose=1, **kwargs):
        # super(PDBPeptide, self).__init__(**kwargs)
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            print(path)
            raise ValueError("Wrong path")
        self.path = path
        self.local_rank = local_rank
        self.targets = defaultdict(list)
        self.sequences = []
        self.pdb_files = []
        self.data = []
        pkl_file = os.path.join(path, self.processed_file)
        pdb_files = glob.glob(os.path.join(path, "*.pdb"))

        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, verbose=verbose)
        else:
            self.load_pdbs(pdb_files, verbose=verbose)
            self.save_pickle(pkl_file, verbose=verbose)

        if split_indices is not None:
            self.data = [self.data[i] for i in split_indices]
            self.sequences = [self.sequences[i] for i in split_indices]
            self.pdb_files = [self.pdb_files[i] for i in split_indices]
            self.filter_data()
            self.sort_by_length()

        self.filter_data()
        print('pep pdb dataset length: ', len(self.data))

    def filter_data(self):

        for i in range(len(self.data) - 1, -1, -1):
            seq = self.sequences[i]
            if len(str(seq)) <= 2:
                self.data.pop(i)
                self.sequences.pop(i)
                self.pdb_files.pop(i)

    def sort_by_length(self):
        sequences_len = [len(x) for x in self.sequences]
        self.data_all = [x for _, x in sorted(zip(sequences_len, self.data, self.sequences, self.pdb_files), key=lambda a:a[0])]
        self.data = [x[0] for x in self.data_all]
        self.sequences = [x[1] for x in self.data_all]
        self.pdb_files = [x[2] for x in self.data_all]
        # self.sequences = [x for _, x in sorted(zip(sequences_len, self.sequences), key=lambda a:a[0])]
        # self.pdb_files = [x for _, x in sorted(zip(sequences_len, self.pdb_files), key=lambda a:a[0])]

    def load_pickle(self, pkl_file, verbose=1):
        """
        Load the dataset from a pickle file.

        Parameters:
            pkl_file (str): file name
        """
        with utils.smart_open(pkl_file, "rb") as fin:
            num_sample = pickle.load(fin)
            indexes = range(num_sample)
            if verbose:
                indexes = tqdm(indexes, "Loading %s" % pkl_file)
            for i in indexes:
                pdb_file, sequence, protein = pickle.load(fin)
                self.sequences.append(sequence)
                self.pdb_files.append(pdb_file)
                self.data.append(protein)

    def load_pdbs(self, pdb_files, lazy=False, verbose=0, **kwargs):
        """
        Load the dataset from pdb files.

        Parameters:
            pdb_files (list of str): pdb file names
            transform (Callable, optional): protein sequence transformation function
            lazy (bool, optional): if lazy mode is used, the proteins are processed in the dataloader.
                This may slow down the data loading process, but save a lot of CPU memory and dataset loading time.
            verbose (int, optional): output verbose level
            **kwargs
        """
        self.num_sample = len(pdb_files)
        self.lazy = lazy
        self.kwargs = kwargs
        #

        if verbose:
            pdb_files = tqdm(pdb_files, "Constructing proteins from pdbs")
        for i, pdb_file in enumerate(pdb_files):
            seq = None
            protein = None
            if not lazy or i == 0:
                mol = Chem.MolFromPDBFile(pdb_file)
                if mol:
                    seq = Chem.MolToFASTA(mol)[2:].strip()
                    seq = convert_unnatural_aa(seq)
                try:
                    protein = data.Protein.from_molecule(mol, **kwargs)
                except:
                    pass

            if hasattr(protein, "residue_feature"):
                with protein.residue():
                    protein.residue_feature = protein.residue_feature.to_sparse()

            if protein and len(seq) > 0:
                self.data.append(protein)
                self.pdb_files.append(pdb_file)
                self.sequences.append(seq)


    def get_item(self, index):

        if getattr(self, "lazy", False):
            protein = data.Protein.from_pdb(self.pdb_files[index], self.kwargs)
        else:
            protein = self.data[index].clone()

        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()

        item = {"graph": protein}
        # if self.transform:
        #     item = self.transform(item)

        item["sequence"] = self.sequences[index]
        item["pdbfile"] = self.pdb_files[index]

        return item

class AlphaFoldDB(data.ProteinDataset):
    """
    3D protein structures predicted by AlphaFold.
    See https://alphafold.ebi.ac.uk/download
    """

    def __init__(self, local_rank, path, pkl_path, verbose=1, mini_dataset_len=0, **kwargs):
        super().__init__()
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            print(path)
            raise ValueError("Wrong path")

        self.path = path
        self.pkl_path = pkl_path
        self.local_rank = local_rank
        
        if self.path.endswith('plk.gz'):
            self.load_pickle(self.path, verbose=verbose)
        
        else:
            self.processed_file = 'AlphaFold_sampled_{}.plk.gz'.format(mini_dataset_len)
            pkl_file = os.path.join(pkl_path, self.processed_file)

            if os.path.exists(pkl_file):
                self.load_pickle(pkl_file, verbose=verbose)
            else:
                print('start load pdb files')
                pdb_files = glob.glob(os.path.join(self.path, "*.pdb"))
                if mini_dataset_len > 0:
                    random.seed(1234)
                    pdb_files = random.sample(pdb_files, mini_dataset_len)
                    print(pdb_files)

                self.load_pdbs(pdb_files, verbose=verbose)
                self.save_pickle(pkl_file, verbose=verbose)
        # self.filter_data()

    def load_pickle(self, pkl_file, verbose=1):
        """
        Load the dataset from a pickle file.

        Parameters:
            pkl_file (str): file name
        """
        with utils.smart_open(pkl_file, "rb") as fin:
            num_sample = pickle.load(fin)

            self.sequences = []
            self.pdb_files = []
            self.data = []
            indexes = range(num_sample)
            if verbose:
                indexes = tqdm(indexes, "Loading %s" % pkl_file)
            for i in indexes:
                pdb_file, sequence, protein = pickle.load(fin)
                self.sequences.append(sequence)
                self.pdb_files.append(pdb_file)
                self.data.append(protein)

    def filter_data(self):
        for i in range(len(self.data)-1, -1, -1):
            protein = self.data[i]
            if protein.num_residue > 50:
                self.data.pop(i)

    def __getitem__(self, index):

        protein = self.data[index].clone()
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        item = {"graph": protein}
        # if self.transform:
        #     item = self.transform[0](item)

        item["sequence"] = self.sequences[index]
        item["pdbfile"] = self.pdb_files[index]
        return item

class CustomSubset(data.ProteinDataset):
    '''A custom subset class'''

    def __init__(self, dataset, indices, transform, local_rank):
        self.transform = transform,
        self.atom_feature = None
        self.bond_feature = None
        self.local_rank = local_rank
        self.sequences = [dataset.sequences[i] for i in indices]
        self.data = [dataset.data[i] for i in indices]
        self.pdb_files = [dataset.pdb_files[i] for i in indices]
        self.sort_by_length()

    def sort_by_length(self):
        # sequences_len = [len(x) for x in self.sequences]
        # self.data = [x for _, x in sorted(zip(sequences_len, self.data), key=lambda a: a[0])]
        # self.sequences = [x for _, x in sorted(zip(sequences_len, self.sequences), key=lambda a: a[0])]
        # self.pdb_files = [x for _, x in sorted(zip(sequences_len, self.pdb_files), key=lambda a: a[0])]
        sequences_len = [len(x) for x in self.sequences]
        self.data_all = [(x, y, z) for _, x, y, z in sorted(zip(sequences_len, self.data, self.sequences, self.pdb_files), key=lambda a:a[0])]
        self.data = [x[0] for x in self.data_all]
        self.sequences = [x[1] for x in self.data_all]
        self.pdb_files = [x[2] for x in self.data_all]

    def __getitem__(self, idx):
        protein = self.data[idx].clone()
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()

        item = {"graph": protein}
        if self.transform:
            item = self.transform[0](item)
        item["sequence"] = self.sequences[idx]
        item["pdbfile"] = self.pdb_files[idx]
        return item
    def __len__(self):
        return len(self.sequences)
