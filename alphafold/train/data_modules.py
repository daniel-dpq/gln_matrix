import pandas as pd
import pickle as pkl
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Mapping, List, Tuple, Sequence


class AlphafoldDataset(Dataset):
    'Generate dataset used for refining backbone'
    
    def __init__(
        self, 
        dataset_path: str,
        feature_dir: str,
        pdb_dir: str='/home/pdengad/protein/data/pdb/pdb_full',
        gln_matrix_dir: str='/home/pdengad/protein/data/pdb/gln_matrix',
    ):
        self.pids = list(pd.read_csv(dataset_path, header=None, sep='\\s+')[1])

        # Features from sequence/MSA, last dimension is the iteration number
        # aatype, target_feat, residue_index, msa_feat, seq_mask, msa_mask...
        self.seq_msa_feat = [
            pkl.load(os.path.join(feature_dir, pid, 'features.pkl'))
            for pid in self.pids
        ]

        # Get ground truth labels from pdb files 
        # 'gt_backb_positions': [*, 2N, 3, 3]
        # 'gt_affine_tensor': [*, 2N, 4, 4]
        # self.pdb_feats = [
        #     process_pdb(os.path.join(pdb_dir, f'{pid}_loopGS.pdb'))
        #     for pid in self.pids
        # ]

        # Load GLN matrix
        matrix_paths = [
            os.path.join(gln_matrix_dir, f'{pid}_discrete_matrix.txt')
            for pid in self.pids
        ]
        self.gln_matrix = [
            np.array(pd.read_csv(path, sep=' ', header=None))
            for path in matrix_paths
        ]
        
    def __getitem__(self, idx):
        #pdb_feat = self.pdb_feats[idx]
        pid = self.pids[idx]
        seq_msa_feat = self.seq_msa_feat[idx]
        matrix = self.gln_matrix[idx]

        return pid, seq_msa_feat, matrix
    
    def __len__(self):
        return len(self.pids)
    

class BatchConverter(object):
    '''
    Define a callable class that converts a batch of data but can also 
    accept other parameters besides raw_batch
    '''
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype
    
    def _dict_list_to_device(self, dict_list):
        new_dict_list = []
        for dict_ in dict_list:
            new_dict_list.append({
                k: v.to(self.device).type(self.dtype)
                for k, v in dict_.items() 
            })
        return new_dict_list

    def _dict_multistack(self, dicts):
        first = dicts[0]
        new_dict = {}
        for k, v in first.items():
            all_v = []
            for d in dicts:
                v = d[k]
                all_v.append(v)
            new_dict[k] = torch.stack(all_v)

        return new_dict

    def __call__(
        self, 
        raw_batch: Sequence[Tuple[str, Mapping, np.ndarray]],
    ):
        '''
        Postprocess a batch generated from DataLoader
        raw_batch is a list of tuples which are outputs of Dataset.__getitem__
        batch_size = 1
        '''
        batch_pid, batch_feats, batch_matrix = zip(*raw_batch)

        batch_feats = self._dict_multistack(
            self._dict_list_to_device(batch_feats))
        
        batch_matrix = torch.tensor(np.array(batch_matrix)).to(self.device)

        batch = {
            'pids': batch_pid,
            'gt_matrix': batch_matrix,
            **batch_feats
        }

        return batch