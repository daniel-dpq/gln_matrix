from model.model import Model
from utils.gln_utils import compare_number, compare_loss_residual
from utils.data_utils import process_pdb, matrix_from_bb_pos
from training.train_utils import load_alphafold2_params
import training.loss as L

from torch.utils.tensorboard import SummaryWriter
import torch.utils.data
import torch.nn as nn
import torch

from typing import List, Tuple, Sequence

import numpy as np
import pickle as pkl
import pandas as pd

import time
import argparse
import os

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
dtype = torch.float32


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}', end='\t')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class StructureModuleDataset(torch.utils.data.Dataset):
    'Generate dataset used for refining backbone'
    
    def __init__(self, dataset_file: str, data_dir: str):

        self.pids = list(pd.read_csv(os.path.join(data_dir, dataset_file), header=None, sep='\\s+')[1])[:10]
        pdb_dir = os.path.join(data_dir, 'pdb')

        # Get feature dict from pdb files 
        # 'aatype': [*, 2N]
        # 'backbone_atom_pos': [*, 2N, 3, 3]
        # 'backbone_atom_mask': [*, 2N, 3]
        # 'backbone_rigid_tensor': [*, 2N, 4, 4]
        # 'backbone_rigid_mask': [*, 2N]
        
        pdb_paths = [
            os.path.join(pdb_dir, f'{pid}_loopGS.pdb')
            for pid in self.pids
        ]
        self.pdb_feats = [
            process_pdb(pdb)
            for pdb in pdb_paths
        ]

        # Get GLN matrix
        self.gln_matrix = [
            matrix_from_bb_pos(feat['backbone_atom_pos']).numpy()
            for feat in self.pdb_feats
        ]
        
    def __getitem__(self, idx):
        pdb_feat = self.pdb_feats[idx]
        pid = self.pids[idx]
        matrix = self.gln_matrix[idx]

        return pid, pdb_feat, matrix
    
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

    def _dict_multistack(self, dicts, max_len):
        first = dicts[0]
        new_dict = {}
        for k, v in first.items():
            all_v = []
            for d in dicts:
                v = d[k]
                if k == 'pair':
                    pad_v = nn.functional.pad(v, (0, 0, 0, max_len-len(v), 0, max_len-len(v)), 'constant', 0)
                else:
                    pad_v = torch.concat(
                        [v, torch.zeros((max_len-len(v), *(v.size()[1:])), device=self.device)])
                all_v.append(pad_v)
            new_dict[k] = torch.stack(all_v)

        return new_dict

    def __call__(self, raw_batch: Sequence[Tuple[str, np.ndarray, np.ndarray]]):
        '''
        Postprocess a batch generated from DataLoader
        raw_batch is a list of tuples which are outputs of Dataset.__getitem__
        batch_size = 1
        '''
        batch_pid, batch_feats, batch_matrix = zip(*raw_batch)

        max_len = 0
        for feats in batch_feats:
            if max_len < len(feats['backbone_rigid_mask']):
                max_len = len(feats['backbone_rigid_mask'])

        evoformer_output_dict = []
        for pid in batch_pid:

            dict_ = pkl.load(open(f'/media/puqing/9CDC3B71DC3B44B4/puqing/alphafold_representations/{pid}/result_model_1_multimer_v2_pred_0.pkl',"rb"), encoding='iso-8859-1')
            dict_['single'] = torch.from_numpy(dict_.pop('single_reprersentations'))
            dict_['pair'] = torch.from_numpy(dict_.pop('pair_representations'))
            # dict_ = pkl.load(open(f'/scratch/PI/hanyugao/puqing/alphafold_outputs3/{pid}/result_model_1_multimer_v2_pred_0.pkl',"rb"), encoding='iso-8859-1')
            # dict_['single'] = torch.from_numpy(dict_.pop('single'))
            # dict_['pair'] = torch.from_numpy(dict_.pop('pair'))
            evoformer_output_dict.append(dict_)

        evoformer_output_dict = self._dict_multistack(
            self._dict_list_to_device(evoformer_output_dict), max_len)

        batch_feats = self._dict_multistack(
            self._dict_list_to_device(batch_feats), max_len)
        
        batch_matrix = torch.tensor(np.array(batch_matrix)).to(self.device)

        batch = {
            'evoformer_output_dict': evoformer_output_dict,
            'gt_matrix': batch_matrix,
            **batch_feats
        }

        return batch


def main(args):

    # output directory name
    output_dir = args.out_dir
    print(f"Args: {args}")
    print(f'Ouput directory: {output_dir}')

    # Load data
    print('=====> Preparing data...')
    train_data = StructureModuleDataset('PDB_C2_train.txt', args.data_dir)
    valid_data = StructureModuleDataset('PDB_C2_valid.txt', args.data_dir)
    test_data = StructureModuleDataset('PDB_C2_test.txt', args.data_dir)
    collate_fn = BatchConverter(device, dtype)
    train_loader, valid_loader, test_loader = [
        torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=(dataset!=test_data), collate_fn=collate_fn)
        for dataset in [train_data, valid_data, test_data]
    ]
    train_size, valid_size, test_size = [len(x) for x in (train_data, valid_data, test_data)]

    # Get model structure
    model = Model().to(device)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))

    # Evaluate only
    if args.evaluate:
            print('=====> Evaluateing only...')
            if args.load_model:
                print(f'Loading model from {args.load_model}...')
                model.load_state_dict(torch.load(args.load_model), strict=True)
            else:
                params = os.path.join(output_dir, 'checkpoint.pt')
                print(f'Loading model from {params}...')
                model.load_state_dict(torch.load(params), strict=True)
            model.eval()
            pred_mat = []
            true_mat = []
            loss = []
            loss_fn = L.BackboneLoss().to(device)

            with torch.no_grad():
                for batch in test_loader:
                    out = model(batch['evoformer_output_dict'], batch['aatype'])
                    pred_mat.append(*out['gln_matrix'].cpu())
                    true_mat.append(*batch['gt_matrix'].cpu())
                    loss.append(loss_fn(out, batch).cpu().item())

                    del out
            fig1, fig6 = compare_number(true_mat, pred_mat)
            fig4, fig5 = compare_loss_residual(loss, true_mat, pred_mat)
            # draw_heatmap(true_mat, pred_mat, test_data.pids, os.path.join(output_dir, 'heatmap'))
            fig1.savefig(os.path.join(output_dir, 'GLN.png'))
            writer.add_figure('gln true vs pred testing', figure=fig1, close=False)
            writer.add_figure('loss_mae', figure=fig4, close=False)
            writer.add_figure('loss_mae2', figure=fig5, close=False)
            writer.add_figure('metrics vs. different thresholds', figure=fig6, close=False)

            return

    # Load parameters
    load_alphafold2_params(args.alphafold_path, model, device, dtype)

    # Optmizer, Scheduler and Loss function
    if args.loss == 'mseloss':
        loss_fn = L.MSEBackboneLoss().to(device)
    elif args.loss == 'unitmseloss':
        loss_fn = L.UnitMSEBackboneLoss(args.weight).to(device)
    elif args.loss == 'unitmaeloss':
        loss_fn = L.UnitMAEBackboneLoss(args.weight).to(device)
    elif args.loss is None:
        loss_fn = L.BackboneLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=args.factor,
        patience=args.patience[0],
        min_lr=1e-9
    )
    early_stopping = EarlyStopping(
        patience=args.patience[1], 
        path=os.path.join(output_dir, 'checkpoint.pt')
    )

    # Training
    print('=====> Training...')
    print(f'{train_size} training data in total!')
    print(f'{valid_size} validation data in total!')
    for epoch in range(args.epoch):
        print(f'Epoch {epoch+1}/{args.epoch}')
        start_time = time.time()

        # Training set
        model.train()
        for batch in train_loader:
            out = model(batch['evoformer_output_dict'], batch['aatype'])
            loss = loss_fn(out, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del out, batch
        loss = loss.cpu().item()

        # Validation set
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in valid_loader:
                out = model(batch['evoformer_output_dict'], batch['aatype'])
                loss = loss_fn(out, batch).cpu().item()
                val_loss += loss*args.batch_size

                del out

        val_loss /= valid_size
        print(f'loss: {loss:.10f}\t', f'val_loss:{val_loss:.10f}', end='\t')

        # adapt scheduler and early_stopper
        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print()
            break

        # Record time
        end_time =time.time()
        print(f'({end_time-start_time:.2f}s)')

        writer.add_scalar('loss', loss, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
    
    torch.save(model.state_dict(), os.path.join(output_dir, 'finalpoint.pt'))


    # Testing
    print('=====> Testing...')
    print(f'{test_size} test data in total!')
    model.eval()
    pred_mat = []
    true_mat = []
    loss = []
    loss_fn = L.BackboneLoss().to(device)

    with torch.no_grad():
        for batch in test_loader:
            out = model(batch['evoformer_output_dict'], batch['aatype'])
            pred_mat.append(*out['gln_matrix'].cpu())
            true_mat.append(*batch['gt_matrix'].cpu())
            loss.append(loss_fn(out, batch).cpu().item())

            del out
            
    fig1, fig6 = compare_number(true_mat, pred_mat)
    fig4, fig5 = compare_loss_residual(loss, true_mat, pred_mat)
    # draw_heatmap(true_mat, pred_mat, test_data.pids, os.path.join(output_dir, 'heatmap'))
    writer.add_figure('gln true vs pred testing', figure=fig1, close=False)
    writer.add_figure('loss_mae', figure=fig4, close=False)
    writer.add_figure('loss_mae2', figure=fig5, close=False)
    writer.add_figure('metrics vs. different thresholds', figure=fig6, close=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default='outputs', help='output directory')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory')
    parser.add_argument('--rep_dir', type=str, default='', help='representation directory')
    parser.add_argument('--alphafold_path', type=str, default='/media/puqing/9CDC3B71DC3B44B4/puqing/alphafold_data/params/params_model_1_multimer_v2.npz', help='path of alphafold2 param')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train')    
    parser.add_argument('--lr', type=float, default=1e-7, help='initial learning rate')
    parser.add_argument('--factor', type=float, default=0.2, help='learning rate reduce factor')
    parser.add_argument('--patience', type=int, default=[10, 8], nargs='+', help='lr schedule (when to drop lr by factor)')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--loss', type=str, default=None, help='which loss to use', choices=['mseloss', 'unitmseloss', 'unitmaeloss'])
    parser.add_argument('--weight', type=float, default=None, help='weight of unit mse/mae when using unitmseloss')

    args=parser.parse_args()
    main(args)
