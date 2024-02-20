import pickle as pkl
import numpy as np
import torch
import os

ori_dir = '/media/puqing/9CDC3B71DC3B44B4/puqing/alphafold_representations'
dest_dir = 'data/representation'

# Load the data
for i, pid in enumerate(os.listdir(ori_dir)):
    print(i, end='\r')
    if os.path.exists(f'{dest_dir}/single/{pid}.npy') and os.path.exists(f'{dest_dir}/pair/{pid}.npy'):
        continue
    dict_ = pkl.load(open(f'{ori_dir}/{pid}/result_model_1_multimer_v2_pred_0.pkl',"rb"), encoding='iso-8859-1')
    single = dict_['single_reprersentations']
    pair = dict_['pair_representations']
    print(single.shape, pair.shape)
    np.save(f'{dest_dir}/single/{pid}.npy', single)
    np.save(f'{dest_dir}/pair/{pid}.npy', pair)
