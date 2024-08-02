import numpy as np
import matplotlib.pyplot as plt
import logging
import pickle as pkl
import seaborn
import os
import torch
from typing import List


def calculate_gln_accuracy(y_true, y_pred, threshold=0.8):
    'GLN_accuracy is defined as the percentage of samples with prediction residuals < 0.1'
    tp, fp, tn, fn = [0, 0, 0, 0]
    for y1, y2 in zip(y_true, y_pred):
        if abs(y1) >= threshold and abs(y2) >= threshold:
            tp += 1
        elif abs(y1) >= threshold and abs(y2) < threshold:
            fn += 1
        elif abs(y1) < threshold and abs(y2) >= threshold:
            fp += 1
        else:
            tn += 1
    recall = f"{(tp/(tp+fn)):.4f}" if tp+fn != 0 else 'NAN'
    precision = f"{(tp/(tp+fp)):.4f}" if tp+fp !=0 else 'NAN'
    return {'recall': recall, 'precision':precision}


def draw_single_heatmap(matrix, title: str, save_path: str):
    'Draw Heat Map for one matrix'
    matrix = np.squeeze(matrix)
    plt.figure(dpi=300)
    cmap = plt.get_cmap('seismic')
    res = seaborn.heatmap(data=matrix, cmap=cmap, xticklabels=20, yticklabels=20, center=0)
    plt.title(title, fontsize=14)
    # Drawing the frame
    res.axhline(y = 0, color='k',linewidth = 1.5)
    res.axhline(y = matrix.shape[0], color = 'k', linewidth = 1.5)
    res.axvline(x = 0, color = 'k', linewidth = 1.5)
    res.axvline(x = matrix.shape[1], color = 'k', linewidth = 1.5)
    plt.savefig(save_path)
    plt.close()


def scatter_mae_msa(mae, msa_depth):
    fig = plt.figure(dpi=300)
    plt.scatter(x=msa_depth, y=mae, s=8, alpha=0.4, c='b')
    plt.xlabel("Neff", {'size': 14})
    plt.ylabel("Abs(residuals)", {'size': 14})
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    
    return fig


def scatter_2mae(mae_us, mae_base):
    mae_us = np.where(mae_us>3, 3, mae_us)
    mae_base = np.where(mae_base>3, 3, mae_base)

    fig = plt.figure(dpi=300)
    plt.scatter(mae_us, mae_base, s=10, alpha=0.4, c='b')
    plt.xlabel("Abs(residuals) after fine-tuning", {'size': 14})
    plt.ylabel("Abs(residuals) before fine-tuning", {'size': 14})
    plt.plot([-4, 4], [-4, 4], ls="--", alpha=0.8, c="lightgrey")    
    plt.xlim((0, 3.1))
    plt.ylim((0, 3.1))
    plt.xticks(np.arange(0, 4, step=1))
    plt.yticks(np.arange(0, 4, step=1))
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)

    return fig


def backb_pos_to_gln_matrix(
        bb_pos: torch.Tensor, 
        asym_id: torch.Tensor
):
    '''
        Calculate GLN matrix in matrix form
        Args:
            backbone_pos: [*, N1+N2, 3, 3], [N, CA, C]
            asym_id: [*, N1+N2,]
        Return:
            GLN matrix: [*, N1-1, N2-1]
    '''

    # Only consider Ca
    # [*, N1+N2, 3]        
    ca_pos = bb_pos[..., 1, :]

    # Split two chains according to asym_id: [*, N1, 3], [*, N2, 3]
    n2 = torch.count_nonzero(asym_id - 1)
    n1 = asym_id.size(-1) - n2
    ca_pos_0, ca_pos_1 = torch.split(ca_pos, [n1, n2], dim=-2)

    # [*, N1-1, 3], [*, N2-1, 3]
    r_0 = (ca_pos_0[..., 1:, :] + ca_pos_0[..., :-1, :]) / 2
    r_1 = (ca_pos_1[..., 1:, :] + ca_pos_1[..., :-1, :]) / 2
    delta_r_0 = ca_pos_0[..., 1:, :] - ca_pos_0[..., :-1, :]
    delta_r_1 = ca_pos_1[..., 1:, :] - ca_pos_1[..., :-1, :]

    # [*, N1-1, N2-1, 3]
    r_0 = r_0.unsqueeze(-2).expand(*r_0.shape[:-2], n1-1, n2-1, 3)
    r_1 = r_1.unsqueeze(-3).expand(*r_1.shape[:-2], n1-1, n2-1, 3)
    delta_r_0 = delta_r_0.unsqueeze(-2).expand(*delta_r_0.shape[:-2], n1-1, n2-1, 3)
    delta_r_1 = delta_r_1.unsqueeze(-3).expand(*delta_r_1.shape[:-2], n1-1, n2-1, 3)

    # [*, N1-1, N2-1]
    numerator = torch.matmul(
        (r_0 - r_1).unsqueeze(-2), 
        torch.cross(delta_r_0, delta_r_1, dim=-1).unsqueeze(-1)
    ).squeeze()
    denomnator = torch.pow(torch.linalg.norm(r_0 - r_1, dim=-1), 3) * 4 * np.pi

    return numerator / denomnator