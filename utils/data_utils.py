from utils.rigid_utils import Rotation, Rigid
from utils import protein

import constants.residue_constants as rc
from typing import Mapping, Optional, Sequence, Any, Dict
import numpy as np
import torch


FeatureDict = Mapping[str, np.ndarray]
def make_sequence_features(
    sequence: str, description: str, num_res: int
) -> FeatureDict:
    """Construct a feature dict of sequence features."""
    features = {}
    features["aatype"] = rc.sequence_to_onehot(
        sequence=sequence,
        mapping=rc.restype_order_with_x,
        map_unknown_to_x=True,
    )
    features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)
    features["domain_name"] = np.array(
        [description.encode("utf-8")], dtype=np.object_
    )
    features["residue_index"] = np.array(range(num_res), dtype=np.int32)
    features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
    features["sequence"] = np.array(
        [sequence.encode("utf-8")], dtype=np.object_
    )
    return features
def _aatype_to_str_sequence(aatype):
    return ''.join([
        rc.restypes_with_x[aatype[i]] 
        for i in range(len(aatype))
    ])


def make_protein_features(
    protein_object: protein.Protein, 
    description: str,
    _is_distillation: bool = False,
) -> FeatureDict:
    pdb_feats = {}
    aatype = protein_object.aatype
    sequence = _aatype_to_str_sequence(aatype)
    pdb_feats.update(
        make_sequence_features(
            sequence=sequence,
            description=description,
            num_res=len(protein_object.aatype),
        )
    )

    all_atom_positions = protein_object.atom_positions
    all_atom_mask = protein_object.atom_mask

    pdb_feats["all_atom_positions"] = all_atom_positions.astype(np.float32)
    pdb_feats["all_atom_mask"] = all_atom_mask.astype(np.float32)

    pdb_feats["resolution"] = np.array([0.]).astype(np.float32)
    pdb_feats["is_distillation"] = np.array(
        1. if _is_distillation else 0.
    ).astype(np.float32)
    pdb_feats["aatype"] = np.array(aatype, dtype=np.int32)

    return pdb_feats


def make_pdb_features(
    protein_object: protein.Protein,
    description: str,
    is_distillation: bool = True,
    confidence_threshold: float = 50.,
) -> FeatureDict:
    pdb_feats = make_protein_features(
        protein_object, description, _is_distillation=True
    )

    if(is_distillation):
        high_confidence = protein_object.b_factors > confidence_threshold
        high_confidence = np.any(high_confidence, axis=-1)
        pdb_feats["all_atom_mask"] *= high_confidence[..., None]

    return pdb_feats


def get_backbone_frames(protein, eps=1e-8):
    backbone_atom_pos = protein["backbone_atom_pos"]
    rigids = Rigid.from_3_points(
        p_neg_x_axis=backbone_atom_pos[..., 0, :],
        origin=backbone_atom_pos[..., 1, :],
        p_xy_plane=backbone_atom_pos[..., 2, :],
        eps=eps,
    )
    rigids_tensor = rigids.to_tensor_4x4()
    protein["backbone_rigid_tensor"] = rigids_tensor

    return protein


def process_pdb(pdb_path) -> Mapping[str, torch.Tensor]:
    '''
    Args:
        pdb_path: pdb file path
    Return:
        feature dict
    '''
    with open(pdb_path, 'r') as f:
        pdb_str = f.read()
    protein_object = protein.from_pdb_string(pdb_str, None)
    feat_dict = make_pdb_features(
        protein_object, 
        description=pdb_path, 
        is_distillation=False
    )
    processed_feat_dict = {
        # [2N, ]
        'aatype': feat_dict['aatype'],
        # [2N, 3, 3]
        'backbone_atom_pos': feat_dict['all_atom_positions'][..., :3, :],
        # [2N, 3]
        'backbone_atom_mask': feat_dict['all_atom_mask'][..., :3],
        # [2N, ]
        'backbone_rigid_mask': feat_dict['all_atom_mask'][..., 0],
    }

    # for multimer only:
    length = int(processed_feat_dict["aatype"].shape[0]/2)
    left = np.concatenate(
        [np.full((length, length), 1, dtype=np.float32), np.full((length, length), 0, dtype=np.float32)],
        axis=0
    )
    right = np.concatenate(
        [np.full((length, length), 0, dtype=np.float32), np.full((length, length), 1, dtype=np.float32)],
        axis=0
    )
    processed_feat_dict['intra_chain_mask'] = np.concatenate([left, right], axis=1) 

    processed_feat_dict = {
        k: torch.tensor(v) for k, v in processed_feat_dict.items() 
    }        
    processed_feat_dict = get_backbone_frames(processed_feat_dict)
    
    return processed_feat_dict



def matrix_from_bb_pos(bb_pos: torch.Tensor) -> torch.Tensor:
    '''
        Calculate GLN matrix in matrix form
        Args:
            bb_pos: [*, 2N, 3, 3], [N, CA, C]
        Return:
            GLN matrix: [*, N-1, N-1]
    '''
    # Only consider Ca
    # [*, 2N, 3]
    ca_pos = bb_pos[..., 1, :]

    # Split two chains, each [*, N, 3]
    length = ca_pos.size(-2) // 2
    ca_pos_0, ca_pos_1 = torch.split(ca_pos, length, dim=-2)

    # [*, N-1, 3]
    r_0 = (ca_pos_0[..., 1:, :] + ca_pos_0[..., :-1, :]) / 2
    r_1 = (ca_pos_1[..., 1:, :] + ca_pos_1[..., :-1, :]) / 2
    delta_r_0 = ca_pos_0[..., 1:, :] - ca_pos_0[..., :-1, :]
    delta_r_1 = ca_pos_1[..., 1:, :] - ca_pos_1[..., :-1, :]

    # [*, N-1, N-1, 3]
    r_0 = r_0.unsqueeze(-2).expand(*r_0.shape[:-2], length-1, length-1, 3)
    r_1 = r_1.unsqueeze(-3).expand(*r_1.shape[:-2], length-1, length-1, 3)
    delta_r_0 = delta_r_0.unsqueeze(-2).expand(*delta_r_0.shape[:-2], length-1, length-1, 3)
    delta_r_1 = delta_r_1.unsqueeze(-3).expand(*delta_r_1.shape[:-2], length-1, length-1, 3)

    # [*, N-1, N-1]
    numerator = torch.matmul(
        (r_0 - r_1).unsqueeze(-2), 
        torch.cross(delta_r_0, delta_r_1, dim=-1).unsqueeze(-1)
    ).squeeze()
    denomnator = torch.pow(torch.linalg.norm(r_0 - r_1, dim=-1), 3) * 4 * np.pi

    return numerator / denomnator


