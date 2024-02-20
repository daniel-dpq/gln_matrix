import torch.nn as nn
import numpy as np
import torch

param_map_dict = {
    'alphafold/alphafold_iteration/structure_module/single_layer_norm//scale': 'structure_module.layer_norm_s.weight',
    'alphafold/alphafold_iteration/structure_module/single_layer_norm//offset': 'structure_module.layer_norm_s.bias',
    'alphafold/alphafold_iteration/structure_module/pair_layer_norm//scale': 'structure_module.layer_norm_z.weight',
    'alphafold/alphafold_iteration/structure_module/pair_layer_norm//offset': 'structure_module.layer_norm_z.bias',
    'alphafold/alphafold_iteration/structure_module/initial_projection//weights': 'structure_module.linear_in.weight',
    'alphafold/alphafold_iteration/structure_module/initial_projection//bias': 'structure_module.linear_in.bias',
    'alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention//trainable_point_weights': 'structure_module.ipa.head_weights',
    'alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/attention_2d//weights': 'structure_module.ipa.linear_b.weight',
    'alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/attention_2d//bias': 'structure_module.ipa.linear_b.bias',
    'alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/output_projection//weights': 'structure_module.ipa.linear_out.weight',
    'alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/output_projection//bias': 'structure_module.ipa.linear_out.bias',
    'alphafold/alphafold_iteration/structure_module/fold_iteration/attention_layer_norm//scale': 'structure_module.layer_norm_ipa.weight',
    'alphafold/alphafold_iteration/structure_module/fold_iteration/attention_layer_norm//offset': 'structure_module.layer_norm_ipa.bias',
    'alphafold/alphafold_iteration/structure_module/fold_iteration/transition//weights': 'structure_module.transition.layers.0.linear_1.weight',
    'alphafold/alphafold_iteration/structure_module/fold_iteration/transition//bias': 'structure_module.transition.layers.0.linear_1.bias',
    'alphafold/alphafold_iteration/structure_module/fold_iteration/transition_1//weights': 'structure_module.transition.layers.0.linear_2.weight',
    'alphafold/alphafold_iteration/structure_module/fold_iteration/transition_1//bias': 'structure_module.transition.layers.0.linear_2.bias',
    'alphafold/alphafold_iteration/structure_module/fold_iteration/transition_2//weights': 'structure_module.transition.layers.0.linear_3.weight',
    'alphafold/alphafold_iteration/structure_module/fold_iteration/transition_2//bias': 'structure_module.transition.layers.0.linear_3.bias',
    'alphafold/alphafold_iteration/structure_module/fold_iteration/transition_layer_norm//scale': 'structure_module.transition.layer_norm.weight',
    'alphafold/alphafold_iteration/structure_module/fold_iteration/transition_layer_norm//offset': 'structure_module.transition.layer_norm.bias',
    'alphafold/alphafold_iteration/structure_module/fold_iteration/quat_rigid/rigid//weights': 'structure_module.bb_update.linear.weight',
    'alphafold/alphafold_iteration/structure_module/fold_iteration/quat_rigid/rigid//bias': 'structure_module.bb_update.linear.bias',
}
ipa = 'alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention/'


def load_alphafold2_params(alphafold_path: str, model: nn.Module, device, dtype):
    print('Loading pretrained parameters from alphafold2...')
    # Load parameters
    pretrained_params_npz = np.load(alphafold_path)
    param_dict = {}

    for file_name in pretrained_params_npz.files:
        if file_name in param_map_dict.keys():
            key = param_map_dict[file_name]
            param = pretrained_params_npz[file_name]
            if param.ndim == 2:
                param = param.transpose([1, 0])
            param_dict[key] = param

    # Origin
    param_dict['structure_module.ipa.linear_q.weight'] = pretrained_params_npz[f'{ipa}q_scalar_projection//weights'].reshape((384, 192)).transpose([1, 0])
    param_dict['structure_module.ipa.linear_q_points.weight'] = pretrained_params_npz[f'{ipa}q_point_projection/point_projection//weights'].reshape((384, 144)).transpose([1, 0])
    param_dict['structure_module.ipa.linear_q_points.bias'] = pretrained_params_npz[f'{ipa}q_point_projection/point_projection//bias'].reshape((144, ))
    param_dict['structure_module.ipa.linear_kv.weight'] = np.concatenate([pretrained_params_npz[f'{ipa}k_scalar_projection//weights'].reshape((384, 192)).transpose([1, 0]), pretrained_params_npz[f'{ipa}v_scalar_projection//weights'].reshape((384, 192)).transpose([1, 0])], axis=0)
    param_dict['structure_module.ipa.linear_kv_points.weight'] = np.concatenate([pretrained_params_npz[f'{ipa}k_point_projection/point_projection//weights'].reshape((384, 144)).transpose([1, 0]), pretrained_params_npz[f'{ipa}v_point_projection/point_projection//weights'].reshape((384, 288)).transpose([1, 0])], axis=0)
    param_dict['structure_module.ipa.linear_kv_points.bias'] = np.concatenate([pretrained_params_npz[f'{ipa}k_point_projection/point_projection//bias'].reshape((144,)), pretrained_params_npz[f'{ipa}v_point_projection/point_projection//bias'].reshape((288,))], axis=0)

    # Load parameters for each layer
    for layer_param in model.named_parameters():
        layer_param[1].data = torch.tensor(param_dict[layer_param[0]], device=device, dtype=dtype)
    

