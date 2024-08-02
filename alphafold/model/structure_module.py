# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple, Union

from alphafold.model.primitives import Linear, LayerNorm, ipa_point_weights_init_
from alphafold.common.residue_constants import (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
)
from alphafold.utils.geometry.quat_rigid import QuatRigid
from alphafold.utils.geometry.rigid_matrix_vector import Rigid3Array
from alphafold.utils.geometry.vector import Vec3Array
from alphafold.utils.rigid_utils import Rotation, Rigid
from alphafold.utils.tensor_utils import (
    dict_multimap,
    permute_final_dims,
    flatten_final_dims,
)
from alphafold.utils.gln_utils import backb_pos_to_gln_matrix
from alphafold.utils.feats import atom14_to_atom37


class PointProjection(nn.Module):
    def __init__(
        self,
        c_hidden: int,
        num_points: int,
        no_heads: int,
        return_local_points: bool = False,
    ):
        super().__init__()
        self.return_local_points = return_local_points
        self.no_heads = no_heads

        self.linear = Linear(c_hidden, no_heads * 3 * num_points)

    def forward(
        self,
        activations: torch.Tensor,
        rigids: Rigid3Array,
    ) -> Union[Vec3Array, Tuple[Vec3Array, Vec3Array]]:
        # TODO: Needs to run in high precision during training
        points_local = self.linear(activations)
        points_local = points_local.reshape(
            *points_local.shape[:-1],
            self.no_heads,
            -1,
        )
        points_local = torch.split(points_local, points_local.shape[-1] // 3, dim=-1)
        points_local = Vec3Array(*points_local)
        points_global = rigids[..., None, None].apply_to_point(points_local)

        if self.return_local_points:
            return points_global, points_local

        return points_global


class InvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        inf: float = 1e5,
        eps: float = 1e-8,
        is_multimer: bool = False,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(InvariantPointAttention, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps
        self.is_multimer = is_multimer

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        if not self.is_multimer:
            hc = self.c_hidden * self.no_heads
            self.linear_q = Linear(self.c_s, hc, bias=(not is_multimer))
            self.linear_kv = Linear(self.c_s, 2 * hc)

            hpq = self.no_heads * self.no_qk_points * 3
            self.linear_q_points = Linear(self.c_s, hpq)

            hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
            self.linear_kv_points = Linear(self.c_s, hpkv)

            # hpv = self.no_heads * self.no_v_points * 3

        else:
            hc = self.c_hidden * self.no_heads
            self.linear_q = Linear(self.c_s, hc, bias=(not is_multimer))
            self.linear_q_points = PointProjection(
                self.c_s, self.no_qk_points, self.no_heads
            )

            self.linear_k = Linear(self.c_s, hc, bias=False)
            self.linear_v = Linear(self.c_s, hc, bias=False)
            self.linear_k_points = PointProjection(
                self.c_s,
                self.no_qk_points,
                self.no_heads,
            )

            self.linear_v_points = PointProjection(
                self.c_s,
                self.no_v_points,
                self.no_heads,
            )
        self.linear_b = Linear(self.c_z, self.no_heads)

        self.head_weights = nn.Parameter(torch.zeros((no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.no_heads * (
            self.c_z + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        r: Union[Rigid, Rigid3Array],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        #######################################
        # Generate scalar and point activations
        #######################################

        # The following two blocks are equivalent
        # They're separated only to preserve compatibility with old AF weights
        if self.is_multimer:
            # [*, N_res, H * C_hidden]
            q = self.linear_q(s)

            # [*, N_res, H, C_hidden]
            q = q.view(q.shape[:-1] + (self.no_heads, -1))

            # [*, N_res, H, P_qk]
            q_pts = self.linear_q_points(s, r)
            # [*, N_res, H * C_hidden]
            k = self.linear_k(s)
            v = self.linear_v(s)

            # [*, N_res, H, C_hidden]
            k = k.view(k.shape[:-1] + (self.no_heads, -1))
            v = v.view(v.shape[:-1] + (self.no_heads, -1))

            # [*, N_res, H, P_qk, 3]
            k_pts = self.linear_k_points(s, r)

            # [*, N_res, H, P_v, 3]
            v_pts = self.linear_v_points(s, r)
        else:
            # [*, N_res, H * C_hidden]
            q = self.linear_q(s)
            kv = self.linear_kv(s)

            # [*, N_res, H, C_hidden]
            q = q.view(q.shape[:-1] + (self.no_heads, -1))

            # [*, N_res, H, 2 * C_hidden]
            kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

            # [*, N_res, H, C_hidden]
            k, v = torch.split(kv, self.c_hidden, dim=-1)

            # [*, N_res, H * P_q * 3]
            q_pts = self.linear_q_points(s)

            # This is kind of clunky, but it's how the original does it
            # [*, N_res, H * P_q, 3]
            q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
            q_pts = torch.stack(q_pts, dim=-1)
            q_pts = r[..., None].apply(q_pts)

            # [*, N_res, H, P_q, 3]
            q_pts = q_pts.view(q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3))

            # [*, N_res, H * (P_q + P_v) * 3]
            kv_pts = self.linear_kv_points(s)

            # [*, N_res, H * (P_q + P_v), 3]
            kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
            kv_pts = torch.stack(kv_pts, dim=-1)
            kv_pts = r[..., None].apply(kv_pts)

            # [*, N_res, H, (P_q + P_v), 3]
            kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

            # [*, N_res, H, P_q/P_v, 3]
            k_pts, v_pts = torch.split(
                kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
            )

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z)

        # [*, H, N_res, N_res]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1))

        if self.is_multimer:
            # [*, N_res, N_res, H, P_q, 3]
            pt_att = q_pts[..., None, :, :] - k_pts[..., None, :, :, :]
            # [*, N_res, N_res, H, P_q]
            pt_att = sum([c**2 for c in pt_att])
        else:
            # [*, N_res, N_res, H, P_q, 3]
            ######################################
            q_pts_t0 = q_pts.unsqueeze(-4)
            q_shape = q_pts_t0.shape
            q_pts_t0 = q_pts_t0.reshape([q_shape[0], q_shape[1], -1])
            k_pts_t0 = k_pts.unsqueeze(-5)
            k_shape = k_pts_t0.shape
            k_pts_t0 = k_pts_t0.reshape([k_shape[0], k_shape[1], -1])
            q_k = q_pts_t0 - k_pts_t0
            q_k = q_k ** 2
            q_k_shape = q_k.shape
            pt_att = q_k.reshape(q_k_shape[:2] + q_shape[-3:])
            #####################################
            pt_att = pt_att.permute(0, 4, 1, 2, 3)
            pt_att = torch.sum(pt_att, 1)

        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        ##############################
        pt_att_t0 = pt_att.permute(0, 3, 1, 2)
        head_weights_t0 = head_weights.permute(0, 3, 1, 2)
        pt_att_o = pt_att_t0 * head_weights_t0
        pt_att = pt_att_o.permute(0, 2,3, 1)
        ##############################

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))
        a = a + pt_att
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(a, v.transpose(-2, -3).to(dtype=a.dtype)).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # As DeepMind explains, this manual matmul ensures that the operation
        # happens in float32.
        if self.is_multimer:
            # [*, N_res, H, P_v]
            o_pt = v_pts * permute_final_dims(a, (1, 2, 0)).unsqueeze(-1)
            o_pt = o_pt.sum(dim=-3)

            # [*, N_res, H, P_v]
            o_pt = r[..., None, None].apply_inverse_to_point(o_pt)

            # [*, N_res, H * P_v, 3]
            o_pt = o_pt.reshape(o_pt.shape[:-2] + (-1,))

            # [*, N_res, H * P_v]
            o_pt_norm = o_pt.norm(self.eps)
        else:
            # [*, H, 3, N_res, P_v]
            ###################################
            a1 = a[..., None, :, :, None]
            a1 = a1.permute(0, 1, 2, 4, 3)
            b = permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            b = b.permute(0, 1, 2, 4, 3)
            c = a1 * b
            o_pt = torch.sum(c, -1)
            ###################################

            # [*, N_res, H, P_v, 3]
            o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
            o_pt = r[..., None, None].invert_apply(o_pt)

            # [*, N_res, H * P_v]
            o_pt_norm = flatten_final_dims(
                torch.sqrt(torch.sum(o_pt**2, dim=-1) + self.eps), 2
            )

            # [*, N_res, H * P_v, 3]
            o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        # [*, N_res, H, C_z]
        o_pair = torch.matmul(a.transpose(-2, -3), z.to(dtype=a.dtype))

        # [*, N_res, H * C_z]
        o_pair = flatten_final_dims(o_pair, 2)

        # [*, N_res, C_s]
        if self.is_multimer:
            s = self.linear_out(
                torch.cat((o, *o_pt, o_pt_norm, o_pair), dim=-1).to(dtype=z.dtype)
            )
        else:
            s = self.linear_out(
                torch.cat(
                    (o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pair), dim=-1
                ).to(dtype=z.dtype)
            )

        return s

class BackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, c_s: int):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super(BackboneUpdate, self).__init__()

        self.c_s = c_s

        self.linear = Linear(self.c_s, 6, init="final")

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector 
        """
        # [*, 6]
        update = self.linear(s)

        return update 


class StructureModuleTransitionLayer(nn.Module):
    def __init__(self, c: int):
        super(StructureModuleTransitionLayer, self).__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")

        self.relu = nn.ReLU()

    def forward(self, s: torch.Tensor):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        s = s + s_initial

        return s


class StructureModuleTransition(nn.Module):
    def __init__(self, c: int, num_layers: int, dropout_rate: float):
        super(StructureModuleTransition, self).__init__()

        self.c = c
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            l = StructureModuleTransitionLayer(self.c)
            self.layers.append(l)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = LayerNorm(self.c)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            s = l(s)

        s = self.dropout(s)
        s = self.layer_norm(s)

        return s


class StructureModule(nn.Module):
    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_ipa: int,
        c_resnet: int,
        no_heads_ipa: int,
        no_qk_points: int,
        no_v_points: int,
        dropout_rate: float,
        no_blocks: int,
        no_transition_layers: int,
        no_resnet_blocks: int,
        no_angles: int,
        trans_scale_factor: float,
        epsilon: float,
        inf: float,
        is_multimer: bool = False,
        **kwargs,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_ipa:
                IPA hidden channel dimension
            c_resnet:
                Angle resnet (Alg. 23 lines 11-14) hidden channel dimension
            no_heads_ipa:
                Number of IPA heads
            no_qk_points:
                Number of query/key points to generate during IPA
            no_v_points:
                Number of value points to generate during IPA
            dropout_rate:
                Dropout rate used throughout the layer
            no_blocks:
                Number of structure module blocks
            no_transition_layers:
                Number of layers in the single representation transition
                (Alg. 23 lines 8-9)
            no_resnet_blocks:
                Number of blocks in the angle resnet
            no_angles:
                Number of angles to generate in the angle resnet
            trans_scale_factor:
                Scale of single representation transition hidden dimension
            epsilon:
                Small number used in angle resnet normalization
            inf:
                Large number used for attention masking
            is_multimer:
                whether running under multimer mode
        """
        super(StructureModule, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_ipa = c_ipa
        self.c_resnet = c_resnet
        self.no_heads_ipa = no_heads_ipa
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.dropout_rate = dropout_rate
        self.no_blocks = no_blocks
        self.no_transition_layers = no_transition_layers
        self.no_resnet_blocks = no_resnet_blocks
        self.no_angles = no_angles
        self.trans_scale_factor = trans_scale_factor
        self.epsilon = epsilon
        self.inf = inf
        self.is_multimer = is_multimer

        # To be lazily initialized later
        self.default_frames = None
        self.group_idx = None
        self.atom_mask = None
        self.lit_positions = None

        self.layer_norm_s = LayerNorm(self.c_s)
        self.layer_norm_z = LayerNorm(self.c_z)

        self.linear_in = Linear(self.c_s, self.c_s)

        self.ipa = InvariantPointAttention(
            self.c_s,
            self.c_z,
            self.c_ipa,
            self.no_heads_ipa,
            self.no_qk_points,
            self.no_v_points,
            inf=self.inf,
            eps=self.epsilon,
            is_multimer=self.is_multimer,
        )

        self.ipa_dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm_ipa = LayerNorm(self.c_s)

        self.transition = StructureModuleTransition(
            self.c_s,
            self.no_transition_layers,
            self.dropout_rate,
        )

        self.bb_update = QuatRigid(self.c_s, full_quat=False)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        aatype: torch.Tensor,
        asym_id: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            aatype:
                [*, N_res] amino acid indices
            mask:
                Optional [*, N_res] sequence mask
        Returns:
            A dictionary of outputs
        """
        if mask is None:
            # [*, N]
            mask = s.new_ones(s.shape[:-1])

        # [*, N, C_s]
        s = self.layer_norm_s(s)

        # [*, N, N, C_z]
        z = self.layer_norm_z(z)

        # [*, N, C_s]
        s_initial = s
        s = self.linear_in(s)

        # [*, N]
        rigids = Rigid3Array.identity(
            s.shape[:-1],
            s.device,
        )
        outputs = []
        for i in range(self.no_blocks):
            # [*, N, C_s]
            s = s + self.ipa(s, z, rigids, mask)
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)

            # [*, N]
            rigids = rigids @ self.bb_update(s)

            # To hew as closely as possible to AlphaFold, we convert our
            # quaternion-based transformations to rotation-matrix ones
            # here
            # [*, N]
            
            backb_frames_to_global = rigids.scale_translation(
                self.trans_scale_factor
            )

            # [*, N, 1]
            backb_frames_to_global = backb_frames_to_global.unsqueeze(-1)

            # [*, N, 4, 3]
            pred_xyz = self.frames_and_literature_positions_to_backb_pos(
                backb_frames_to_global,
                aatype,
            )

            preds = {
                "frames": rigids.scale_translation(self.trans_scale_factor).to_tensor(),
                "positions": pred_xyz,
                "gln_matrix": backb_pos_to_gln_matrix(pred_xyz[..., :3, :], asym_id)
            }

            outputs.append(preds)

            if i < (self.no_blocks - 1):
                rigids = rigids.stop_rot_gradient()

        outputs = dict_multimap(torch.stack, outputs)

        return outputs

    def _init_residue_constants(self, float_dtype: torch.dtype, device: torch.device):
        if self.default_frames is None:
            self.default_frames = torch.tensor(
                restype_rigid_group_default_frame,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )
        if self.group_idx is None:
            self.group_idx = torch.tensor(
                restype_atom14_to_rigid_group,
                device=device,
                requires_grad=False,
            )
        if self.atom_mask is None:
            self.atom_mask = torch.tensor(
                restype_atom14_mask,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )
        if self.lit_positions is None:
            self.lit_positions = torch.tensor(
                restype_atom14_rigid_group_positions,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )

    def frames_and_literature_positions_to_backb_pos(
        self, r: Rigid3Array, aatype  # [*, N, 4]  # [*, N]
    ):
        # Lazily initialize the residue constants on the correct device
        if type(r) == Rigid3Array:
            self._init_residue_constants(r.dtype, r.device)
        else:
            raise ValueError("Unknown rigid type")
        
        # [*, N, 5], repeat r for 5 times
        r = Rigid3Array.cat([r]*5, dim=-1,)

        # [21, 14, 3] -> [*, N, 14, 3]
        lit_positions = self.lit_positions[aatype.long(), ...]

        # [*, N, 5, 3], need Cb position for recyclying
        # Cb index is 4 in atom14 but 3 in atom37
        pred_positions = r.apply(lit_positions[..., :5, :]).to_tensor()

        # [*, N, 4, 3], Transfer Cb index from 4 to 3
        pred_positions = pred_positions[..., [0, 1, 2, 4], :]

        return pred_positions
