from functools import partial

import torch
import torch.nn as nn
from typing import Tuple, Dict

from alphafold.model.primitives import Linear, LayerNorm

class InputEmbedderMultimer(nn.Module):
    """
    Embeds a subset of the input features.

    Implements Algorithms 3 (InputEmbedder) and 4 (relpos).
    """

    def __init__(
        self,
        tf_dim: int,
        msa_dim: int,
        c_z: int,
        c_m: int,
        max_relative_idx: int,
        use_chain_relative: bool,
        max_relative_chain: int,
        **kwargs,
    ):
        """
        Args:
            tf_dim:
                Final dimension of the target features
            msa_dim:
                Final dimension of the MSA features
            c_z:
                Pair embedding dimension
            c_m:
                MSA embedding dimension
            relpos_k:
                Window size used in relative positional encoding
        """
        super(InputEmbedderMultimer, self).__init__()

        self.tf_dim = tf_dim
        self.msa_dim = msa_dim

        self.c_z = c_z
        self.c_m = c_m

        self.linear_tf_z_i = Linear(tf_dim, c_z)
        self.linear_tf_z_j = Linear(tf_dim, c_z)
        self.linear_tf_m = Linear(tf_dim, c_m)
        self.linear_msa_m = Linear(msa_dim, c_m)

        # RPE stuff
        self.max_relative_idx = max_relative_idx
        self.use_chain_relative = use_chain_relative
        self.max_relative_chain = max_relative_chain
        if self.use_chain_relative:
            self.no_bins = 2 * max_relative_idx + 2 + 1 + 2 * max_relative_chain + 2
        else:
            self.no_bins = 2 * max_relative_idx + 1
        self.linear_relpos = Linear(self.no_bins, c_z)

    def relpos(self, batch: Dict[str, torch.Tensor]):
        pos = batch["residue_index"]
        asym_id = batch["asym_id"]
        asym_id_same = asym_id[..., None] == asym_id[..., None, :]
        offset = pos[..., None] - pos[..., None, :]

        clipped_offset = torch.clamp(
            offset + self.max_relative_idx, 0, 2 * self.max_relative_idx
        )

        rel_feats = []
        if self.use_chain_relative:
            final_offset = torch.where(
                asym_id_same,
                clipped_offset,
                (2 * self.max_relative_idx + 1) * torch.ones_like(clipped_offset),
            )

            rel_pos = torch.nn.functional.one_hot(
                final_offset,
                2 * self.max_relative_idx + 2,
            )

            rel_feats.append(rel_pos)

            entity_id = batch["entity_id"]
            entity_id_same = entity_id[..., None] == entity_id[..., None, :]
            rel_feats.append(entity_id_same[..., None])

            sym_id = batch["sym_id"]
            rel_sym_id = sym_id[..., None] - sym_id[..., None, :]

            max_rel_chain = self.max_relative_chain
            clipped_rel_chain = torch.clamp(
                rel_sym_id + max_rel_chain,
                0,
                2 * max_rel_chain,
            )

            final_rel_chain = torch.where(
                entity_id_same,
                clipped_rel_chain,
                (2 * max_rel_chain + 1) * torch.ones_like(clipped_rel_chain),
            )

            rel_chain = torch.nn.functional.one_hot(
                final_rel_chain.long(),
                2 * max_rel_chain + 2,
            )

            rel_feats.append(rel_chain)
        else:
            rel_pos = torch.nn.functional.one_hot(
                clipped_offset,
                2 * self.max_relative_idx + 1,
            )
            rel_feats.append(rel_pos)

        rel_feat = torch.cat(rel_feats, dim=-1).to(self.linear_relpos.weight.dtype)

        return self.linear_relpos(rel_feat)

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tf = batch["target_feat"]
        msa = batch["msa_feat"]

        # [*, N_res, c_z]
        tf_emb_i = self.linear_tf_z_i(tf)
        tf_emb_j = self.linear_tf_z_j(tf)

        # [*, N_res, N_res, c_z]
        pair_emb = tf_emb_i[..., None, :] + tf_emb_j[..., None, :, :]
        pair_emb = pair_emb + self.relpos(batch)

        # [*, N_clust, N_res, c_m]
        n_clust = msa.shape[-3]
        tf_m = (
            self.linear_tf_m(tf)
            .unsqueeze(-3)
            .expand(((-1,) * len(tf.shape[:-2]) + (n_clust, -1, -1)))
        )
        msa_emb = self.linear_msa_m(msa) + tf_m

        return msa_emb, pair_emb
