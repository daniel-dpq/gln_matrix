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

from enum import Enum
from dataclasses import dataclass
from functools import partial
import numpy as np
import torch
from typing import Union, List


_NPZ_KEY_PREFIX = "alphafold/alphafold_iteration/"


# With Param, a poor man's enum with attributes (Rust-style)
class ParamType(Enum):
    LinearWeight = partial(  # hack: partial prevents fns from becoming methods
        lambda w: w.transpose(-1, -2)
    )
    LinearWeightMHA = partial(
        lambda w: w.reshape(*w.shape[:-2], -1).transpose(-1, -2)
    )
    LinearMHAOutputWeight = partial(
        lambda w: w.reshape(*w.shape[:-3], -1, w.shape[-1]).transpose(-1, -2)
    )
    LinearBiasMHA = partial(lambda w: w.reshape(*w.shape[:-2], -1))
    LinearWeightOPM = partial(
        lambda w: w.reshape(*w.shape[:-3], -1, w.shape[-1]).transpose(-1, -2)
    )
    Other = partial(lambda w: w)

    def __init__(self, fn):
        self.transformation = fn


@dataclass
class Param:
    param: Union[torch.Tensor, List[torch.Tensor]]
    param_type: ParamType = ParamType.Other
    stacked: bool = False


def process_translation_dict(d, top_layer=True):
    flat = {}
    for k, v in d.items():
        if type(v) == dict:
            prefix = _NPZ_KEY_PREFIX if top_layer else ""
            sub_flat = {
                (prefix + "/".join([k, k_prime])): v_prime
                for k_prime, v_prime in process_translation_dict(
                    v, top_layer=False
                ).items()
            }
            flat.update(sub_flat)
        else:
            k = "/" + k if not top_layer else k
            flat[k] = v

    return flat


def stacked(param_dict_list, out=None):
    """
    Args:
        param_dict_list:
            A list of (nested) Param dicts to stack. The structure of
            each dict must be the identical (down to the ParamTypes of
            "parallel" Params). There must be at least one dict
            in the list.
    """
    if out is None:
        out = {}
    template = param_dict_list[0]
    for k, _ in template.items():
        v = [d[k] for d in param_dict_list]
        if type(v[0]) is dict:
            out[k] = {}
            stacked(v, out=out[k])
        elif type(v[0]) is Param:
            stacked_param = Param(
                param=[param.param for param in v],
                param_type=v[0].param_type,
                stacked=True,
            )

            out[k] = stacked_param

    return out


def assign(translation_dict, orig_weights):
    for k, param in translation_dict.items():
        with torch.no_grad():
            weights = torch.as_tensor(orig_weights[k])
            ref, param_type = param.param, param.param_type
            if param.stacked:
                weights = torch.unbind(weights, 0)
            else:
                weights = [weights]
                ref = [ref]

            try:
                weights = list(map(param_type.transformation, weights))
                for p, w in zip(ref, weights):
                    p.copy_(w)
            except:
                print(k)
                print(ref[0].shape)
                print(weights[0].shape)
                raise


def generate_translation_dict(model, version):
    #######################
    # Some templates
    #######################

    LinearWeight = lambda l: (Param(l, param_type=ParamType.LinearWeight))

    LinearBias = lambda l: (Param(l))

    LinearParams = lambda l: {
        "weights": LinearWeight(l.weight),
        "bias": LinearBias(l.bias),
    }

    LayerNormParams = lambda l: {
        "scale": Param(l.weight),
        "offset": Param(l.bias),
    }

    # see commit b88f8da on the Alphafold repo
    # Alphafold swaps the pseudocode's a and b between the incoming/outcoming
    # iterations of triangle multiplication, which is confusing and not
    # reproduced in our implementation.
    IPAParams = lambda ipa: {
        "q_scalar": LinearParams(ipa.linear_q),
        "kv_scalar": LinearParams(ipa.linear_kv),
        "q_point_local": LinearParams(ipa.linear_q_points),
        "kv_point_local": LinearParams(ipa.linear_kv_points),
        "trainable_point_weights": Param(
            param=ipa.head_weights, param_type=ParamType.Other
        ),
        "attention_2d": LinearParams(ipa.linear_b),
        "output_projection": LinearParams(ipa.linear_out),
    }

    FoldIterationParams = lambda sm: {
        "invariant_point_attention": IPAParams(sm.ipa),
        "attention_layer_norm": LayerNormParams(sm.layer_norm_ipa),
        "transition": LinearParams(sm.transition.layers[0].linear_1),
        "transition_1": LinearParams(sm.transition.layers[0].linear_2),
        "transition_2": LinearParams(sm.transition.layers[0].linear_3),
        "transition_layer_norm": LayerNormParams(sm.transition.layer_norm),
        "affine_update": LinearParams(sm.bb_update.linear),
    }

    ############################
    # translations dict overflow
    ############################

    translations = {
        "structure_module": {
            "single_layer_norm": LayerNormParams(
                model.layer_norm_s
            ),
            "initial_projection": LinearParams(
                model.linear_in
            ),
            "pair_layer_norm": LayerNormParams(
                model.layer_norm_z
            ),
            "fold_iteration": FoldIterationParams(model),
        },
    }

    return translations


def import_jax_weights_(model, npz_path, version="model_1"):
    data = np.load(npz_path)

    translations = generate_translation_dict(model, version)

    # Flatten keys and insert missing key prefixes
    flat = process_translation_dict(translations)

    # Sanity check
    keys = list(data.keys())
    flat_keys = list(flat.keys())
    incorrect = [k for k in flat_keys if k not in keys]
    missing = [k for k in keys if k not in flat_keys]
    # print(f"Incorrect: {incorrect}")
    # print(f"Missing: {missing}")

    assert len(incorrect) == 0
    # assert(sorted(list(flat.keys())) == sorted(list(data.keys())))

    # Set weights
    assign(flat, data)
