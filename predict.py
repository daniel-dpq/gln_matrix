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

import argparse
import os
import random
import json
import time
from datetime import date
import tempfile
import contextlib
import pathlib
from absl import logging

import numpy as np
import torch
from torchsummary import summary
import pickle
import shutil

from alphafold.model.alphafold import AlphaFold
from alphafold.config import model_config
from alphafold.data import data_pipeline, feature_pipeline
from alphafold.utils.import_weights import import_jax_weights_
from alphafold.utils.gln_utils import draw_single_heatmap

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f'Using device {device}')

logging.set_verbosity(logging.INFO)

if int(torch.__version__.split(".")[0]) >= 1 and int(torch.__version__.split(".")[1]) > 11:
    torch.backends.cuda.matmul.allow_tf32 = True

KEYS_TO_REMOVE = [
    'atom14_atom_exists', 'residx_atom14_to_atom37', 'residx_atom37_to_atom14', 
    'atom37_atom_exists', 'deletion_matrix', 'bert_mask', 'true_msa', 'seq_length',
    'cluster_profile', 'cluster_deletion_mean', 'msa_profile', 'ensemble_index',
    'num_alignments',
]


@contextlib.contextmanager
def temp_fasta_file(fasta_str: str):
    with tempfile.NamedTemporaryFile('w', suffix='.fasta') as fasta_file:
        fasta_file.write(fasta_str)
        fasta_file.seek(0)
        yield fasta_file.name


def predict(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    data_processor: data_pipeline.DataPipelineMultimer,
    feature_processor: feature_pipeline.FeaturePipeline,
    alignment_runner: data_pipeline.AlignmentRunnerMultimer,
    config,
    args,
):
    logging.info('Predicting %s', fasta_name)
    timings = {}
    output_dir = os.path.join(output_dir_base, fasta_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # If we already have feature.pkl file, skip the MSA and template finding step
    t0 = time.time()
    features_output_path = os.path.join(output_dir, 'features.pkl')    
    if os.path.exists(features_output_path):
        logging.info('Loading features from %s', features_output_path)
        batch = pickle.load(open(features_output_path, 'rb'))
    else:
        # Search for MSAs if no precomputed alignments
        if(not args.use_precomputed_alignments):
            alignment_dir = os.path.join(output_dir, "alignments")
            if not os.path.exists(alignment_dir):
                os.makedirs(alignment_dir)

            alignment_runner.run_msa_tools(
                fasta_path=fasta_path,
                fasta_name=fasta_name,
                alignment_dir=alignment_dir,
            )
        else:
            alignment_dir = args.use_precomputed_alignments

        feature_dict = data_processor.process_fasta(
            fasta_path=fasta_path, fasta_name=fasta_name, alignment_dir=alignment_dir
        )
        processed_feature_dict = feature_processor.process_features(
            feature_dict, mode='predict', is_multimer=True,
        )
        for k in KEYS_TO_REMOVE:
            if k in processed_feature_dict.keys():
                del processed_feature_dict[k]
        batch = processed_feature_dict
        # Write out features as a pickled dictionary.
        with open(features_output_path, 'wb') as f:
            pickle.dump(batch, f, protocol=4)

    timings['process_features'] = time.time() - t0
    logging.info('Total feature processing time: %.1fs', timings['process_features'])

    # Set up model
    model = AlphaFold(config)

    # Load weights
    if args.param_path is not None:
        print('Loading our model parameters...')
        model.load_state_dict(torch.load(args.param_path, map_location=torch.device('cpu')))
    elif args.alphafold_param_path is not None and args.alphafold_model_name is not None:
        print('Loading AlphaFold parameters...')
        import_jax_weights_(
            model, args.alphafold_param_path, version=args.alphafold_model_name)
    else:
        raise ValueError('No model parameters provided.')

    # Move to device
    model = model.to(device)
    for k, v in batch.items():
        batch[k] = v.to(device)
    
    # Run model
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        out = model(batch)
    timings['prediction'] = time.time() - t0
    logging.info('Total prediction time: %.1fs', timings['prediction'])
    logging.info(
        "Output shapes: {'final_backb_positions': %s 'final_gln_matrix': %s}",
        out['final_backb_positions'].cpu().numpy().shape, 
        out['final_gln_matrix'].cpu().numpy().shape,
    )
    
    # Get and save results
    gln_matrix = out['final_gln_matrix'].cpu().numpy()
    gln = np.sum(gln_matrix)
    logging.info('Preidicted GLN: %.3f', gln)

    matrix_output_path = os.path.join(output_dir, f'gln_matrix.txt')
    np.savetxt(matrix_output_path, gln_matrix)

    heatmap_output_path = os.path.join(output_dir, f'gln_matrix.png')
    draw_single_heatmap(
        gln_matrix, 
        f'Predicted GLN Matrix of {fasta_name}', 
        heatmap_output_path
    )

    timings_output_path = os.path.join(output_dir, 'timings.json')
    with open(timings_output_path, 'w') as f:
        f.write(json.dumps(timings, indent=4))

    del out


def main(args):
    logging.info('running in multimer mode...')
    config = model_config()
    config.globals.chunk_size = args.chunk_size

    fasta_path = args.fasta_path
    fasta_name = pathlib.Path(fasta_path).stem
    output_dir_base = args.output_dir
    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)
    
    if(not args.use_precomputed_alignments):
        alignment_runner = data_pipeline.AlignmentRunnerMultimer(
            hhblits_binary_path=args.hhblits_binary_path,
            jackhmmer_binary_path=args.jackhmmer_binary_path,
            uniclust30_database_path=args.uniclust30_database_path,
            uniprot_database_path=args.uniprot_database_path,
            no_cpus=args.cpus
        )
    else:
        alignment_runner = None

    monomer_data_processor = data_pipeline.DataPipeline()

    data_processor = data_pipeline.DataPipelineMultimer(
        monomer_data_pipeline=monomer_data_processor,
    )

    feature_processor = feature_pipeline.FeaturePipeline(config.data)

    predict(
        fasta_path=fasta_path,
        fasta_name=fasta_name,
        output_dir_base=output_dir_base,
        data_processor=data_processor,
        feature_processor=feature_processor,
        alignment_runner=alignment_runner,
        config=config,
        args=args,
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta_path", type=str)
    parser.add_argument(
        "--use_precomputed_alignments", type=str, default=None,
        help="""Path to alignment directory. If provided, alignment computation 
            is skipped and database path arguments are ignored."""
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.path.join(os.getcwd(), 'out'),
        help="""Name of the directory in which to output the prediction""",
    )
    parser.add_argument(
        "--param_path", type=str, default=None, 
        help="""Path to our model parameters. If none, the model will load original parameters from AlphaFold."""
    )
    parser.add_argument(
        "--alphafold_param_path", type=str, default=None, 
        help="""Path to alphafold model parameters."""
    )
    parser.add_argument(
        "--alphafold_model_name", type=str, default=None, 
        help="""The name of alphafold model to load parameters."""
    )
    parser.add_argument('--chunk_size', type=int, default=None)
    parser.add_argument(
        "--cpus", type=int, default=12,
        help="""Number of CPUs with which to run alignment tools"""
    )
    parser.add_argument('--data_random_seed', type=str, default=None)
    parser.add_argument(
        '--uniclust30_database_path',
        type=str,
        default=None,
    )
    parser.add_argument(
        "--uniprot_database_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        '--hhblits_binary_path', 
        type=str, 
        default='/usr/bin/hhblits'
    )
    parser.add_argument(
        '--jackhmmer_binary_path', 
        type=str, 
        default='/usr/bin/jackhmmer'
    )
    args = parser.parse_args()

    main(args)
