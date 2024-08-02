#!/bin/bash

python predict.py \
    --fasta_path example/4BJQ.pdb5.fasta \
    --cpus 12 \
    --uniclust30_database_path /media/puqing/alphafold_data/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
    --hhblits_binary_path `which hhblits` \
    --alphafold_param_path params/params_model_1_multimer_v2.npz \
    --alphafold_model_name model_1_multimer_v2 \
    --param_path params/fine-tuned.pt \
