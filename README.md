# GLN Matrix Prediction
This is the code for data preprocessing, model training and evaluation described 
in the paper "Deep Learning-Assisted Discovery of Protein Entangling Motifs".

# Installation

To run our codes, you will need:
+ Python 3.8 or 3.9.
+ [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) 11.3 or above
+ PyTorch 1.12 or above 

Lines below would create a new conda environment called "gln_matrix":

```shell
git clone https://github.com/daniel-dpq/gln_matrix.git
cd gln_matrix
conda env create --name=gln_matrix -f environment.yml
conda activate gln_matrix
```

You will need to download [Uniclust30 database](https://gwdu111.gwdg.de/~compbiol/uniclust/2023_02/)
 (~ 86G) for MSA searching

Alternatively, you can follow the instructions at [OpenFold Page](https://github.com/aqlaboratory/openfold) 
to set up your environment.

## Parameters

Parameter files you may use for your own trial are deposited in `params/`. `params_model_1_multimer_v2.npz` 
is the original AlphaFold2 parameters we used in our paper. `fine-tuned.pt` and `finalpoint.pt` are the final
checkpoint and the checkpoint with the minimum validation loss during our finetuning. Data reported in 
our paper was based on `fine-tuned.pt`. Pass `--param_path` in `predict.sh` to specify `.pt` parameter path and 
pass `--alphafold_param_path` to sepcify the original AlphaFold2 parameter path.

## Data

Data used for our training, validation and test are deposited in `data/`

## Input file

To predict the GLN matrix of a homodimer, you need to prepare a fasta file containing two identical 
sequences with different sequence ids. Examples are given in the `example/` directory.

## Inference

we provide a bash script `predict.sh` and some sequence examples in `example/` for model inference. Please 
change the uniclust30_database_path before running `./predict.sh`

## Output

`predict.py` gives multi-sequence alignments, prediction timings and generated features for prediction in the output directory. The predicted GLN matrices were given in both `.txt` and `.png` formats. Example outputs are given in `out/`