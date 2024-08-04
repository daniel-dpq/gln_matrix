# GLN Matrix Prediction
This is the code for data preprocessing, prediction described in the paper 
"Deep Learning-Assisted Discovery of Protein Entangling Motifs".

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

You will need to download Uniclust30 database https://gwdu111.gwdg.de/~compbiol/uniclust/2023_02/ 
 (~ 86G) for MSA searching

## Input file

To predict the GLN matrix of a homodimer, you need to prepare a fasta file containing two identical 
sequences with different sequence ids. Examples are given in the example/ directory.

## Inference

we provide a bash script and some sequence examples for model inference. Please 
change the uniclust30_database_path before running ./predict.sh
