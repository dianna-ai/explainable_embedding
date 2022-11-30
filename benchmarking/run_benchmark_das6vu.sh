#!/usr/bin/env bash
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH -C A4000

source ~/.bashrc
module load cuda11.2/toolkit
mamba activate embeddings
cd ~/scratch/explainable_embedding/
python benchmarking/distance_benchmark.py

