#!/bin/bash

#SBATCH --mail-user=nir.yarden@mail.huji.ac.il
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --gres=gpu:1,vmem:40g
#SBATCH --mem=24g
#SBATCH --time=2-0
#SBATCH -c4
#SBATCH --output=bash_outputs/process_dataset_output.txt
#SBATCH --job-name=anlp_process_dataset

module load cuda
module load nvidia
nvidia-smi

# LOAD YOUR PYTHON ENV:
source .venv/bin/activate

# RUN YOUR PYTHON CODE:
python3 /cs/labs/roys/nir.yarden/anlp-project/NBA-Recaps/process_dataset.py