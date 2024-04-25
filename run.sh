#!/bin/bash

#SBATCH --job-name=0.1-S0     # Job name
#SBATCH --output=/l/users/mukhtar.mohamed/output.txt
#SBATCH --nodes=1                   # Run all processes on a single node   
#SBATCH --ntasks=1              # Run on a single CPU
#SBATCH --cpus-per-task=64           # Number of CPU cores (changed from 64 to 4)
#SBATCH --mem=40G   
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
#SBATCH -p gpu                      # Use the gpu partition
#SBATCH --time=8:00:00             # Specify the time needed for your experiment
#SBATCH --qos gpu-8                 # To enable the use of up to 8 GPUs

hostname
 
# python /l/users/rzan.alhaddad/DUB-dysarthria/src/run_translate_with_pseudo.sh --additional_data 0.1 --severity 0
python main.py --batch_size 16 --epochs 100 --EXP_name Binary_original --wandb_run_name Binary_original