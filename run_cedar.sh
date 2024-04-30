#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=15:0:0    
#SBATCH --mail-user=<youremail@gmail.com>
#SBATCH --mail-type=ALL

cd ~/projects/def-mageed/mukh/DPO
module purge
module load python/3.7.9 scipy-stack
source ~/py37/bin/activate

python main.py