#!/bin/bash

#SBATCH --partition main
#SBATCH --time=7-00:00:00      # time (D-H:MM:SS)
#SBATCH --job-name Pi_From_Scratch
#SBATCH --output=/home/amirbial/Computational_Learning/Scratch/job-%J.out
#SBATCH --mail-user=amirbial@post.bgu.ac.il
#SBATCH --mail-type=NONE
#SBATCH --gres=gpu:1              # Number of GPUs (per node)

##SBATCH --mem=8G
##SBATCH --cpus-per-task=1


### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

echo "/home/amirbial/Computational_Learning/Scratch/train.py"
module load anaconda
source activate Pi_Scratch
CUDA_LAUNCH_BLOCKING=1 python "/home/amirbial/Computational_Learning/Scratch/train.py"
###python "/home/amirbial/Computational_Learning/Scratch/train.py"