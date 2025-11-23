#!/bin/bash
#SBATCH --partition=superpod
#SBATCH --nodes=13
#SBATCH --nodelist=sp-[0004-0016]
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:h100:8
#SBATCH --mem=1500000M
#SBATCH --time=28-00:00:00
#SBATCH --job-name=obsidian-stage-3-cns-finetune-full-sp
#SBATCH --output=/gpfs/data/oermannlab/users/alyaka01/logs/obsidian/stage-3-cns-finetune-full-sp/pyScript_%J_%N_%x.log

# loading relevant modules
module load cuda/12.1
source /cm/shared/apps/anaconda3/2023/etc/profile.d/conda.sh

# conda activate cns-llava-hf-11-scratch
conda activate cns-llava-hf-12-scratch

# Print config and status for debugging
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
module list
nvidia-smi
conda info
conda list

# start time
echo "starting at `date` on `hostname`"

###################
# Set the necessary environment variables for distributed training
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=31412  # You can use a different port if needed

export RANK=$SLURM_PROCID

# # run script
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export TORCH_CUDNN_V8_API_DISABLED=1
# export CUDA_LAUNCH_BLOCKING=1

srun python /gpfs/data/oermannlab/users/alyaka01/CNS-Obsidian/cns_obsidian/train/obsidian_stage_3_cns_finetune.py
###################

# end time
echo "ending at `date` on `hostname`"

