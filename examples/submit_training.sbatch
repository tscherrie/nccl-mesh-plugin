#!/bin/bash
#SBATCH --job-name=qwen14b-train
#SBATCH --nodes=3
#SBATCH --exclusive
#SBATCH --output=training_%j.log
#SBATCH --error=training_%j.log
#SBATCH --time=24:00:00

# Training parameters (adjust as needed)
STEPS=${1:-12000}
WARMUP=${2:-100}
LR=${3:-2e-5}

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Steps: $STEPS"
echo "Started: $(date)"
echo "=========================================="

cd /home/titanic/nccl-mesh-plugin

# Run the training
./examples/run_qwen14b_deepspeed.sh --steps $STEPS --warmup $WARMUP --lr $LR

echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
