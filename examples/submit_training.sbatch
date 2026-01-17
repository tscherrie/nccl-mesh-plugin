#!/bin/bash
#SBATCH --job-name=qwen14b-train
#SBATCH --nodes=3
#SBATCH --exclusive
#SBATCH --output=training_%j.log
#SBATCH --error=training_%j.log
#SBATCH --time=24:00:00

#
# Qwen2.5-14B Training with DeepSpeed ZeRO-3
#
# Usage:
#   sbatch submit_training.sbatch                    # Use defaults
#   sbatch submit_training.sbatch 12000              # Custom steps
#   sbatch submit_training.sbatch 12000 100 2e-5    # steps, warmup, lr
#
# Monitor:
#   tail -f training_<jobid>.log
#   squeue -u $USER
#
# Cancel:
#   scancel <jobid>
#

# Training parameters (can be overridden via positional args)
STEPS=${1:-12000}
WARMUP=${2:-100}
LR=${3:-2e-5}

echo "=========================================="
echo "Qwen2.5-14B Training Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Steps: $STEPS"
echo "Warmup: $WARMUP"
echo "Learning rate: $LR"
echo "Started: $(date)"
echo "=========================================="

cd /home/titanic/nccl-mesh-plugin

# Run the training
./examples/run_qwen14b_deepspeed.sh \
    --steps $STEPS \
    --warmup $WARMUP \
    --lr $LR

EXIT_CODE=$?

echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
