#!/bin/bash
#SBATCH --job-name=qwen14b-train
#SBATCH --nodes=3
#SBATCH --exclusive
#SBATCH --output=training_%j.log
#SBATCH --error=training_%j.log
#SBATCH --time=72:00:00

#
# Qwen2.5-14B Training with DeepSpeed ZeRO-3
#
# Usage:
#   sbatch submit_training.sbatch                              # Fresh start with defaults
#   sbatch submit_training.sbatch 12000                        # Custom steps
#   sbatch submit_training.sbatch 12000 500 latest             # Resume from latest checkpoint
#
# Arguments: STEPS SAVE_STEPS RESUME
#   STEPS      - Total training steps (default: 12000)
#   SAVE_STEPS - Checkpoint every N steps (default: 500)
#   RESUME     - 'latest' to auto-resume, or path to checkpoint
#
# Monitor:
#   tail -f training_<jobid>.log
#   squeue -u $USER
#
# Cancel:
#   scancel <jobid>
#

# Training parameters
STEPS=${1:-12000}
SAVE_STEPS=${2:-500}
RESUME=${3:-}

echo "=========================================="
echo "Qwen2.5-14B Training Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Steps: $STEPS"
echo "Save every: $SAVE_STEPS steps"
if [ -n "$RESUME" ]; then
    echo "Resume: $RESUME"
fi
echo "Started: $(date)"
echo "=========================================="

cd /home/titanic/nccl-mesh-plugin

# Build resume argument
RESUME_ARG=""
if [ -n "$RESUME" ]; then
    RESUME_ARG="--resume $RESUME"
fi

# Run the training
./examples/run_qwen14b_deepspeed.sh \
    --steps $STEPS \
    --save-steps $SAVE_STEPS \
    $RESUME_ARG

EXIT_CODE=$?

echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
