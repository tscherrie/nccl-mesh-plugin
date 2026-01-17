#!/bin/bash
#
# Qwen2.5-14B optimized training launcher for 3-node Spark cluster
#
# Memory requirements (~50GB/node with plenty of headroom):
#   Model weights (BF16):     28GB total (~9GB/node)
#   8-bit optimizer states:   28GB total (~9GB/node)
#   Gradients (BF16):         28GB total (~9GB/node)
#   Activations (with ckpt): ~20GB/node
#   --------------------------------
#   Total per node:          ~50GB (vs 117GB available)
#

set -e

# Cluster configuration
HEAD_NODE="${HEAD_NODE:-spark-a}"
NODES="${NODES:-spark-a,spark-b,spark-c}"
NUM_NODES="${NUM_NODES:-3}"

# NCCL mesh plugin configuration - use actual .so file, not symlink
export NCCL_NET_PLUGIN=/home/titanic/nccl-mesh-plugin/libnccl-net.so
export LD_LIBRARY_PATH=/home/titanic/nccl-mesh-plugin:${LD_LIBRARY_PATH:-}
export NCCL_DEBUG=INFO
export NCCL_MESH_DEBUG=1
export NCCL_SOCKET_IFNAME=enP7s7  # Primary mesh interface

# Triton cache (avoid NFS issues mentioned in your logs)
export TRITON_CACHE_DIR=/tmp/triton_cache_$$

# Memory allocation settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_qwen14b_optimized.py"

# Parse arguments
DRY_RUN=""
MAX_STEPS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --steps)
            MAX_STEPS="--steps $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "QWEN-14B TRAINING LAUNCHER"
echo "=============================================="
echo "Head node: ${HEAD_NODE}"
echo "Nodes: ${NODES}"
echo "NCCL plugin: mesh"
echo "=============================================="

# Check if running under SLURM
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Running under SLURM (job $SLURM_JOB_ID)"

    srun --nodes=${NUM_NODES} \
         --ntasks-per-node=1 \
         --cpus-per-task=12 \
         --export=ALL \
         bash -c "echo \"[\$(hostname)] Checking mesh plugin...\" && \
                  ls -la /home/titanic/nccl-mesh-plugin/libnccl-net.so && \
                  echo \"[\$(hostname)] Verifying plugin symbols...\" && \
                  nm -D /home/titanic/nccl-mesh-plugin/libnccl-net.so 2>/dev/null | grep -E 'ncclNet|ncclCollNet' || echo 'WARNING: No symbols found!' && \
                  echo \"[\$(hostname)] Checking plugin dependencies...\" && \
                  ldd /home/titanic/nccl-mesh-plugin/libnccl-net.so | grep -v 'not found' | head -5 && \
                  export LD_LIBRARY_PATH=/home/titanic/nccl-mesh-plugin:\${LD_LIBRARY_PATH:-} && \
                  export NCCL_NET_PLUGIN=/home/titanic/nccl-mesh-plugin/libnccl-net.so && \
                  export NCCL_DEBUG=INFO && \
                  export NCCL_MESH_DEBUG=1 && \
                  export NCCL_SOCKET_IFNAME=enP7s7 && \
                  torchrun \
                     --nproc_per_node=1 \
                     --nnodes=${NUM_NODES} \
                     --rdzv_backend=c10d \
                     --rdzv_endpoint=${HEAD_NODE}:29500 \
                     ${TRAIN_SCRIPT} ${DRY_RUN} ${MAX_STEPS}"
else
    echo "Not running under SLURM - use sbatch or srun"
    echo ""
    echo "Example:"
    echo "  salloc -N3 --exclusive bash -c './run_qwen14b_training.sh'"
    echo ""
    echo "Or submit as job:"
    echo "  sbatch --nodes=3 --ntasks-per-node=1 ./run_qwen14b_training.sh"
    exit 1
fi
