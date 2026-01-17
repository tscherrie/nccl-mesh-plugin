# NCCL Mesh Plugin

**Custom NCCL network plugin enabling distributed ML over direct-connect RDMA mesh topologies.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This plugin enables NCCL (NVIDIA Collective Communications Library) to work with **direct-connect mesh topologies** where each node pair is on a different subnet. Standard NCCL plugins assume either a switched InfiniBand fabric (all nodes on same subnet) or TCP/IP networking (slow, high latency). Neither works for direct-cabled RDMA meshes. This plugin does.

**Tested configuration**: 3x DGX Spark workstations with 100Gbps direct RDMA links, running distributed LLM training (Qwen2.5-14B) with DeepSpeed ZeRO-3.

## Quick Start

```bash
# 1. Build the plugin
git clone https://github.com/yourusername/nccl-mesh-plugin.git
cd nccl-mesh-plugin
make

# 2. Set environment variables
export NCCL_NET_PLUGIN=$(pwd)/libnccl-net.so
export NCCL_SOCKET_IFNAME=eth0  # Your management network interface

# 3. Run distributed training (with SLURM)
salloc -N3 --exclusive ./examples/run_qwen14b_deepspeed.sh --steps 100
```

See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.

## The Problem We Solved

```
                    +-----------+
                    |  Node A   |
                    | (spark-a) |
                    +-----+-----+
           192.168.101.x  |  192.168.100.x
              (100Gbps)   |     (100Gbps)
                    +-----+-----+
                    |           |
              +-----+-----+ +---+-------+
              |  Node B   | |  Node C   |
              | (spark-b) | | (spark-c) |
              +-----+-----+ +-----+-----+
                    |             |
                    +------+------+
                    192.168.102.x
                      (100Gbps)
```

**Three DGX Spark workstations** connected in a triangle mesh with direct 100Gbps RDMA cables. Each link is on a **different subnet** - a configuration not covered by standard NCCL network plugins.

## Performance

| Metric | Value |
|--------|-------|
| Effective Bandwidth | **8+ GB/s** |
| Line Rate Utilization | ~64% |
| Topology | 3-node triangle mesh |
| Link Speed | 100 Gbps per link |

## LLM Training Example

This plugin was developed for distributed LLM training. Here's how to train Qwen2.5-14B across 3 nodes:

### Prerequisites

- PyTorch 2.0+
- DeepSpeed
- Transformers
- FlashAttention 2 (optional, for memory efficiency)

### Running Training

**Interactive (for testing):**
```bash
salloc -N3 --exclusive
./examples/run_qwen14b_deepspeed.sh --steps 100
```

**Batch job (for full training):**
```bash
# Submit with default settings (12000 steps, ~1 epoch)
sbatch examples/submit_training.sbatch

# Or customize: steps, warmup_steps, learning_rate
sbatch examples/submit_training.sbatch 12000 100 2e-5

# Monitor
tail -f training_<jobid>.log
```

### Memory Usage

With DeepSpeed ZeRO-3 on 3 nodes (117GB unified memory each):

| Component | Per-Node Memory |
|-----------|-----------------|
| Sharded parameters | ~9 GB |
| Sharded optimizer states | ~19 GB |
| Sharded gradients | ~9 GB |
| **Total allocated** | **~38 GB** |
| Reserved (PyTorch cache) | ~42 GB |

Plenty of headroom for larger batch sizes or longer sequences.

### Scaling

| Model Size | Minimum Nodes | Recommended |
|------------|---------------|-------------|
| 7-14B | 2 | 3 |
| 32B | 3 | 4 |
| 70B | 6 | 8 |

## Installation

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install libibverbs-dev librdmacm-dev

# Verify RDMA devices
ibv_devices
```

### Build

```bash
git clone https://github.com/yourusername/nccl-mesh-plugin.git
cd nccl-mesh-plugin
make
```

### Environment Setup

```bash
# Required
export NCCL_NET_PLUGIN=/path/to/libnccl-net.so
export LD_LIBRARY_PATH=/path/to/nccl-mesh-plugin:$LD_LIBRARY_PATH
export NCCL_SOCKET_IFNAME=eth0  # Bootstrap interface (management network)

# Optional
export NCCL_DEBUG=INFO          # Or WARN for less output
export NCCL_MESH_GID_INDEX=3    # RoCE GID index (try 0-3 if issues)
```

## Architecture

### Key Innovations

1. **Multi-Address Handle Exchange**: Each node advertises ALL its subnet IPs in the NCCL handle
2. **Subnet-Aware NIC Selection**: `connect()` finds the local NIC on the same subnet as the peer
3. **Background Handshake Thread**: Eliminates deadlock when both ranks call `connect()` simultaneously
4. **Bidirectional QP Exchange**: Fresh Queue Pairs created for each connection

### Connection Flow

```
Rank 0 (listen)                    Rank 1 (connect)
     |                                   |
     v                                   |
 listen()                                |
 +- Create QPs on ALL NICs               |
 +- Start handshake thread               |
 +- Return handle with all IPs           |
     |                                   |
     |<-------- handle exchange -------->|
     |                                   |
     |                                   v
     |                              connect()
     |                              +- Find matching subnet
     |                              +- Create QP on that NIC
     |                              +- TCP handshake ---------->|
     |                                   |                      |
     |<------------------------------ QP info -----------------|
     |                                   |                      |
     v                                   v                      v
 accept()                           Connect QP            [handshake thread]
 +- Get QP from queue               to peer's QP          +- Accept TCP
 +- Return recv_comm                     |                +- Create new QP
                                         |                +- Connect QPs
                                         |                +- Queue for accept()
                                    +----+----+
                                    | RDMA OK |
                                    +---------+
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for implementation details.

## Configuration Reference

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `NCCL_NET_PLUGIN` | - | Path to `libnccl-net.so` |
| `NCCL_SOCKET_IFNAME` | - | Bootstrap network interface |
| `NCCL_DEBUG` | `WARN` | Log level: `INFO`, `WARN`, `TRACE` |
| `NCCL_MESH_GID_INDEX` | `3` | RoCE GID index to use |
| `NCCL_MESH_DEBUG` | `0` | Plugin debug: 0=off, 1=info, 2=verbose |
| `NCCL_MESH_TIMEOUT_MS` | `5000` | Connection timeout (ms) |
| `NCCL_MESH_RETRY_COUNT` | `3` | Connection retry attempts |

## Project Structure

```
nccl-mesh-plugin/
+-- src/mesh_plugin.c          # Main plugin implementation
+-- include/mesh_plugin.h      # Data structures
+-- nccl/                      # NCCL header files (net.h, net_v8.h)
+-- examples/
|   +-- train_qwen14b_deepspeed.py   # LLM training script
|   +-- run_qwen14b_deepspeed.sh     # Launcher script
|   +-- submit_training.sbatch       # SLURM batch submission
|   +-- test_allreduce.py            # Basic communication test
|   +-- benchmark_bandwidth.py       # Bandwidth benchmark
+-- docs/
|   +-- ARCHITECTURE.md        # Deep dive into implementation
|   +-- SETUP.md               # Hardware setup guide
+-- QUICKSTART.md              # Step-by-step getting started
+-- Makefile
```

## Troubleshooting

### Plugin not loading

```
NET/Plugin: Could not find: /path/to/libnccl-net.so
```

- Verify the path is correct on ALL nodes
- Check file permissions
- Rebuild with `make clean && make`

### Connection timeout with 192.168.x.x addresses

```
Call to ibv_modify_qp failed with 110 Connection timed out
local GID ::ffff:192.168.100.1, remote GID ::ffff:192.168.101.3
```

The mesh plugin didn't load, and NCCL fell back to the built-in IB plugin which uses unroutable addresses. Fix:
- Ensure `NCCL_NET_PLUGIN` points to the correct path
- Look for `Loaded net plugin Mesh (v9)` in logs

### GID index issues

Try different values: `export NCCL_MESH_GID_INDEX=0` (or 1, 2, 3)

Check available GIDs:
```bash
show_gids
# or
cat /sys/class/infiniband/*/ports/1/gids/*
```

## Limitations

- **Full mesh required**: Non-adjacent nodes can't communicate (no relay routing yet)
- **Single channel per port**: Uses 100Gbps, not full 200Gbps per ConnectX-7 port
- **RoCE v2 only**: No InfiniBand fabric support

## Roadmap

- [ ] Ring topology support (for 4+ nodes with only 2 NICs each)
- [ ] Dual-channel per port (200Gbps)
- [ ] Multi-QP aggregation
- [ ] Checkpoint saving in training scripts

## Documentation

- [QUICKSTART.md](QUICKSTART.md) - Get running in 15 minutes
- [docs/SETUP.md](docs/SETUP.md) - Hardware setup and network configuration
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Plugin internals and design

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

Built to connect DGX Spark workstations in ways that go beyond standard configurations.
