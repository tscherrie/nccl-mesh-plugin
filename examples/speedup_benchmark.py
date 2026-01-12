#!/usr/bin/env python3
"""
Speedup Benchmark with comparison report
Run with different node counts, then use --report to see comparison
"""
import argparse
import time
import json
import os
import torch
import torch.nn as nn

RESULTS_DIR = os.path.expanduser("~/nccl-mesh-plugin/benchmark_results")

def benchmark_matmul(size, iterations=100, warmup=10):
    A = torch.randn(size, size, device='cuda', dtype=torch.float16)
    B = torch.randn(size, size, device='cuda', dtype=torch.float16)
    for _ in range(warmup):
        torch.matmul(A, B)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        torch.matmul(A, B)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tflops = (2 * (size ** 3) * iterations / elapsed) / 1e12
    return tflops

def benchmark_allreduce(size_mb, iterations=50, warmup=10):
    import torch.distributed as dist
    tensor = torch.randn((size_mb * 1024 * 1024) // 2, device='cuda', dtype=torch.float16)
    for _ in range(warmup):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (size_mb * iterations / 1024) / elapsed

def benchmark_training_sim(size, iterations=50, warmup=10):
    import torch.distributed as dist
    world_size = dist.get_world_size()
    model = nn.Linear(size, size, dtype=torch.float16, device='cuda')
    x = torch.randn(size // world_size, size, device='cuda', dtype=torch.float16)
    
    def step():
        y = model(x)
        y.sum().backward()
        for p in model.parameters():
            dist.all_reduce(p.grad)
            p.grad /= world_size
        model.zero_grad()
    
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (iterations * size) / elapsed

def save_results(results, num_nodes):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, f"results_{num_nodes}node.json")
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filepath}")

def load_results(num_nodes):
    filepath = os.path.join(RESULTS_DIR, f"results_{num_nodes}node.json")
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def print_report():
    print("\n")
    print("=" * 80)
    print("NCCL MESH PLUGIN - SPEEDUP BENCHMARK REPORT")
    print("=" * 80)
    
    # Load all results
    results = {}
    for n in [1, 2, 3]:
        r = load_results(n)
        if r:
            results[n] = r
    
    if not results:
        print("No results found. Run benchmarks first:")
        print("  Single node:  python3 speedup_benchmark.py --mode single")
        print("  2 nodes:      python3 speedup_benchmark.py --mode multi --world-size 2 ...")
        print("  3 nodes:      python3 speedup_benchmark.py --mode multi --world-size 3 ...")
        return
    
    # All-Reduce Bandwidth Table
    print("\n" + "-" * 80)
    print("ALL-REDUCE BANDWIDTH (GB/s)")
    print("-" * 80)
    header = f"{'Size':<15}"
    for n in sorted(results.keys()):
        if n > 1:
            header += f"{'%d Nodes' % n:<15}"
    print(header)
    print("-" * 80)
    
    for size_mb in [64, 128, 256]:
        row = f"{size_mb} MB{'':<10}"
        for n in sorted(results.keys()):
            if n > 1 and f'allreduce_{size_mb}' in results[n]:
                row += f"{results[n][f'allreduce_{size_mb}']:<15.2f}"
        print(row)
    
    # Training Throughput Table
    print("\n" + "-" * 80)
    print("TRAINING SIMULATION THROUGHPUT (samples/sec)")
    print("-" * 80)
    header = f"{'Size':<15}"
    for n in sorted(results.keys()):
        header += f"{'%d Node%s' % (n, 's' if n > 1 else ''):<15}"
    if 1 in results:
        header += f"{'2N Speedup':<12}{'3N Speedup':<12}"
    print(header)
    print("-" * 80)
    
    for size in [4096, 8192, 16384]:
        row = f"{size:<15}"
        baseline = None
        for n in sorted(results.keys()):
            key = f'training_{size}' if n > 1 else f'matmul_{size}'
            if key in results[n]:
                val = results[n][key]
                if n == 1:
                    # For single node, show TFLOPS, but we need samples/sec for comparison
                    # Let's use a normalized throughput instead
                    row += f"{val:<15.1f}"
                    baseline = val
                else:
                    row += f"{val:<15.0f}"
        
        # Calculate speedups if we have baseline
        if 1 in results and f'training_{size}' in results.get(1, {}):
            baseline = results[1][f'training_{size}']
            if 2 in results and f'training_{size}' in results[2]:
                speedup2 = results[2][f'training_{size}'] / baseline
                row += f"{speedup2:<12.2f}x"
            if 3 in results and f'training_{size}' in results[3]:
                speedup3 = results[3][f'training_{size}'] / baseline
                row += f"{speedup3:<12.2f}x"
        print(row)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if 1 in results and 2 in results:
        baseline = results[1].get('training_8192', 0)
        two_node = results[2].get('training_8192', 0)
        if baseline and two_node:
            print(f"2-Node Speedup (8192): {two_node/baseline:.2f}x")
    
    if 1 in results and 3 in results:
        baseline = results[1].get('training_8192', 0)
        three_node = results[3].get('training_8192', 0)
        if baseline and three_node:
            print(f"3-Node Speedup (8192): {three_node/baseline:.2f}x")
    
    print("=" * 80)
    print()

def run_single():
    results = {}
    print("Running single-node baseline benchmarks...")
    
    for size in [4096, 8192, 16384]:
        print(f"  MatMul {size}...", end=" ", flush=True)
        results[f'matmul_{size}'] = benchmark_matmul(size)
        print(f"{results[f'matmul_{size}']:.1f} TFLOPS")
    
    # Also do a "training sim" equivalent for fair comparison
    # Single node "training" is just matmul without communication
    for size in [4096, 8192, 16384]:
        print(f"  Training sim {size}...", end=" ", flush=True)
        # Simulate: forward + backward, 50 iterations
        model = nn.Linear(size, size, dtype=torch.float16, device='cuda')
        x = torch.randn(size, size, device='cuda', dtype=torch.float16)
        
        for _ in range(10):  # warmup
            y = model(x); y.sum().backward(); model.zero_grad()
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(50):
            y = model(x); y.sum().backward(); model.zero_grad()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        throughput = (50 * size) / elapsed
        results[f'training_{size}'] = throughput
        print(f"{throughput:.0f} samples/sec")
    
    save_results(results, 1)

def run_multi(rank, world_size, master_ip):
    import torch.distributed as dist
    dist.init_process_group('nccl', rank=rank, world_size=world_size,
                           init_method=f'tcp://{master_ip}:29500')
    
    results = {}
    
    if rank == 0:
        print(f"Running {world_size}-node distributed benchmarks...")
    
    for size_mb in [64, 128, 256]:
        if rank == 0:
            print(f"  AllReduce {size_mb} MB...", end=" ", flush=True)
        bw = benchmark_allreduce(size_mb)
        results[f'allreduce_{size_mb}'] = bw
        if rank == 0:
            print(f"{bw:.2f} GB/s")
    
    dist.barrier()
    
    for size in [4096, 8192, 16384]:
        if rank == 0:
            print(f"  Training sim {size}...", end=" ", flush=True)
        tp = benchmark_training_sim(size)
        results[f'training_{size}'] = tp
        if rank == 0:
            print(f"{tp:.0f} samples/sec")
    
    dist.barrier()
    
    if rank == 0:
        save_results(results, world_size)
    
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['single', 'multi', 'report'], required=True)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=2)
    parser.add_argument('--master-ip', type=str, default='10.0.0.170')
    args = parser.parse_args()
    
    if args.mode == 'single':
        run_single()
        print("\nNow run 2-node and 3-node tests, then use --mode report")
    elif args.mode == 'multi':
        run_multi(args.rank, args.world_size, args.master_ip)
        if args.rank == 0:
            print("\nRun --mode report to see comparison")
    else:
        print_report()

if __name__ == '__main__':
    main()
