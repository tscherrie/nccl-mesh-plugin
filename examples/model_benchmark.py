#!/usr/bin/env python3
"""
Model Inference Benchmark - tok/s across model sizes and node counts
"""
import argparse
import time
import torch
import os
import json

# Models to test (HuggingFace IDs - will download if needed)
MODELS = {
    "7B": "Qwen/Qwen2.5-7B-Instruct",
    "14B": "Qwen/Qwen2.5-14B-Instruct", 
    "32B": "Qwen/Qwen2.5-32B-Instruct",
    "72B": "Qwen/Qwen2.5-72B-Instruct",
}

RESULTS_FILE = os.path.expanduser("~/nccl-mesh-plugin/model_benchmark_results.json")

def benchmark_model(model_id, num_tokens=100, num_runs=3):
    """Load model and measure tok/s"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"  Loading {model_id}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        prompt = "Explain the theory of relativity in simple terms:"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Warmup
        print(f"  Warmup...", end=" ", flush=True)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=20, do_sample=False)
        torch.cuda.synchronize()
        print("done")
        
        # Benchmark
        print(f"  Benchmarking {num_tokens} tokens x {num_runs} runs...", end=" ", flush=True)
        times = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=num_tokens, do_sample=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        tok_per_sec = num_tokens / avg_time
        
        print(f"{tok_per_sec:.1f} tok/s")
        
        del model, tokenizer
        torch.cuda.empty_cache()
        
        return tok_per_sec
        
    except Exception as e:
        error_str = str(e).lower()
        if "out of memory" in error_str or "cuda" in error_str and "memory" in error_str:
            print(f"OOM!")
            torch.cuda.empty_cache()
            return "OOM"
        else:
            print(f"Error: {e}")
            torch.cuda.empty_cache()
            return "ERROR"

def run_single_node():
    print("=" * 70)
    print("SINGLE NODE (1x 128GB GPU)")
    print("=" * 70)
    
    results = {"nodes": 1, "models": {}}
    
    for size, model_id in MODELS.items():
        print(f"\n[{size} Model]")
        result = benchmark_model(model_id)
        results["models"][size] = result
    
    all_results = load_results()
    all_results["1_node"] = results
    save_results(all_results)
    
    print("\n" + "=" * 70)
    print_report()

def run_distributed(rank, world_size, master_ip):
    import torch.distributed as dist
    
    dist.init_process_group('nccl', rank=rank, world_size=world_size,
                           init_method=f'tcp://{master_ip}:29500')
    
    if rank == 0:
        print("=" * 70)
        print(f"DISTRIBUTED ({world_size}x 128GB GPUs = {world_size * 128}GB)")
        print("=" * 70)
    
    results = {"nodes": world_size, "models": {}}
    
    for size, model_id in MODELS.items():
        dist.barrier()
        
        if rank == 0:
            print(f"\n[{size} Model]")
            result = benchmark_model(model_id)
            results["models"][size] = result
        
        dist.barrier()
    
    if rank == 0:
        all_results = load_results()
        all_results[f"{world_size}_node"] = results
        save_results(all_results)
        print("\n" + "=" * 70)
        print_report()
    
    dist.destroy_process_group()

def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}

def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

def print_report():
    results = load_results()
    
    print("\n" + "=" * 70)
    print("MODEL INFERENCE BENCHMARK RESULTS")
    print("=" * 70)
    print(f"\n{'Model':<8} {'1 Node':<12} {'3 Nodes':<12} {'Notes':<20}")
    print("-" * 55)
    
    for size in ["7B", "14B", "32B", "72B"]:
        row = f"{size:<8}"
        
        v1 = results.get("1_node", {}).get("models", {}).get(size, "-")
        v3 = results.get("3_node", {}).get("models", {}).get(size, "-")
        
        if isinstance(v1, (int, float)):
            row += f"{v1:<12.1f}"
        else:
            row += f"{str(v1):<12}"
            
        if isinstance(v3, (int, float)):
            row += f"{v3:<12.1f}"
        else:
            row += f"{str(v3):<12}"
        
        # Notes
        if v1 == "OOM" and isinstance(v3, (int, float)):
            row += "← ONLY WORKS DISTRIBUTED"
        elif isinstance(v1, (int, float)) and isinstance(v3, (int, float)):
            speedup = v3 / v1
            row += f"{speedup:.2f}x"
        
        print(row)
    
    print("-" * 55)
    print("\n72B on 3 nodes = the whole point of clustering!")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['single', 'distributed', 'report'], required=True)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=3)
    parser.add_argument('--master-ip', type=str, default='10.0.0.170')
    args = parser.parse_args()
    
    if args.mode == 'single':
        run_single_node()
    elif args.mode == 'distributed':
        run_distributed(args.rank, args.world_size, args.master_ip)
    else:
        print_report()

if __name__ == '__main__':
    main()
