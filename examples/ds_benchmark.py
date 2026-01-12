import torch
import time
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-32B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

ds_model = deepspeed.init_inference(
    model,
    tensor_parallel={"tp_size": 3},
    dtype=torch.bfloat16,
)

prompt = "Explain relativity:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Warmup
ds_model.generate(**inputs, max_new_tokens=20)

# Benchmark
start = time.perf_counter()
out = ds_model.generate(**inputs, max_new_tokens=100)
elapsed = time.perf_counter() - start
print(f"32B on 3 nodes: {100/elapsed:.1f} tok/s")
