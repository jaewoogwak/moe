import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "/workspace/models/Mixtral-8x7B-v0.1"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={
        0: "20GiB",
        "cpu": "200GiB",
    },
    low_cpu_mem_usage=True,
)

print("Loaded.")

print(model.hf_device_map)

print(
    "GPU allocated:",
    torch.cuda.memory_allocated() / 1024**3,
    "GiB",
)
