import inspect
import torch
import transformers

from transformers import AutoModelForCausalLM


MODEL = "/workspace/models/Mixtral-8x7B-v0.1"


print("Transformers:", transformers.__version__)
print("PyTorch:", torch.__version__)


print("\nLoading model on CPU...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)


layer = model.model.layers[0]

print("\n==============================")
print("Decoder Layer")
print("==============================")
print(type(layer))


print("\n==============================")
print("Attention")
print("==============================")
print(type(layer.self_attn))
print(inspect.signature(layer.self_attn.forward))


print("\n==============================")
print("MoE Block")
print("==============================")
print(type(layer.mlp))
print(inspect.signature(layer.mlp.forward))


print("\n==============================")
print("Router")
print("==============================")
print(type(layer.mlp.gate))
print(inspect.signature(layer.mlp.gate.forward))


print("\n==============================")
print("Experts")
print("==============================")
print(type(layer.mlp.experts))
print(inspect.signature(layer.mlp.experts.forward))


print("\n==============================")
print("Expert Parameters")
print("==============================")

for name, param in layer.mlp.experts.named_parameters():
    print(
        name,
        tuple(param.shape),
        param.dtype,
        param.device,
        f"{param.numel() * param.element_size() / 1024**2:.2f} MiB",
    )


print("\n==============================")
print("MoE forward source")
print("==============================")
print(inspect.getsource(layer.mlp.forward))


print("\n==============================")
print("Experts forward source")
print("==============================")
print(inspect.getsource(layer.mlp.experts.forward))