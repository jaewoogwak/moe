#!/usr/bin/env python3
"""Train the eight independent full-rank FP32 linear surrogates sequentially."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import torch
import torch.nn as nn

H, E = 4096, 8
def parse() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--trace-dir",type=Path,default=Path("results/surrogate_per_expert/layer_16")); p.add_argument("--output-dir",type=Path); p.add_argument("--batch-size",type=int,default=256); p.add_argument("--learning-rate",type=float,default=1e-3); p.add_argument("--epochs",type=int,default=5); p.add_argument("--seed",type=int,default=42); p.add_argument("--experts",type=int,nargs="+",default=list(range(E))); p.add_argument("--mixed-precision",action="store_true"); return p.parse_args()
def trace(path: Path) -> dict[str, object]:
    d=torch.load(path,map_location="cpu",weights_only=True); x,y=d["x"],d["y"]
    if not isinstance(x,torch.Tensor) or not isinstance(y,torch.Tensor) or x.shape != y.shape or x.ndim != 2 or x.shape[1] != H or not x.shape[0]: raise ValueError(f"invalid trace {path}")
    return d
@torch.no_grad()
def loss(model: nn.Module,d: dict[str,object],batch:int,amp:bool)->float:
    x,y=d["x"],d["y"]; assert isinstance(x,torch.Tensor) and isinstance(y,torch.Tensor); total=0.; elements=0
    for start in range(0,x.shape[0],batch):
        bx=x[start:start+batch].to("cuda",torch.float32); by=y[start:start+batch].to("cuda",torch.float32)
        with torch.autocast("cuda",torch.bfloat16,enabled=amp): pred=model(bx)
        total += (pred.float()-by).square().sum().item(); elements += by.numel()
    return total/elements
def run(a:argparse.Namespace,expert:int,out:Path)->dict[str,object]:
    tr,va=trace(a.trace_dir/"train"/f"expert_{expert}.pt"),trace(a.trace_dir/"val"/f"expert_{expert}.pt"); x,y=tr["x"],tr["y"]; assert isinstance(x,torch.Tensor) and isinstance(y,torch.Tensor)
    model=nn.Linear(H,H,bias=True,device="cuda",dtype=torch.float32); opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=0); scaler=torch.amp.GradScaler("cuda",enabled=a.mixed_precision); gen=torch.Generator().manual_seed(a.seed+expert); best=float("inf"); history=[]
    for epoch in range(1,a.epochs+1):
        model.train(); order=torch.randperm(x.shape[0],generator=gen)
        for start in range(0,order.numel(),a.batch_size):
            ix=order[start:start+a.batch_size]; bx=x[ix].to("cuda",torch.float32); by=y[ix].to("cuda",torch.float32); opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",torch.bfloat16,enabled=a.mixed_precision): l=(model(bx)-by).square().mean()
            scaler.scale(l).backward(); scaler.step(opt); scaler.update()
        model.eval(); train_loss=loss(model,tr,a.batch_size,a.mixed_precision); val_loss=loss(model,va,a.batch_size,a.mixed_precision); history.append({"epoch":epoch,"train_mse":train_loss,"val_mse":val_loss}); print(f"E{expert} epoch {epoch}/{a.epochs}: train_mse={train_loss:.6e} val_mse={val_loss:.6e}")
        if val_loss < best:
            best=val_loss; torch.save({"expert_id":expert,"hidden_size":H,"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"best_epoch":epoch,"best_val_mse":best,"train_samples":x.shape[0],"val_samples":va["x"].shape[0],"optimization_dtype":"bf16_autocast" if a.mixed_precision else "fp32"},out/f"expert_{expert}.pt")
    return {"expert_id":expert,"best_val_mse":best,"history":history}
def main()->None:
    a=parse()
    if not torch.cuda.is_available() or min(a.batch_size,a.epochs)<1 or a.learning_rate<=0 or sorted(set(a.experts))!=sorted(a.experts) or any(x not in range(E) for x in a.experts): raise ValueError("invalid CUDA/training arguments")
    if not (a.trace_dir/"metadata.json").is_file(): raise FileNotFoundError(a.trace_dir/"metadata.json")
    random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    out=a.output_dir or a.trace_dir/"checkpoints"; out.mkdir(parents=True,exist_ok=True); summary={"trace_dir":str(a.trace_dir),"batch_size":a.batch_size,"learning_rate":a.learning_rate,"epochs":a.epochs,"seed":a.seed,"mixed_precision":a.mixed_precision,"experts":[]}
    for e in a.experts: print(f"Training independent linear surrogate for E{e}"); summary["experts"].append(run(a,e,out))
    (out/"training_summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(f"Saved best checkpoints to: {out}")
if __name__ == "__main__": main()
