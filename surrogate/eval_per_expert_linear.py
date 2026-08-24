#!/usr/bin/env python3
"""Evaluate zero, training-output mean, and learned linear expert surrogates."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
H,E,EPS=4096,8,1e-12
def parse()->argparse.Namespace:
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--trace-dir",type=Path,default=Path("results/surrogate_per_expert/layer_16")); p.add_argument("--checkpoint-dir",type=Path); p.add_argument("--output-dir",type=Path); p.add_argument("--batch-size",type=int,default=256); p.add_argument("--experts",type=int,nargs="+",default=list(range(E))); return p.parse_args()
def trace(path:Path)->dict[str,object]:
 d=torch.load(path,map_location="cpu",weights_only=True); x,y=d["x"],d["y"]
 if not isinstance(x,torch.Tensor) or not isinstance(y,torch.Tensor) or x.shape != y.shape or x.ndim != 2 or x.shape[1] != H or not x.shape[0]: raise ValueError(f"invalid trace {path}")
 return d
def stats(xs:list[torch.Tensor])->dict[str,float]:
 x=torch.cat(xs).float(); return {"mean":x.mean().item(),"median":x.median().item()}
@torch.no_grad()
def measure(x:torch.Tensor,y:torch.Tensor,batch:int,predict:Callable[[torch.Tensor],torch.Tensor])->dict[str,dict[str,float]]:
 ms=[]; rs=[]; cs=[]
 for start in range(0,x.shape[0],batch):
  bx=x[start:start+batch].to("cuda",torch.float32); by=y[start:start+batch].to("cuda",torch.float32); p=predict(bx).float(); delta=p-by
  ms.append(delta.square().mean(1).cpu()); rs.append((torch.linalg.vector_norm(delta,dim=1)/torch.linalg.vector_norm(by,dim=1).clamp_min(EPS)).cpu()); cs.append(F.cosine_similarity(p,by,dim=1,eps=EPS).cpu())
 return {"mse":stats(ms),"relative_l2":stats(rs),"cosine_similarity":stats(cs)}
def surrogate(path:Path)->nn.Linear:
 d=torch.load(path,map_location="cpu",weights_only=True)
 if d.get("hidden_size") != H: raise ValueError(f"incompatible checkpoint {path}")
 m=nn.Linear(H,H,bias=True,device="cuda",dtype=torch.float32); m.load_state_dict(d["state_dict"]); return m.eval()
def sizes()->dict[str,float|int]:
 exact=28672*4096+4096*14336; linear=H*H+H; return {"exact_expert_parameters":exact,"linear_surrogate_parameters":linear,"exact_expert_bf16_bytes":exact*2,"linear_surrogate_bf16_bytes":linear*2,"compression_ratio":exact/linear}
def main()->None:
 a=parse()
 if not torch.cuda.is_available() or a.batch_size<1 or sorted(set(a.experts))!=sorted(a.experts) or any(x not in range(E) for x in a.experts): raise ValueError("invalid evaluation arguments")
 ck=a.checkpoint_dir or a.trace_dir/"checkpoints"; out=a.output_dir or a.trace_dir/"evaluation"; out.mkdir(parents=True,exist_ok=True); records=[]; rows=[]
 for e in a.experts:
  tr,te=trace(a.trace_dir/"train"/f"expert_{e}.pt"),trace(a.trace_dir/"test"/f"expert_{e}.pt"); ty,tx,ey=tr["y"],te["x"],te["y"]; assert isinstance(ty,torch.Tensor) and isinstance(tx,torch.Tensor) and isinstance(ey,torch.Tensor); mean=ty.float().mean(0,keepdim=True).to("cuda")
  zero=measure(tx,ey,a.batch_size,lambda b:torch.zeros_like(b)); average=measure(tx,ey,a.batch_size,lambda b:mean.expand(b.shape[0],-1)); m=surrogate(ck/f"expert_{e}.pt"); linear=measure(tx,ey,a.batch_size,m); del m,mean
  for base,values in (("zero",zero),("mean_output",average),("learned_linear",linear)):
   for metric,value in values.items(): records.append({"expert":e,"baseline":base,"metric":metric,**value})
  rows.append({"expert":e,"train_n":ty.shape[0],"test_n":ey.shape[0],"zero_rel_l2":zero["relative_l2"]["mean"],"mean_rel_l2":average["relative_l2"]["mean"],"linear_rel_l2":linear["relative_l2"]["mean"],"linear_cos":linear["cosine_similarity"]["mean"]})
 with (out/"evaluation_summary.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=("expert","baseline","metric","mean","median")); w.writeheader(); w.writerows(records)
 report={"trace_dir":str(a.trace_dir),"checkpoint_dir":str(ck),"experts":rows,"metrics":records,"size":sizes()}; (out/"evaluation_summary.json").write_text(json.dumps(report,indent=2)+"\n")
 print("Expert | Train N | Test N | Zero RelL2 | Mean RelL2 | Linear RelL2 | Linear Cos")
 for r in rows: print(f"E{r['expert']}     | {r['train_n']:>7,} | {r['test_n']:>6,} | {r['zero_rel_l2']:.6f} | {r['mean_rel_l2']:.6f} | {r['linear_rel_l2']:.6f} | {r['linear_cos']:.6f}")
 if rows:
  mean={k:sum(r[k] for r in rows)/len(rows) for k in ("zero_rel_l2","mean_rel_l2","linear_rel_l2","linear_cos")}; print(f"Mean   |         |        | {mean['zero_rel_l2']:.6f} | {mean['mean_rel_l2']:.6f} | {mean['linear_rel_l2']:.6f} | {mean['linear_cos']:.6f}")
 s=report["size"]; print(f"Exact expert size: {s['exact_expert_parameters']:,} parameters, {s['exact_expert_bf16_bytes']/1024**2:.2f} MiB BF16"); print(f"Linear surrogate size: {s['linear_surrogate_parameters']:,} parameters, {s['linear_surrogate_bf16_bytes']/1024**2:.2f} MiB BF16"); print(f"Compression ratio: {s['compression_ratio']:.2f}x"); print(f"Saved evaluation to: {out}")
if __name__ == "__main__": main()
