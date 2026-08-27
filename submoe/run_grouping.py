#!/usr/bin/env python3
"""Build Sub-MoE expert-to-group mappings without merging any weights."""
from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from submoe.grouping import cluster_expert_outputs
from offload.expert_cache import FIXED_PER_LAYER_LRU, GPUExpertCache
from offload.host_expert_store import HostExpertStore
from offload.offloaded_experts import replace_with_cached_offloaded_experts

MODELS={'mixtral':'mistralai/Mixtral-8x7B-v0.1','qwen':'Qwen/Qwen1.5-MoE-A2.7B-Chat'}

def args():
 p=argparse.ArgumentParser(); p.add_argument('--model-type',choices=MODELS,required=True); p.add_argument('--model'); p.add_argument('--calib-dataset',choices=('c4','wikitext2'),default='c4'); p.add_argument('--calib-data-file',type=Path); p.add_argument('--num-blocks',type=int,default=32); p.add_argument('--block-size',type=int,default=2048); p.add_argument('--num-groups',type=int); p.add_argument('--seed',type=int,default=0); p.add_argument('--max-iter',type=int,default=100); p.add_argument('--chunk-size',type=int,default=2048); p.add_argument('--gpu-memory',default='14GiB'); p.add_argument('--output-dir',type=Path,required=True); return p.parse_args()
def blocks(tok,a):
 if a.calib_data_file: ds=load_dataset('json',data_files={'train':str(a.calib_data_file)},split='train').shuffle(seed=a.seed)
 elif a.calib_dataset=='c4': ds=load_dataset('allenai/c4','en',split='train',streaming=True)
 else: ds=load_dataset('wikitext','wikitext-2-raw-v1',split='train')
 buf=[]; out=[]
 for row in ds:
  text=row.get('text',''); buf.extend(tok(text,add_special_tokens=False)['input_ids'])
  while len(buf)>=a.block_size and len(out)<a.num_blocks: out.append(torch.tensor(buf[:a.block_size])); del buf[:a.block_size]
  if len(out)==a.num_blocks:return out
 raise ValueError('not enough calibration text')
def execution_device(module):
 hook=getattr(module,'_hf_hook',None); d=getattr(hook,'execution_device',None)
 if d is not None:return torch.device(d)
 return next(p for p in module.parameters() if p.device.type!='meta').device
def load_mixtral(model_id):
 model=AutoModelForCausalLM.from_pretrained(model_id,dtype=torch.bfloat16,device_map='cpu',low_cpu_mem_usage=True); model.eval()
 store=HostExpertStore(model)
 model.model.embed_tokens.to('cuda'); model.model.norm.to('cuda'); model.model.rotary_emb.to('cuda'); model.lm_head.to('cuda')
 for layer in model.model.layers:
  layer.self_attn.to('cuda'); layer.input_layernorm.to('cuda'); layer.post_attention_layernorm.to('cuda'); layer.mlp.gate.to('cuda')
 cache=GPUExpertCache(store,capacity_slots=len(model.model.layers),device='cuda',cache_policy=FIXED_PER_LAYER_LRU)
 replace_with_cached_offloaded_experts(model,cache); return model,store,cache
def main():
 a=args(); model_id=a.model or MODELS[a.model_type]; tok=AutoTokenizer.from_pretrained(model_id); default_groups=4 if a.model_type=='mixtral' else 30; groups=a.num_groups or default_groups
 store=cache=None
 if a.model_type=='mixtral': model,store,cache=load_mixtral(model_id)
 else: model=AutoModelForCausalLM.from_pretrained(model_id,dtype=torch.bfloat16,device_map='auto',max_memory={0:a.gpu_memory,'cpu':'900GiB'},low_cpu_mem_usage=True); model.eval()
 layers=model.model.layers; inputs=defaultdict(list)
 if a.model_type=='qwen': print(f'Qwen sanity: routed experts={model.config.num_experts}, shared experts={int(hasattr(layers[0].mlp,"shared_expert"))}, target groups={groups}')
 hs=[]
 for i,l in enumerate(layers): hs.append(l.mlp.register_forward_pre_hook(lambda _,x,i=i: inputs[i].append(x[0].detach().cpu().to(torch.bfloat16))))
 for ids in blocks(tok,a):
  dev=torch.device('cuda') if a.model_type=='mixtral' else execution_device(model.get_input_embeddings()); kw={'input_ids':ids[None].to(dev),'use_cache':False,'return_dict':True}
  with torch.inference_mode():
   if a.model_type=='mixtral': model(**kw,logits_to_keep=1)
   else: model.model(**kw)
 for h in hs:h.remove()
 state={}; meta={'model':model_id,'method':'submoe','calibration_dataset':a.calib_dataset,'num_blocks':a.num_blocks,'block_size':a.block_size,'num_groups':groups,'seed':a.seed,'max_iter':a.max_iter,'similarity':'mean_tokenwise_cosine','initialization':'kmeans++','convergence_iterations':{},'empty_cluster_events':{}}
 for i,l in enumerate(layers):
  x=torch.cat(inputs.pop(i),dim=0)
  x=x.reshape(-1,x.shape[-1])
  assert x.shape == (a.num_blocks*a.block_size,x.shape[-1]), f'Unexpected flattened calibration shape: {tuple(x.shape)}'
  experts=l.mlp.experts; outputs=[]; num_experts=model.config.num_local_experts if a.model_type=='mixtral' else len(experts)
  print(f'L{i:02d} Sub-MoE outputs CPU BF16: {num_experts * x.shape[0] * x.shape[1] * 2 / 1024**3:.2f} GiB')
  for e in range(num_experts):
   chunks=[]
   for s in range(0,len(x),a.chunk_size):
    with torch.inference_mode():
     if a.model_type=='mixtral':
      gate_up,down=cache.get(i,e); gate,up=F.linear(x[s:s+a.chunk_size].cuda(),gate_up).chunk(2,-1); y=F.linear(F.silu(gate)*up,down)
     else:
      expert=experts[e]; y=expert(x[s:s+a.chunk_size].to(execution_device(expert)))
     chunks.append(y.detach().cpu().to(torch.bfloat16))
   outputs.append(torch.cat(chunks)); del chunks
  expert_outputs=torch.stack(outputs)
  assert expert_outputs.shape == (num_experts,x.shape[0],x.shape[1]), f'Unexpected expert output shape: {tuple(expert_outputs.shape)}'
  result=cluster_expert_outputs(expert_outputs,groups,seed=a.seed,max_iter=a.max_iter,chunk_size=a.chunk_size); key=f'model.layers.{i}.mlp'; state[key]=result.labels; meta['convergence_iterations'][str(i)]=result.iterations; meta['empty_cluster_events'][str(i)]=result.empty_cluster_events
  members={g:torch.where(result.labels==g)[0].tolist() for g in range(groups)}; print(f'L{i:02d} '+ ' '.join(f'G{g}:{m}' for g,m in members.items())); del x,outputs; gc.collect(); torch.cuda.empty_cache()
 a.output_dir.mkdir(parents=True,exist_ok=True); torch.save(state,a.output_dir/'group_state_dict.pt'); (a.output_dir/'group_mapping_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':main()
