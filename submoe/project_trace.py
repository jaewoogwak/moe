"""Project arbitrary top-k routing CSVs onto a Sub-MoE mapping."""
from __future__ import annotations
import argparse, csv, json, os, re
from collections import Counter, defaultdict
from pathlib import Path
import torch

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--trace',type=Path,required=True); p.add_argument('--group-state',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    try: state=torch.load(a.group_state,map_location='cpu',weights_only=True)
    except TypeError: state=torch.load(a.group_state,map_location='cpu')
    with a.trace.open(newline='') as src:
        reader=csv.DictReader(src); fields=reader.fieldnames or []
        ranks=sorted((int(m.group(1)), f) for f in fields if (m:=re.fullmatch(r'top(\d+)_expert',f)))
        if not ranks: raise ValueError('trace has no topN_expert columns')
        extra=[f'top{rank}_group' for rank,_ in ranks]+['unique_group_count','group_collapse']
        a.output.parent.mkdir(parents=True,exist_ok=True); temp=a.output.with_suffix(a.output.suffix+'.tmp')
        counts=defaultdict(Counter); total=Counter()
        with temp.open('w',newline='') as dst:
            writer=csv.DictWriter(dst,fieldnames=fields+extra); writer.writeheader()
            for row in reader:
                layer=int(row['layer']); key=f'model.layers.{layer}.mlp'
                if key not in state: key=f'model.layers.{layer}.block_sparse_moe'
                labels=state[key].flatten().to(torch.long); groups=[int(labels[int(row[field])]) for _,field in ranks]
                unique=len(set(groups)); row.update({f'top{rank}_group':f'L{layer}:G{group}' for (rank,_),group in zip(ranks,groups)})
                row.update(unique_group_count=unique,group_collapse=str(unique < len(groups)).lower()); writer.writerow(row)
                counts[layer][unique]+=1; total[unique]+=1
    os.replace(temp,a.output); n=sum(total.values()); k=len(ranks)
    def metrics(counter):
        return {'rows':sum(counter.values()),'mean_unique_groups':sum(x*c for x,c in counter.items())/sum(counter.values()),'unique_group_count_distribution':dict(sorted(counter.items())), 'p_unique_equals_top_k':counter[k]/sum(counter.values()), 'p_unique_less_than_top_k':sum(c for x,c in counter.items() if x<k)/sum(counter.values()), 'mean_unique_groups_over_top_k':sum(x*c for x,c in counter.items())/sum(counter.values())/k}
    summary={'source_trace':str(a.trace),'group_state':str(a.group_state),'top_k':k,'overall':metrics(total),'per_layer':{str(l):metrics(c) for l,c in sorted(counts.items())}}
    a.output.with_suffix(a.output.suffix+'.summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
