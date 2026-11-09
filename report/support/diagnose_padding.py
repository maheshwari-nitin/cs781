"""Tiny random-model implementation check, not a competition performance experiment."""
import json, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/common' if (ROOT/'code/common/lm_lib').exists() else ROOT))
from lm_lib.utils import load_arch, build
from lm_lib.ner.model import hidden_states

def diagnose():
    torch.set_num_threads(1)
    results=[]
    for arch in ["A_qwen_dense", "B_encdec"]:
        torch.manual_seed(1337)
        extra=dict(d_model=32,n_heads=4,n_kv_heads=2,head_dim=8,ffn_dim=64)
        extra.update(dict(n_layers=2) if arch.startswith("A") else dict(n_enc_layers=2,n_dec_layers=2))
        model=build(load_arch(arch),32,16,extra).eval()
        original=torch.tensor([[3,4,5,6]])
        with torch.no_grad():
            before=hidden_states(model,original)
            for padding in [4,8]:
                padded=torch.cat([original,torch.ones((1,padding),dtype=torch.long)],dim=1)
                after=hidden_states(model,padded)[:,:4]
                results.append(dict(architecture=arch,padding_tokens=padding,
                    relative_l2=float((after-before).norm()/before.norm()),
                    max_absolute_change=float((after-before).abs().max())))
    return dict(seed=1337,vocab_size=32,width=32,layers=2,device="cpu",untrained=True,results=results)
if __name__=="__main__":
    result=diagnose()
    target=Path(__file__).resolve().parents[1]/"evidence"/"padding_diagnostic.json"
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
