"""Phase launcher. Recipe provenance is documented beside the JSON; weights stay external."""
import argparse, importlib, inspect, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'common'))
PHASE=6
MODULES={2:('ner','NERConfig','train_ner'),3:('rr','RRConfig','train_rr'),4:('cjpe','CJPEConfig','train_cjpe'),
         5:('lsi','LSIConfig','train_lsi'),6:('summ','SummConfig','train_summ'),7:('pcr','PCRConfig','train_pcr')}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['train','predict','show-config'])
    p.add_argument('--config',default=str(Path(__file__).with_name('recipe.json')))
    p.add_argument('--phase1-run');p.add_argument('--phase1-ckpt');p.add_argument('--data-dir')
    p.add_argument('--out-dir');p.add_argument('--run-dir');p.add_argument('--device')
    a=p.parse_args()
    raw=json.loads(Path(a.config).read_text())
    for arg,key in [(a.phase1_run,'phase1_run'),(a.phase1_ckpt,'phase1_ckpt'),(a.data_dir,'data_dir'),(a.out_dir,'out_dir')]:
        if arg is not None:raw[key]=arg
    if a.action=='show-config':
        print(json.dumps(raw,indent=2));return
    if a.action=='train':
        if str(raw.get('data_dir','')).startswith('EXTERNAL_'):p.error('Supply --data-dir for official prepared data')
        if PHASE>1 and str(raw.get('phase1_run','')).startswith('EXTERNAL_'):p.error('Supply --phase1-run for your original checkpoint/tokenizer')
        if PHASE==1:
            from lm_lib.config import TrainConfig
            from lm_lib.trainer import train
            print(train(TrainConfig.from_dict(raw)))
        else:
            module,klass,train_name=MODULES[PHASE]
            api=importlib.import_module('lm_lib.'+module+'.trainer')
            print(getattr(api,train_name)(getattr(api,klass)(**raw)))
        return
    if not a.run_dir:p.error('predict requires --run-dir')
    import torch
    device=a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if PHASE==1:
        from lm_lib.trainer import load_trained
        from lm_lib.generate import generate_continuations
        from lm_lib.submit import write_submission
        if not a.data_dir:p.error('Phase-1 predict requires --data-dir containing test.json')
        model,tok,cfg,mtype=load_trained(a.run_dir,a.phase1_ckpt or 'best',device)
        test=json.loads((Path(a.data_dir)/'test.json').read_text())
        prompts={str(k):v['text'] for k,v in test.items()}
        pred=generate_continuations(model,tok,mtype,prompts,cfg.seq_len,cfg.src_len,n_words=128,max_new_tokens=256,
                                   batch_size=64,sampling=dict(temperature=0.0,repetition_penalty=1.1),device=device)
        print(write_submission(pred,test,a.out_dir or str(Path(a.run_dir)/'submission')))
    else:
        api=importlib.import_module('lm_lib.'+MODULES[PHASE][0]+'.trainer')
        signature=inspect.signature(api.predict_test);kwargs={}
        if 'device' in signature.parameters:kwargs['device']=device
        if a.data_dir:
            if 'data_dir' not in signature.parameters:p.error('This legacy predictor uses data_dir saved in the run config; update that external config when relocating data')
            kwargs['data_dir']=a.data_dir
        if a.out_dir and 'out_dir' in signature.parameters:kwargs['out_dir']=a.out_dir
        result=api.predict_test(a.run_dir,**kwargs)
        print(result[0] if isinstance(result,tuple) else result)
if __name__=='__main__':main()
