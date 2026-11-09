"""Build code/phase1..phase7 and a clean shared source tree; never copy data, credentials or weights."""
import ast
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'submission_work/package/code'
COMMON=DEST/'common'
MODULES={2:('ner','NERConfig','train_ner'),3:('rr','RRConfig','train_rr'),
         4:('cjpe','CJPEConfig','train_cjpe'),5:('lsi','LSIConfig','train_lsi'),
         6:('summ','SummConfig','train_summ'),7:('pcr','PCRConfig','train_pcr')}
WINNERS={1:(965311,'qwen_dense_best.zip',.8273),2:(977453,'A_ner.zip',.3703),
         3:(992390,'A_rr_2.zip',.5146),4:(1006421,'A_cj.zip',.7935),
         5:(1016369,'A_lsi.zip',.3517),6:(1028885,'A6.zip',.8153),7:(1038509,'A_pcr.zip',.1855)}
TASKS={1:'Legal LM pretraining',2:'Legal named-entity recognition',3:'Rhetorical-role prediction',
       4:'Court judgment prediction',5:'Legal statute identification',6:'Abstractive summarization',7:'Prior-case retrieval'}
NOTEBOOKS={1:['00_data_tokenizer','00b_full_data','01_train','02_generate_submit','03_resume','04_decoding_sweep'],
           2:['10_ner'],3:['20_rr'],4:['30_cjpe'],5:['40_lsi'],6:['50_summ'],7:['60_pcr','61_pcr_a_tuning']}


def dump(obj,path):path.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')


def recipes():
    common=dict(phase1_run='EXTERNAL_PHASE1_RUN_REQUIRED',phase1_ckpt='best',method='full',method_kwargs={},
                data_dir='EXTERNAL_DATA_DIR_REQUIRED',out_dir='runs',run_name='',seed=1337)
    result={}
    result[1]=dict(arch='A_qwen_dense',run_name='A_qwen_dense',data_dir='EXTERNAL_DATA_DIR_REQUIRED',out_dir='runs',
                   train_bin='train.bin',seq_len=1024,src_len=768,global_batch_seqs=256,micro_batch=0,mem_frac=.9,
                   train_tokens=0,lr=3e-4,warmup_frac=.02,decay_frac=.2,min_lr_ratio=.1,weight_decay=.1,
                   compile=True,eval_every=200,ckpt_minutes=15,extra={},seed=1337)
    result[2]=dict(common,run_name='A_ner_reconstructed',window=1024,stride=768,n_val_docs=10,batch_size=8,
                   epochs=10,lr=5e-5,head_lr=1e-3,weight_decay=.01,warmup_frac=.1,dropout=.1,
                   class_weighting='none',decode='argmax',window_agg='owner')
    result[3]=dict(common,run_name='A_rr_reconstructed',domains=['CL','IT'],context='document',pooling='mean',
                   window=1024,stride=768,max_sent_tokens=256,batch_size=8,grad_accum=1,epochs=10,
                   lr=5e-5,head_lr=1e-3,weight_decay=.01,warmup_frac=.1,dropout=.1,class_weighting='none',
                   extra_train=[],augment_docs=0,oversample_rare=0,train_on_dev=False)
    result[4]=dict(common,run_name='A_qwen_dense_full__full__last4x512_reconstructed',strategy='last',
                   chunk_len=512,max_chunks=4,token_pooling='mean',chunk_agg='mean',normalize_text=True,
                   drop_outcome=False,n_train=0,batch_size=8,grad_accum=1,epochs=3,lr=5e-5,head_lr=1e-3,
                   weight_decay=.01,warmup_frac=.1,dropout=.1,class_weighting='balanced',tune_threshold=False)
    result[5]=dict(common,run_name='A_full__spread4x512_reconstructed',phase1_ckpt='latest',strategy='spread',
                   chunk_len=512,max_chunks=4,token_pooling='mean',chunk_agg='mean',normalize_text=True,
                   n_train=0,batch_size=8,grad_accum=4,epochs=1,lr=5e-5,head_lr=1e-3,weight_decay=.01,
                   warmup_frac=.1,dropout=.1,pos_weight='none',threshold=.5,threshold_mode='none')
    result[6]=dict(common,run_name='A_lora__firstlast640_256_reconstructed',phase1_ckpt='latest',method='lora',
                   method_kwargs={'r':8,'lora_alpha':16},val_fraction=.05,split_seed=1337,n_val_generate=32,
                   source_tokens=640,target_tokens=256,strategy='first_last',normalize_text=True,
                   grad_accum=8,epochs=1,lr=5e-5,weight_decay=.01,warmup_frac=.1,
                   selection_metric='bertscore_f1',bert_model='roberta-large',bert_device='cpu',
                   max_new_tokens=256,min_new_tokens=16,temperature=0.0,repetition_penalty=1.0,no_repeat_ngram_size=0)
    result[7]=dict(common,run_name='A_full__firstlast2x256_reconstructed',chunk_len=256,max_chunks=2,
                   strategy='first_last',token_pooling='mean',chunk_agg='mean',normalize_text=True,
                   negatives=4,max_positives=2,temperature=.05,n_train=0,n_dev=0,embed_batch_size=1024,
                   search_batch_size=64,train_batch_size=32,train_chunk_batch_size=64,grad_accum=1,
                   auto_batch=False,effective_batch_queries=32,max_train_batch=32,grad_checkpoint=True,
                   epochs=3,lr=5e-5,weight_decay=.01,warmup_frac=.1)
    return result


RUNNER=r'''"""Phase launcher. Recipe provenance is documented beside the JSON; weights stay external."""
import argparse, importlib, inspect, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'common'))
PHASE=__PHASE__
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
'''


def assemble():
    DEST.mkdir(parents=True,exist_ok=True);COMMON.mkdir(exist_ok=True)
    manifest={}
    for folder in ['lm_lib','models','adapters','tests']:
        for src in sorted((ROOT/folder).rglob('*.py')):
            rel=src.relative_to(ROOT);target=COMMON/rel;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src,target)
            manifest[str(rel).replace('\\','/')]=hashlib.sha256(src.read_bytes()).hexdigest()
        for src in (ROOT/folder).glob('*.md'):
            target=COMMON/src.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
    clean=COMMON/'notebooks';clean.mkdir(exist_ok=True)
    for src in sorted((ROOT/'notebooks').glob('*.ipynb')):
        if src.stat().st_size==0:continue
        n=json.loads(src.read_text(encoding='utf-8'))
        n['metadata']={k:v for k,v in n.get('metadata',{}).items() if k in ['kernelspec','language_info','accelerator']}
        for cell in n['cells']:
            cell['metadata']={}
            if cell['cell_type']=='code':cell['outputs']=[];cell['execution_count']=None
        (clean/src.name).write_text(json.dumps(n,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
    shutil.copyfile(ROOT/'report/support/make_source_zip.py',COMMON/'make_code_zip.py')
    recipes_by_phase=recipes()
    for phase,cfg in recipes_by_phase.items():
        folder=DEST/f'phase{phase}';folder.mkdir(exist_ok=True)
        dump(cfg,folder/'recipe.json')
        source=ROOT/'models/A_qwen_dense.py';shutil.copyfile(source,folder/'best_model.py')
        (folder/'run.py').write_text(RUNNER.replace('__PHASE__',str(phase)),encoding='utf-8')
        ident,filename,score=WINNERS[phase]
        dump(dict(phase=phase,task=TASKS[phase],best_family='A_qwen_dense',submission_id=ident,
                  archive=filename,competition_primary_score=score,evidence='cs781_results.docx',
                  local_run_mapping='inferred from filename and notebook comparison; settings hash unavailable'),
             folder/'selected_submission.json')
        uncertainties={
          1:'Exact winning decoding settings, tokenizer artifact hash and consumed-token count were not retained.',
          2:'Only the notebook default full-A recipe is available; exact winning epoch, checkpoint, decoding settings and hyperparameters are unverified.',
          3:'A_rr_2.zip is the best upload, but the change from A_rr.zip is not retained. This recipe is an implementation reconstruction, not the exact winner config.',
          4:'Chunk selection/pooling/threshold and local full-A run are observed. Epoch/batch/LR values are reconstructed from the retained recipe; original A config.json is missing.',
          5:'Run A_full__spread4x512_3 and its scores are observed. The final saved configuration cell belongs to D; A epoch/checkpoint choices require the original run export.',
          6:'A_lora run and summary budget/holdout metrics are observed. The last saved cell belongs to A_qlora; A_lora epoch/latest-vs-best settings still need its original run export.',
          7:'A full local run, physical/effective batch 32, embedding batch 1024 and 3-epoch source recipe are observed. The exact run artifact/config is external.'}[phase]
        dump(dict(status='reconstructed recipe; not a claimed exact winning-run export',
                  source_notebooks=[s+'.ipynb' for s in NOTEBOOKS[phase]],
                  observed=uncertainties,required_external=['official prepared data','original Phase-1 tokenizer and weights',
                             'task best checkpoint for inference','original run config for exact experiment reproduction']),folder/'config_provenance.json')
        text=f'''# Phase {phase}: {TASKS[phase]}

Selected competition upload: **{filename}**, ID **{ident}**, primary score **{score:.4f}** (Finished).
The selected family is A, the 280,805,120-parameter dense decoder. `best_model.py` is an unchanged
source snapshot; runtime imports the identical implementation in `../common/models/A_qwen_dense.py`.

## Configuration provenance

{uncertainties}

`recipe.json` is deliberately named as a reconstruction. It contains usable starting paths/settings
for the implementation; it does not certify exact reproduction of the leaderboard number. Replace it
with the original external run config when available. No checkpoint or raw dataset is distributed.

## Run

Install `../requirements.txt` in Python 3.12, and prepare the official files with the clean notebooks:
{', '.join('`../common/notebooks/'+s+'.ipynb`' for s in NOTEBOOKS[phase])}.

```powershell
python code/phase{phase}/run.py show-config
python code/phase{phase}/run.py train --data-dir /path/to/prepared/phase{phase}'''+(' --phase1-run /path/to/A_qwen_dense_full' if phase>1 else '')+f'''
python code/phase{phase}/run.py predict --run-dir /path/to/trained/run
```

Commands are run from the extracted archive root. `--config`, `--out-dir`, and `--phase1-ckpt` allow
explicit overrides. Use the original Phase-1 checkpoint for all adaptations; A full-data pretraining
is distinct from the original Phase-1 10% architecture-comparison run. Legacy predictors use the
data path saved in their external run config. See the root code README for schemas and methodological details.
'''
        if phase==1:text+='\nFor Phase-1 prediction also pass `--data-dir /path/to/data` containing `test.json`. The legacy GPU generator expects bf16-capable hardware.\n'
        (folder/'README.md').write_text(text,encoding='utf-8')
    full=dict(recipes_by_phase[1],run_name='A_qwen_dense_full',train_bin='train_full.bin',snapshot_every=2000,keep_stable_ckpt=True)
    dump(full,DEST/'phase1/foundation_full_recipe.json')
    dump(dict(source_sha256=manifest,selected_architecture_sha256=hashlib.sha256((ROOT/'models/A_qwen_dense.py').read_bytes()).hexdigest()),DEST/'source_manifest.json')
    shutil.copyfile(ROOT/'README.md',DEST/'README.md')
    (DEST/'requirements.txt').write_text('''# Training/inference dependencies; historical exact versions were not fully recorded.
# Install the CUDA-compatible torch build for your runtime; BERTScore encoder is scoring-only.
torch>=2.5
numpy
tokenizers
ijson
huggingface_hub[hf_xet]
pandas
scikit-learn
rouge-score
bert-score
sacrebleu
nbformat
# Optional: flash-linear-attention (architecture C; CUDA), bitsandbytes (QLoRA on A; CUDA)
''')
    (DEST/'.gitignore').write_text('''__pycache__/
*.pyc
*.pt
*.pth
*.ckpt
*.safetensors
*.bin
.env
hf_token*.txt
runs/
data/
cs781_code.zip
''')
    evidence=DEST/'evidence';evidence.mkdir(exist_ok=True)
    for name in ['official_submissions.json','official_submissions.csv','phase_bests.json','local_variant_tables.json','learning_curves.json','padding_diagnostic.json','summary_repetition_examples.json','evidence_sha256.json']:
        path=ROOT/'report/evidence'/name
        if path.exists():shutil.copyfile(path,evidence/name)
    submissions=DEST/'submissions';submissions.mkdir(exist_ok=True)
    ledger=json.loads((ROOT/'report/evidence/official_submissions.json').read_text())
    for row in ledger:
        phase=row['phase'];ident=row['submission_id'];status=row['status']
        score='not displayed' if row['score'] is None else f"{row['score']:.4f}"
        filename=row['filename'];lower=filename.lower()
        family=('A' if lower.startswith('a') or 'qwen' in lower else
                'B' if lower.startswith('b') or 'encdec' in lower else
                'C' if lower.startswith('c') or 'gdn' in lower else
                'D' if lower.startswith('d') or 'mla' in lower else 'unverified')
        selected=ident==WINNERS[phase][0]
        note=('This is the selected best Finished upload for this phase.' if selected else
              'This is a comparison attempt; its complete historical run export was not retained.')
        if status!='Finished':note+=' Its status excludes it from the reported Finished results.'
        text=f'''# Submission {ident}: Phase {phase}

| Field | Transcribed evidence |
|---|---|
| Archive | `{filename}` |
| Displayed timestamp | {row['displayed_timestamp']} |
| Status | {status} |
| Displayed primary score | {score} |
| Architecture family inferred from filename | {family} |
| Source | {row['evidence']} |

{note}

The filename mapping is not an artifact hash. It does not establish the exact adapter, checkpoint,
learning rate or epoch count. The available shared implementations are in `../common/models`,
`../common/adapters` and `../common/lm_lib`. The Phase {phase} task implementation, schema and run
instructions are documented in `../phase{phase}/README.md` and `../README.md`.
The reconstructed selected-A recipe is in `../phase{phase}/recipe.json`; it is not claimed as the
configuration for a different attempt. Local variant tables provide measured adaptation methods when
retained; see `../evidence/local_variant_tables.json` and the report for that distinction.
No prediction archive, model weights, tokenizer artifact or historical run configuration is fabricated.
'''
        (submissions/f'{ident}.md').write_text(text,encoding='utf-8')
    for path in DEST.rglob('*.py'):ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    print('Built phase1..phase7 with common code, clean notebooks, configs and provenance')


if __name__=='__main__':assemble()
