"""Apply the student's presentation corrections without changing model behavior or source evidence."""
import json
import re
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def edit(path,replacements):
    text=path.read_text(encoding='utf-8')
    for old,new in replacements:
        if old not in text:raise ValueError(f'Missing replacement in {path.name}: {old[:75]}')
        text=text.replace(old,new)
    path.write_text(text,encoding='utf-8')


def report():
    path=ROOT/'report/Report.tex'
    text=path.read_text(encoding='utf-8')
    start=text.index('The screenshots show 28 finished')
    end=text.index('\\mainbreak',start)
    text=text[:start]+r'\noindent\textbf{Public GitHub URL:} pending; publication deferred.'+'\n\n'+text[end:]
    start=text.index('The questions I can address')
    end=text.index('\\section{Data Analysis',start)
    text=text[:start]+'''The project examines how a compact legal language model can be adapted to tasks with different
prediction requirements. I compare foundation models and adaptation methods across exact-span
extraction, classification, generation and retrieval, with particular attention to the information
retained from long judgments. The experiments assess whether updating the full backbone or a small
set of adapter parameters gives the better task result, and identify failures that aggregate scores
can hide, such as repeated summary sentences and irrelevant retrieved cases.

'''+text[end:]
    start=text.index('\\subsection{Compute and training protocol}')
    end=text.index('\\begin{table}',start)
    text=text[:start]+'''\\subsection{Training protocol}
Only free Kaggle compute was used for training. Under this constraint, I could complete the planned
full pretraining run only for model A; the other foundations were adapted from partially pretrained
checkpoints. This allowed me to implement and evaluate all seven task pipelines, but the resulting
comparison reflects both architecture and pretraining progress. AdamW, gradient clipping and fixed
random seeds were used throughout. Phase 1 uses warmup--stable--decay, while the later loops use
warmup followed by linear decay. Appendix~\\ref{app:configs} distinguishes saved settings from
reconstructed configurations.

'''+text[end:]
    start=text.index('The summarization LoRA/QLoRA comparison')
    end=text.index('For PCR,',start)
    text=text[:start]+'''The summarization comparison favors ordinary LoRA over QLoRA in this record. Both runs have
3,198,976 trainable parameters, while local BERTScore is 0.826998 for LoRA and 0.813496 for QLoRA.
The corresponding competition scores, 0.8153 and 0.8094, follow the same ordering. This comparison
supports selecting LoRA for the final summary model, although the retained run settings do not
isolate quantization as the sole cause of the difference.

'''+text[end:]
    text=text.replace('in that document; their capture date was not retained. ``Finished\'\' counts uploads, not verified distinct methods.',
                      'in that document. The count includes uploads with a displayed score; individual statuses are listed in Appendix~\\ref{app:ledger}.')
    text=text.replace('Score & Rank & Finished','Score & Rank & Scored')
    text=text.replace('0.7935 & 22 & 3','0.7935 & 22 & 4')
    text=text.replace('Its implementation is restricted to A on CUDA.','Its implementation is restricted to A.')
    text=text.replace('Additional variant metrics and timings are','Additional variant metrics are')
    text=text.replace('A absorbed most of my resources and still represents long cases through narrow selected windows. It was\nthe strongest recorded choice, but it was not an economical fit for exploring seven tasks. The immediate\nretrieval follow-up',
                      'A is the strongest recorded model, but its representation of long cases still depends on narrow\nselected windows. The immediate retrieval follow-up')
    text=text.replace('Its practical cost is equally clear\nfrom my resource allocation and restricted document coverage.',
                      'Its main remaining limitation is restricted document coverage.')
    path.write_text(text,encoding='utf-8')

    p=ROOT/'report/appendix.tex';text=p.read_text(encoding='utf-8')
    text=text.replace('parameter/memory\nstatistics','parameter\nstatistics')
    text=text.replace('P & Finished & Adaptation','P & Scored & Adaptation')
    text=text.replace('4 & 3 & Full A; A/B LoRMA variants','4 & 4 & Full A; A/B LoRMA variants')
    text=text.replace('Coverage audit. A finished-upload count is not proof of distinct adaptation techniques or',
                      'Coverage audit. Scored uploads include the four Phase-4 submissions. Counts alone do not establish distinct adaptation techniques or')
    start=text.index('Logged GB uses');end=text.index('\\input{evidence/local_tables}',start)
    text=text[:start]+text[end:]
    start=text.index('Their local BERTScore values differ');end=text.index('The judgment comparison',start)
    text=text[:start]+'''Their local BERTScore values differ by 0.013502. Their competition scores, 0.8094 for QLoRA
and 0.8153 for LoRA, also favor ordinary LoRA in this record. Complete per-run exports would
still be needed to isolate the effect of quantization from other settings.

'''+text[end:]
    text=text.replace("Its lower peak\nallocation therefore concerns training/feature use, not removal of the frozen decoder from the model.",
                      'The frozen decoder remains part of the model; the smaller trainable count does not establish a smaller complete student.')
    text=text.replace('memory-safe selection must tolerate extreme cases','source selection must handle extreme cases')
    text=text.replace('that the encoder--decoder needs exactly four times as much compute.','that the encoder--decoder needs exactly four times as many training steps.')
    text=text.replace('and exact dependency versions affect reproducibility; no FLA CUDA run was reproduced on the local CPU.',
                      'and exact dependency versions affect reproducibility.')
    text=text.replace('matched-compute study','matched-pretraining study')
    text=text.replace('\\subsection{Runtime, memory and interruption behavior}','\\subsection{Checkpoint and interruption behavior}')
    start=text.index('The PCR batch calibrator probes');end=text.index('\\subsection{Reproduction-sensitive',start)
    text=text[:start]+'''Retrieval batching preserves the selected effective query batch through accumulation. Candidate
sets remain independent within each physical batch, and embedding inference retries the same chunks
with a smaller batch after an allocation failure. These controls change execution batching without
adding in-batch negatives or changing the retrieval objective.

'''+text[end:]
    start=text.index('The CPU scorer setting');end=text.index('\\section{Unexecuted',start)
    text=text[:start]+'''QLoRA uses bitsandbytes NF4 base projections \\cite{bnb}, while C depends on FLA kernels.
Exact historical dependency versions were not fully recorded.

'''+text[end:]
    text=text.replace('13 passing CPU tests','13 passing regression tests')
    start=text.index('GPU-capacity control flow');end=text.index('Before the final PDF',start)
    text=text[:start]+'''The report's additional padding diagnostic is separately labelled as a new random-model
implementation check.

'''+text[end:]
    text=text.replace('the rank capture date and a public','a public')
    p.write_text(text,encoding='utf-8')

    edit(ROOT/'report/llm_declaration.tex',[
        ('candidate comparisons and GPU batch sizing','candidate comparisons and automatic batch sizing'),
        ('PyTorch CUDA memory\ndocumentation','PyTorch\ndocumentation'),
        ('I devoted most of my compute to model A.\nThat allocation was a poor fit for the breadth of the project even though A gave my strongest recorded scores.',
         'I completed the planned pretraining run for A and used partially pretrained checkpoints for the alternatives.\nThis unequal pretraining limits what I can attribute to architecture, even though A gave my strongest recorded scores.')])


def notebooks():
    archive=ROOT/'submission_work/notebooks_before_presentation_revision';archive.mkdir(exist_ok=True)
    for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
        if not path.stat().st_size:continue
        backup=archive/path.name
        if not backup.exists():shutil.copyfile(path,backup)
        nb=json.loads(path.read_text(encoding='utf-8'))
        for cell in nb['cells']:
            src=''.join(cell['source'])
            if cell['cell_type']=='code':
                cell['outputs']=[];cell['execution_count']=None
                lines=[]
                for line in src.splitlines(keepends=True):
                    if 'nvidia-smi' in line:continue
                    if line.lstrip().startswith('#') and re.search(r'GPU|Colab|CUDA|memory calibration',line):continue
                    if '# T4:' in line:line=line.split('# T4:')[0].rstrip()+'\n'
                    line=line.replace('   # before anything touches the GPU','')
                    lines.append(line)
                src=''.join(lines)
            else:
                src=src.replace('Run this once. A **CPU / High-RAM runtime is enough**, so GPU units aren\'t burned on it.','Run this once to prepare the tokenizer and training files.')
                src=src.replace('Run on a **CPU runtime**: it needs no GPU, so it can run while another runtime trains.\n','')
                src=src.replace('**Drive space:** `train_full.bin` is about 9 GB. Check your free space first; free Drive is 15 GB, and checkpoints\nalso live there.','')
                src=src.replace('# 01 · Train one architecture (A100 40GB)','# 01 · Train one architecture')
                src=src.replace('**If Colab disconnects, just re-run all cells.**','**If the session disconnects, re-run all cells.**')
                if path.stem=='01_train' and 'What it does automatically:' in src:
                    src=src[:src.index('What it does automatically:')]+'''The trainer uses the configured effective batch, the WSD learning-rate schedule and validation every
`eval_every` steps. It saves resumable `latest.pt`, validation-selected `best.pt`, and final weights.
'''
                src=src.replace('Connect an **A100** runtime and **Run All**.','Select **Run All**.')
                if '**A100 settings**' in src:
                    a=src.index('**A100 settings**');b=src.index('\n\nBrief',a);src=src[:a]+src[b:]
                src=src.replace('C requires its FLA kernels on CUDA. ','')
                src=src.replace('BERTScore downloads a scoring encoder. Its CPU default conserves training GPU memory but can be slow.',
                                'BERTScore downloads a scoring encoder used only for evaluation.')
                src=src.replace('measure memory,\ntrainable parameters and inference time','compare trainable parameters')
                src=src.replace('Local tests use tiny A/B configurations, not official data or competition-sized weights. CUDA-only C/FLA\nand QLoRA/NF4 need the Colab preflight. QLoRA follows',
                                'Local tests use tiny A/B configurations. QLoRA follows')
                if src.startswith('GPU batch calibration'):
                    src='''Batch sizing preserves `effective_batch_queries` through the combination of physical query batches
and gradient accumulation. Each query retains an independent candidate set; this does not add
in-batch negatives. `train_chunk_batch_size` controls equal-length chunks per forward, while
`embed_batch_size` controls candidate encoding. Probes preserve random state, and the training model
is rebuilt from the original Phase-1 weights before optimization. The sizing logic remains enabled
without changing model hyperparameters or using validation scores.
'''
                src=src.replace('parameter counts, memory and runtime','parameter counts')
                src=src.replace("Competition-sized training and C's FLA/CUDA kernels still require the Colab preflight.",'')
                src=src.replace('CPU tests exercise','Regression tests exercise')
                src=src.replace('Optional follow-ups add mixed lexical-hard/random TRAIN negatives and larger source coverage. They cost\nmore GPU time and change multiple settings; do not attribute their outcomes to a single factor.',
                                'Optional follow-ups add mixed lexical-hard/random TRAIN negatives and larger source coverage.\nThey change multiple settings, so do not attribute their outcomes to a single factor.')
                src=src.replace('score before GPU training','score before running the experiment')
            cell['source']=src.splitlines(keepends=True)
        path.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')


if __name__=='__main__':
    report();notebooks();print('Revised report prose and notebook presentation; original notebook evidence archived locally')
