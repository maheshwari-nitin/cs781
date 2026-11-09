"""Keep submission documentation consistent with the revised report presentation."""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=ROOT/'README.md';s=p.read_text(encoding='utf-8')
    s=s.replace('Colab notebooks, and CPU regression tests','notebooks, and regression tests')
    s=s.replace('These are not newly reproduced test scores. Rank capture dates were not saved; report freeze is\n9 November 2026. Scores across different tasks should not be averaged.',
                'These are not newly reproduced test scores. Scores across different tasks should not be averaged.')
    s=s.replace('Finished uploads |','Scored uploads |').replace('0.7935 | 22 | 3 |','0.7935 | 22 | 4 |')
    a=s.index('The screenshots show **28 Finished');b=s.index('## Repository organization',a)
    s=s[:a]+'''Phase 4 has four scored submissions, including the KD attempt. The individual statuses remain
in `report/evidence/official_submissions.csv` and the per-submission documentation. The best-score
column uses the best Finished upload for each phase.

'''+s[b:]
    s=s.replace('Colab entry points','Notebook entry points').replace('CPU regression suite','Regression suite')
    a=s.index('Historical GPU dependency versions');b=s.index('Authenticate to the official',a)
    s=s[:a]+'''Only free Kaggle compute was used for training. Under that constraint, only A completed the
planned full pretraining run; the other foundations used partially pretrained checkpoints.
Historical dependency versions were not fully retained. The local BERTScore output records
`roberta-large_L17_no-idf_version=0.3.12(hug_trans=5.22.0)`.

'''+s[b:]
    s=s.replace('A QLoRA scores 0.8094 officially and uses 2.364 GB peak allocation versus 3.388 GB for LoRA locally.',
                'A QLoRA scores 0.8094 officially, compared with 0.8153 for ordinary LoRA.')
    a=s.index('negatives. The GPU calibrator');b=s.index('The longer A recipe',a)
    s=s[:a]+'''negatives. Automatic batch sizing preserves the selected effective query batch. Embedding retries
reduce the batch and retry the same chunks without dropping candidates.

'''+s[b:]
    s=s.replace("A's high compute cost and bounded long-document context are documented.",
                "A's bounded long-document context remains a limitation.")
    s=s.replace('or rank-capture date is invented','is invented')
    p.write_text(s,encoding='utf-8')

    p=ROOT/'report/appendix.tex';s=p.read_text(encoding='utf-8')
    s=s.replace('stack; CPU; original IDs','stack; original IDs')
    s=s.replace('with a smaller batch after an allocation failure','with a smaller batch after a failed encoding attempt')
    s=s.replace('The source/evidence SHA-256 manifest and numerical transcriptions are',
                'The source/evidence SHA-256 manifest records the original output-bearing notebook snapshots, before presentation cleanup. Numerical transcriptions are')
    p.write_text(s,encoding='utf-8')
    ledger=json.loads((ROOT/'report/evidence/official_submissions.json').read_text())
    p=ROOT/'report/evidence/phase_bests.json';best=json.loads(p.read_text())
    for row in best:row['scored_uploads']=sum(x['phase']==row['phase'] and x['score'] is not None for x in ledger)
    p.write_text(json.dumps(best,indent=2)+'\n')
    p=ROOT/'report/metadata.json';metadata=json.loads(p.read_text())
    metadata['training_compute_statement']='Only free Kaggle compute was used; only A completed planned full pretraining, confirmed by student.'
    p.write_text(json.dumps(metadata,indent=2)+'\n')

    # Validate standard Python cells; shell/IPython magic cells remain notebook-specific.
    checked=0
    for path in (ROOT/'notebooks').glob('*.ipynb'):
        if not path.stat().st_size:continue
        nb=json.loads(path.read_text(encoding='utf-8'))
        for i,c in enumerate(nb['cells']):
            if c['cell_type']!='code':continue
            src=''.join(c['source'])
            if any(line.lstrip().startswith(('!','%')) for line in src.splitlines()):continue
            ast.parse(src,filename=f'{path.name}:cell{i}');checked+=1
    print(f'Revised documentation; validated {checked} Python notebook cells')


if __name__=='__main__':main()
