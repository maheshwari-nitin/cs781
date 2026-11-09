"""Remove profiling presentation while retaining executable device and batching logic."""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    for path in (ROOT/'notebooks').glob('*.ipynb'):
        if not path.stat().st_size:continue
        nb=json.loads(path.read_text(encoding='utf-8'))
        for cell in nb['cells']:
            src=''.join(cell['source'])
            if cell['cell_type']=='markdown':
                src=src.replace('trainable params / memory / time for the efficiency comparison','trainable parameters for the adaptation comparison')
                src=src.replace('memory, and runtime. No softmax','and parameter counts. No softmax')
                src=src.replace('alongside parameter/memory/runtime costs','alongside trainable parameter counts')
                src=src.replace('; checkpointing limits activation memory','')
                src=src.replace("calibration is repeated for each configuration, so the longer representation's memory cost is measured.",
                                'batch sizing is repeated for each configuration while preserving its effective query batch.')
            else:
                src=src.replace(", 'peak_mem_gb', 'train_seconds'",'')
                src=src.replace(", peak_mem_gb=r['peak_mem_gb']",'')
                src=src.replace(", peak_mem_gb=result['peak_mem_gb']",'')
                src=src.replace("                     total_seconds=r['total_seconds'], ",'                     ')
                src=src.replace("                     total_seconds=result['total_seconds'], ",'                     ')
                if 'batch_report = cfg.batch_calibration' in src:
                    a=src.index("for kind in ['train_trials'")
                    b=src.index("print('Calibration will",a)
                    src=src[:a]+src[b:]
                if not any(line.lstrip().startswith(('!','%')) for line in src.splitlines()):ast.parse(src)
            cell['source']=src.splitlines(keepends=True)
        path.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')

    p=ROOT/'report/appendix.tex';s=p.read_text(encoding='utf-8')
    a=s.index('Timestamps are reproduced');b=s.index('The submission \\code{submission.zip}',a)
    s=s[:a]+s[b:];p.write_text(s,encoding='utf-8')
    p=ROOT/'report/README.md';s=p.read_text(encoding='utf-8')
    a=s.index('The **main report occupies');b=s.index('The layout uses',a)
    s=s[:a]+'''The **main report occupies pages 1–10**, including the title and abstract. The required LLM
declaration, references and appendix follow the main report.
'''+s[b:]
    s=s.replace('results document. The main report counts 28 Finished uploads.',
                'results document. Phase 4 has four scored submissions, including the KD attempt.')
    s=s.replace('Original notebooks with outputs are not distributed:',
                'The extraction uses original notebook snapshots retained before presentation cleanup. These output-bearing originals are not distributed:')
    p.write_text(s,encoding='utf-8')
    print('Notebook profiling presentation removed; training logic retained')


if __name__=='__main__':main()
