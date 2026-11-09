"""Validate assembled source, notebook sanitation, launchers and recipe field names without weights."""
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
CODE=ROOT/'submission_work/package/code'
if not CODE.exists():CODE=ROOT/'code'


def main():
    manifest=json.loads((CODE/'source_manifest.json').read_text())['source_sha256']
    for rel,digest in manifest.items():
        assert hashlib.sha256((CODE/'common'/rel).read_bytes()).hexdigest()==digest,rel
    files=list(CODE.rglob('*.py'))
    for path in files:ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    notebooks=list((CODE/'common/notebooks').glob('*.ipynb'))
    for path in notebooks:
        notebook=json.loads(path.read_text(encoding='utf-8'))
        assert notebook['nbformat']==4,path
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                assert cell['outputs']==[] and cell['execution_count'] is None,path
    modules={1:('config','TrainConfig'),2:('ner/trainer','NERConfig'),3:('rr/trainer','RRConfig'),
             4:('cjpe/trainer','CJPEConfig'),5:('lsi/trainer','LSIConfig'),
             6:('summ/trainer','SummConfig'),7:('pcr/trainer','PCRConfig')}
    for phase,(module,klass) in modules.items():
        folder=CODE/f'phase{phase}'
        tree=ast.parse((CODE/'common/lm_lib'/f'{module}.py').read_text(encoding='utf-8'))
        config_class=next(c for c in tree.body if isinstance(c,ast.ClassDef) and c.name==klass)
        fields={n.target.id for n in config_class.body if isinstance(n,ast.AnnAssign)}
        cfg=json.loads((folder/'recipe.json').read_text())
        assert not set(cfg)-fields,(phase,sorted(set(cfg)-fields))
        result=subprocess.run([sys.executable,str(folder/'run.py'),'show-config'],capture_output=True,text=True,check=True)
        assert json.loads(result.stdout)==cfg,phase
        assert (folder/'best_model.py').read_bytes()==(CODE/'common/models/A_qwen_dense.py').read_bytes()
    assert len(list((CODE/'submissions').glob('*.md')))==33
    result=dict(python_sources=len(files),clean_notebooks=len(notebooks),phase_launchers=7,
                individual_attempt_docs=33,source_hashes_checked=len(manifest),passed=True)
    (CODE/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
