"""Create the source-only cs781_code.zip used by the Colab bootstrap cells.

Distributed as code/common/make_code_zip.py; run it from that location.
Authentication must be supplied through the runtime environment, not this archive.
"""
from pathlib import Path
import json
import zipfile


def main():
    root=Path(__file__).resolve().parent
    if not (root/'lm_lib').is_dir():
        raise SystemExit('Run the distributed code/common/make_code_zip.py')
    paths=[]
    for folder in ['lm_lib','models','adapters','tests']:
        paths.extend(sorted((root/folder).rglob('*.py')))
    paths.extend(sorted((root/'notebooks').glob('*.ipynb')))
    paths.append(root/'make_code_zip.py')
    out=root/'cs781_code.zip'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            if '__pycache__' in path.parts:continue
            data=path.read_bytes()
            if path.suffix=='.ipynb':
                notebook=json.loads(data)
                for cell in notebook['cells']:
                    if cell['cell_type']=='code':cell['outputs']=[];cell['execution_count']=None
                data=json.dumps(notebook,indent=1,ensure_ascii=False).encode('utf-8')
            z.writestr(path.relative_to(root).as_posix(),data)
    print(f'Created {out} with source and clean notebooks only')


if __name__=='__main__':main()
