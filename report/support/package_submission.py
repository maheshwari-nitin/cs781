"""Package an explicit allowlist with code/ and report/ directly at the ZIP root."""
import ast
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
STAGING=ROOT/'submission_work/package'
STEM='CS781-CP-Nitin-Maheshwari-230711'
PDF_NAME=STEM+'-report.pdf'
ZIP_NAME=STEM+'-report_code.zip'


def check_members(members):
    names=list(members)
    assert {n.split('/')[0] for n in names}=={'code','report'}
    assert f'report/{PDF_NAME}' in names
    assert {f'phase{i}' for i in range(1,8)} <= {n.split('/')[1] for n in names if n.startswith('code/')}
    forbidden={'.pt','.pth','.ckpt','.safetensors','.bin','.pyc','.exe','.dll'}
    for name in names:
        path=Path(name)
        assert path.suffix.lower() not in forbidden,name
        assert not any(part in {'__pycache__','.git','runs','checkpoints'} for part in path.parts),name
        assert not path.name.lower().startswith(('hf_token','.env')),name
        if path.suffix=='.py':ast.parse(members[name].decode('utf-8'),filename=name)
        if path.suffix=='.ipynb':
            nb=json.loads(members[name])
            assert all(c.get('outputs',[])==[] and c.get('execution_count') is None
                       for c in nb['cells'] if c['cell_type']=='code'),name
    # Compare against the local credential without ever printing its contents.
    token_path=ROOT/'hf_token.txt'
    secret=token_path.read_bytes().strip() if token_path.exists() else b''
    if secret:
        for name,data in members.items():
            assert secret not in data,f'Credential found in {name}'
            if name.endswith('.docx'):
                import io
                with zipfile.ZipFile(io.BytesIO(data)) as docx:
                    for part in docx.namelist():
                        assert secret not in docx.read(part),'Credential in results document'


def main():
    audit=json.loads((ROOT/'report/audit.json').read_text())
    assert audit['main_within_ten_pages'] and not audit['warnings']
    assert json.loads((STAGING/'code/validation.json').read_text())['passed']
    tests=json.loads((ROOT/'report/test_validation.json').read_text())
    assert tests['passed'] and tests['tests_run']==13
    dest=STAGING/'report';dest.mkdir(exist_ok=True)
    for name in ['Report.tex','appendix.tex','llm_declaration.tex','references.bib','README.md',
                 'metadata.json','audit.json','test_validation.json']:
        shutil.copyfile(ROOT/'report'/name,dest/name)
    shutil.copyfile(ROOT/'report/Report.pdf',ROOT/'report'/PDF_NAME)
    shutil.copyfile(ROOT/'report'/PDF_NAME,dest/PDF_NAME)
    for folder,extensions in [('figures',{'.pdf'}),('evidence',{'.json','.csv','.tex'}),('support',{'.py'})]:
        for src in sorted((ROOT/'report'/folder).iterdir()):
            if src.is_file() and src.suffix in extensions:
                target=dest/folder/src.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(src,target)
    shutil.copyfile(ROOT/'cs781_results.docx',dest/'evidence/cs781_results.docx')
    members={}
    for path in sorted(STAGING.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:continue
        if path.suffix not in {'.py','.md','.json','.csv','.tex','.bib','.pdf','.txt','.ipynb','.docx'} and path.name!='.gitignore':continue
        members[path.relative_to(STAGING).as_posix()]=path.read_bytes()
    check_members(members)
    summary=dict(archive=ZIP_NAME,author='Nitin Maheshwari',roll='230711',
                 roots=['code','report'],main_report_pages=audit['main_body_pages'],
                 total_pdf_pages=audit['total_pages'],cpu_regression_tests=13,
                 clean_notebooks=13,phase_folders=7,individual_attempt_docs=33,
                 credentials_and_weights_excluded=True,
                 public_github_url='https://github.com/maheshwari-nitin/cs781',
                 github_publication='repository URL supplied by student',
                 pdf_sha256=hashlib.sha256(members[f'report/{PDF_NAME}']).hexdigest())
    members['report/submission_validation.json']=(json.dumps(summary,indent=2)+'\n').encode()
    output=ROOT/ZIP_NAME
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in members.items():z.writestr(name,data)
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        check_members({n:z.read(n) for n in z.namelist()})
    summary['archive_sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
    summary['archive_bytes']=output.stat().st_size;summary['archive_files']=len(members)
    (ROOT/'report/submission_validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
