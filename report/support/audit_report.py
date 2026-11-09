"""Check the counted-page boundary and render a contact sheet for visual review."""
import json
import re
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageDraw
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]


def main():
    pdf=ROOT/'Report.pdf';reader=PdfReader(pdf)
    aux=(ROOT/'Report.aux').read_text(encoding='utf-8')
    match=re.search(r'\\newlabel\{main-end\}\{\{[^}]*\}\{(\d+)\}',aux)
    main_end=int(match.group(1)) if match else None
    source=(ROOT/'Report.tex').read_text(encoding='utf-8')
    abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',source,re.S).group(1)
    abstract_words=len(abstract.split())
    log=(ROOT/'Report.log').read_text(encoding='utf-8',errors='replace')
    overfull=[line.strip() for line in log.splitlines() if 'Overfull' in line or 'undefined' in line.lower()]
    report=dict(total_pages=len(reader.pages),main_body_pages=main_end,abstract_words=abstract_words,
                main_within_ten_pages=main_end is not None and main_end<=10,
                warnings=overfull,page_summaries=[])
    for i,page in enumerate(reader.pages,1):
        text=page.extract_text() or ''
        report['page_summaries'].append(dict(page=i,first_text=' '.join(text.split())[:140],characters=len(text)))
    (ROOT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    doc=fitz.open(pdf)
    thumbs=[]
    render=ROOT/'rendered';render.mkdir(exist_ok=True)
    for i in range(len(doc)):
        pix=doc[i].get_pixmap(matrix=fitz.Matrix(.65,.65),alpha=False)
        im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
        im.thumbnail((350,470))
        tile=Image.new('RGB',(370,500),'#e4e7eb');tile.paste(im,((370-im.width)//2,20))
        ImageDraw.Draw(tile).text((12,478),f'Page {i+1}',fill='black');thumbs.append(tile)
        if i in (0,3,6,7,8,9):
            doc[i].get_pixmap(matrix=fitz.Matrix(1.35,1.35),alpha=False).save(render/f'page_{i+1}.png')
    columns=4;rows=(len(thumbs)+columns-1)//columns
    sheet=Image.new('RGB',(370*columns,500*rows),'white')
    for i,tile in enumerate(thumbs):sheet.paste(tile,((i%columns)*370,(i//columns)*500))
    sheet.save(render/'contact_sheet.png')
    print(json.dumps({k:v for k,v in report.items() if k!='page_summaries'},indent=2))
    for row in report['page_summaries']:print(row['page'],row['first_text'])
    if not report['main_within_ten_pages']:raise SystemExit('Main report exceeds ten pages')
    if abstract_words>150:raise SystemExit('Abstract exceeds 150 words')


if __name__=='__main__':main()
