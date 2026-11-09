"""Convert supplied notebook outputs and manually verified screenshot rows into auditable report data.

No weights or credentials are read. Screenshots were visually checked against cs781_results.docx.
"""
import csv
import hashlib
import json
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report" / "evidence"


class HTMLTable(HTMLParser):
    def __init__(self):
        super().__init__(); self.rows = []; self.row = []; self.cell = None
    def handle_starttag(self, tag, attrs):
        if tag == "tr": self.row = []
        if tag in ("td", "th"): self.cell = []
    def handle_data(self, value):
        if self.cell is not None: self.cell.append(value)
    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split())); self.cell = None
        if tag == "tr" and self.row: self.rows.append(self.row)


# Screenshot transcription: phase, submission ID, file, displayed date, status, displayed score.
ROWS = [
 (1,965311,"qwen_dense_best.zip","2026-10-06 22:55","Finished",.8273),
 (1,965281,"qwen_dense_best.zip","2026-10-06 22:37","Preparing",None),
 (1,964252,"gdn_hybrid.zip","2026-10-06 13:59","Finished",.8144),
 (1,963258,"mla_moe.zip","2026-10-06 03:45","Finished",.8077),
 (1,963212,"gdn_hybrid.zip","2026-10-06 03:06","Preparing",None),
 (1,963147,"encdec.zip","2026-10-06 01:56","Finished",.8016),
 (1,963065,"submission.zip","2026-10-06 00:52","Finished",.8114),
 (2,977524,"D_ner.zip","2026-10-11 18:03","Finished",.2918),
 (2,977517,"C_ner.zip","2026-10-11 18:02","Finished",.2984),
 (2,977456,"B_ner.zip","2026-10-11 17:46","Finished",.0006),
 (2,977453,"A_ner.zip","2026-10-11 17:46","Finished",.3703),
 (3,992390,"A_rr_2.zip","2026-10-16 13:24","Finished",.5146),
 (3,991383,"D_rr.zip","2026-10-16 04:02","Finished",.3891),
 (3,991379,"C_rr.zip","2026-10-16 03:59","Finished",.4092),
 (3,991369,"B_rr.zip","2026-10-16 03:46","Finished",.3410),
 (3,991366,"A_rr.zip","2026-10-16 03:44","Finished",.4837),
 (4,1006453,"B_cj_kd.zip","2026-10-21 04:05","Failed",.6176),
 (4,1006421,"A_cj.zip","2026-10-21 03:49","Finished",.7935),
 (4,1006399,"B_cj.zip","2026-10-21 03:42","Finished",.5914),
 (4,1006398,"A_cj_lorma.zip","2026-10-21 03:42","Finished",.5634),
 (4,1006397,"A_cj.zip","2026-10-21 03:42","Cancelled",None),
 (5,1016707,"D_lsi.zip","2026-10-28 22:54","Finished",.0730),
 (5,1016609,"B_lsi.zip","2026-10-28 22:19","Finished",.0328),
 (5,1016608,"C_lsi.zip","2026-10-28 22:19","Finished",.0951),
 (5,1016369,"A_lsi.zip","2026-10-28 21:02","Finished",.3517),
 (6,1028889,"B6f.zip","2026-11-03 05:23","Finished",.7651),
 (6,1028888,"C6.zip","2026-11-03 05:21","Finished",.7680),
 (6,1028887,"B6.zip","2026-11-03 05:21","Failed",None),
 (6,1028886,"Aq6.zip","2026-11-03 05:21","Finished",.8094),
 (6,1028885,"A6.zip","2026-11-03 05:20","Finished",.8153),
 (7,1038512,"C_pcr.zip","2026-11-07 03:27","Finished",.0732),
 (7,1038510,"B_pcr.zip","2026-11-07 03:22","Finished",.0549),
 (7,1038509,"A_pcr.zip","2026-11-07 03:22","Finished",.1855),
]
RANKS = {1:106,2:113,3:50,4:22,5:92,6:119,7:126}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = [dict(zip(["phase","submission_id","filename","displayed_timestamp","status","score"], row)) for row in ROWS]
    for row in ledger:
        row["evidence"] = f"cs781_results.docx / image{row['phase']}.png"
    (OUT/"official_submissions.json").write_text(json.dumps(ledger,indent=2)+"\n")
    with (OUT/"official_submissions.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=ledger[0]);writer.writeheader();writer.writerows(ledger)
    best=[]
    for phase in range(1,8):
        finished=[r for r in ledger if r["phase"]==phase and r["status"]=="Finished"]
        row=max(finished,key=lambda r:r["score"])
        best.append(dict(row,rank=RANKS[phase],finished_uploads=len(finished),
                         scored_uploads=sum(r['phase']==phase and r['score'] is not None for r in ledger)))
    (OUT/"phase_bests.json").write_text(json.dumps(best,indent=2)+"\n")
    tables={}; streams={}; excerpts={}
    for phase, name in [(4,"30_cjpe"),(5,"40_lsi"),(6,"50_summ"),(7,"60_pcr")]:
        archive=ROOT/'submission_work/notebooks_before_presentation_revision'
        path=(archive if archive.exists() else ROOT/"notebooks")/(name+".ipynb")
        notebook=json.loads(path.read_text(encoding="utf-8")); texts=[]
        for i,cell in enumerate(notebook["cells"]):
            for output in cell.get("outputs",[]):
                display=output.get("data",{})
                text="".join(output.get("text",[]))
                text=re.sub(r"hf_[A-Za-z0-9]{15,}","[REDACTED]",text)
                if text:texts.append(text)
                if "text/html" in display:
                    parser=HTMLTable();parser.feed("".join(display["text/html"]))
                    if parser.rows and any("run" in row for row in parser.rows):
                        header=parser.rows[0]
                        tables[str(phase)]=[dict(zip(header[1:],row[1:])) for row in parser.rows[1:]]
        joined="\n".join(texts)
        streams[str(phase)]=joined
        # Retain only actual numerical trajectories, not raw documents or login output.
    curves={
       "cjpe_kd_dev":[dict(step=int(step),macro_f1=float(score)) for step,score in
           re.findall(r"\[dev\] epoch \d+ step (\d+): macro-F1 ([0-9.]+)",streams["4"])],
       "lsi_kd_running_loss":[dict(batch=int(batch),running_loss=float(loss)) for batch,loss in
           re.findall(r"epoch 1/1 batch (\d+)/5344 loss ([0-9.]+)",streams["5"])],
       "pcr_c_dev":[dict(step=int(step),micro_f1=float(score)) for step,score in
           re.findall(r"\[dev\] step (\d+): micro-F1@7 ([0-9.]+)",streams["7"])],
    }
    (OUT/"local_variant_tables.json").write_text(json.dumps(tables,indent=2)+"\n")
    (OUT/"learning_curves.json").write_text(json.dumps(curves,indent=2)+"\n")
    manifest={}
    for path in [ROOT/"cs781_results.docx",ROOT/"Report_LLM.zip"] + list((ROOT/"notebooks").glob("*.ipynb")):
        if path.stat().st_size:
            manifest[str(path.relative_to(ROOT)).replace('\\','/')]=hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT/"evidence_sha256.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Transcribed {len(ledger)} screenshot rows; {sum(b['finished_uploads'] for b in best)} finished uploads")
    print("Extracted local tables:", {k:len(v) for k,v in tables.items()})
    print("Actual curve points:", {k:len(v) for k,v in curves.items()})


if __name__=="__main__":main()
