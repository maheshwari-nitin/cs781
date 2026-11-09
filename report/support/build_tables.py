"""Generate typeset tables directly from the audited numerical JSON files."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'evidence'


def esc(s):
    return str(s).replace('\\',r'\textbackslash{}').replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('#',r'\#')


def make():
    ledger=json.loads((E/'official_submissions.json').read_text())
    lines=[r'\small',r'\begin{longtable}{crllr}',
           r'\caption{All displayed submission attempts. Status is transcribed exactly; non-finished scores are not validated results.}\\',
           r'\toprule P & ID & File & Status & Score\\\midrule\endfirsthead',
           r'\toprule P & ID & File & Status & Score\\\midrule\endhead',
           r'\bottomrule\endfoot']
    for row in ledger:
        value=f"{row['score']:.4f}" if row['score'] is not None else r'\NA'
        lines.append(f"{row['phase']} & {row['submission_id']} & {esc(row['filename'])} & {row['status']} & {value}"+r'\\')
    lines.extend([r'\end{longtable}',r'\normalsize'])
    (E/'official_ledger.tex').write_text('\n'.join(lines)+'\n')
    data=json.loads((E/'local_variant_tables.json').read_text());lines=[]
    for phase in ['4','5','6','7']:
        rows=data[phase]
        lines += [r'\subsection{Phase '+phase+r' local variants}',r'\begin{table}[H]\centering\small']
        if phase=='4':
            lines += [r'\begin{tabular}{lrr}',r'\toprule Variant & Dev macro-F1 & Trainable (\%)\\\midrule']
            for r in rows:
                label=('A' if r['phase1_run'].startswith('A') else 'B')+' '+('KD' if r['kd_teacher'] else r['method'])
                lines.append(f"{label} & {float(r['best_dev_macro_f1']):.6f} & {float(r['trainable_pct']):.4f}"+r'\\')
        elif phase=='5':
            lines += [r'\begin{tabular}{lrrr}',r'\toprule Variant & Macro-F1 & Micro-F1 & Subset acc.\\\midrule']
            for r in rows:
                label=r['phase1'][0]+' '+('KD' if r['kd']=='True' else r['method'])
                lines.append(f"{label} & {float(r['fixed_macro_f1']):.6f} & {float(r['micro_f1']):.6f} & {float(r['subset_accuracy']):.6f}"+r'\\')
        elif phase=='6':
            lines += [r'\begin{tabular}{lrrr}',r'\toprule Variant & BERTScore & ROUGE-1 & ROUGE-L\\\midrule']
            for r in rows:
                label=r['phase1'][0]+' '+r['method']
                lines.append(f"{label} & {float(r['bertscore_f1']):.6f} & {float(r['rouge1_f1']):.6f} & {float(r['rougeL_f1']):.6f}"+r'\\')
        else:
            lines += [r'\begin{tabular}{lrrr}',r'\toprule Variant & Micro-F1@7 & P@7 & R@7\\\midrule']
            for r in rows:
                label=r['phase1'][0]+' '+r['method']
                lines.append(f"{label} & {float(r['micro_f1_at7']):.6f} & {float(r['micro_precision_at7']):.6f} & {float(r['micro_recall_at7']):.6f}"+r'\\')
        lines += [r'\bottomrule\end{tabular}',r'\end{table}']
        lines.append('Source: saved comparison output in '+r'\code{notebooks/'+{'4':'30_cjpe','5':'40_lsi','6':'50_summ','7':'60_pcr'}[phase]+r'.ipynb}. The exact run identifiers are:')
        for r in rows:lines.append(r'\par\noindent\small\code{'+r['run']+r'}\normalsize')
        if phase=='6':
            lines.append('All rows use 6,678 training documents and the same 32-document evaluation subset; no KD row is present.')
        if phase=='7':
            lines.append('All rows use 817 training queries and 118 dev queries. The physical/effective query batch is 32 and embedding batch is 1,024; no KD row is present.')
    (E/'local_tables.tex').write_text('\n'.join(lines)+'\n')
    print('Generated ledger and local-result LaTeX tables')


if __name__=='__main__':make()
