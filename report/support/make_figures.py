"""Vector report figures from supplied observations; no synthetic learning curves."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8, "axes.spines.top":False,
                     "axes.spines.right":False,"pdf.fonttype":42,"savefig.bbox":"tight"})
BLUE, ORANGE, GREEN, GRAY = "#24577a", "#b05d27", "#2d7161", "#737b84"


def pipeline():
    fig, ax=plt.subplots(figsize=(7.5,1.7));ax.set_xlim(0,10);ax.set_ylim(0,2);ax.axis("off")
    def box(x,y,w,h,text,color=BLUE):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04",lw=.8,
                                   edgecolor=color,facecolor="#f3f6f8"))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=7.5)
    def arrow(x,y,xx,yy):ax.annotate("",xy=(xx,yy),xytext=(x,y),arrowprops=dict(arrowstyle="->",lw=.8,color=GRAY))
    box(.03,.55,1.25,.9,"Official legal\ncorpus\n32k byte BPE")
    box(1.67,.45,2.7,1.1,"A: embedding → 22 pre-norm blocks\nGQA + QK norm + RoPE\nSwiGLU; tied LM output")
    ax.text(3.0,.10,"B: enc–dec  |  C: GDN hybrid  |  D: MLA–MoE",ha="center",fontsize=6.7)
    arrow(1.32,1.0,1.61,1.0)
    box(4.78,.55,1.35,.9,"Independent\nfine-tuned\ndescendants",GREEN);arrow(4.42,1,4.72,1)
    labels=[("P2: spans",1.2),("P3: roles",.63),("P4: outcome",.06)]
    for text,y in labels:box(6.57,y,1.28,.43,text);arrow(6.17,1,6.52,y+.21)
    for text,y in [("P5: statutes",1.2),("P6: summary",.63),("P7: top-7",.06)]:
        box(8.30,y,1.30,.43,text);arrow(6.17,1,8.25,y+.21)
    ax.text(3.0,1.83,"P1: autoregressive / prefix-continuation pretraining",ha="center",fontsize=7.3)
    fig.savefig(FIG/"pipeline.pdf");plt.close(fig)


def lengths():
    fig, axes=plt.subplots(1,3,figsize=(7.1,1.85))
    groups=[("LSI cases",["Train","Dev","Test"],[[430,2766,8142],[543,2957,8855],[532,2811,8306]]),
            ("Summarization",["Document","Reference"],[[3129,7852,24246],[638,1501,4250]]),
            ("Retrieval (train)",["Query","Candidate"],[[4508,12602,42997],[3632,12422,47409]])]
    for ax,(title,labels,values) in zip(axes,groups):
        for i,row in enumerate(values):
            ax.plot([row[0],row[2]],[i,i],color=GRAY,lw=1.6)
            for j,color,marker in [(0,BLUE,"o"),(1,ORANGE,"s"),(2,GREEN,"|")]:
                ax.scatter(row[j],i,s=19,color=color,marker=marker,zorder=3)
        ax.set_xscale("log");ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis()
        ax.set_title(title,fontweight="bold",fontsize=8);ax.set_xlabel("Words (log scale)")
        ax.grid(axis="x",alpha=.18);ax.set_ylim(len(labels)-.5,-.5)
    fig.text(.5,-.06,"● median     ■ 90th percentile     | 99th percentile",ha="center",fontsize=7.5)
    fig.tight_layout(w_pad=1.5);fig.savefig(FIG/"length_profiles.pdf");plt.close(fig)


def curves():
    d=json.loads((ROOT/"evidence/learning_curves.json").read_text())
    fig, axes=plt.subplots(1,3,figsize=(7.1,1.8))
    specs=[("cjpe_kd_dev","step","macro_f1","CJPE: B KD","Dev macro-F1",BLUE),
           ("lsi_kd_running_loss","batch","running_loss","LSI: D KD","Cumulative train loss",ORANGE),
           ("pcr_c_dev","step","micro_f1","PCR: C partial","Dev micro-F1@7",GREEN)]
    for ax,(key,x,y,title,ylabel,color) in zip(axes,specs):
        rows=d[key];ax.plot([v[x] for v in rows],[v[y] for v in rows],color=color,lw=1.3,
                           marker="o" if len(rows)<10 else None,markersize=3)
        ax.set_title(title,fontweight="bold",fontsize=8);ax.set_xlabel("Optimizer step" if x=="step" else "Training microbatch")
        ax.set_ylabel(ylabel);ax.grid(alpha=.15)
    fig.tight_layout(w_pad=1.0);fig.savefig(FIG/"learning_curves.pdf");plt.close(fig)


def padding():
    d=json.loads((ROOT/"evidence/padding_diagnostic.json").read_text())
    fig,ax=plt.subplots(figsize=(3.1,1.7))
    for arch,color,label in [("A_qwen_dense",BLUE,"A (causal)"),("B_encdec",ORANGE,"B (bidirectional)")]:
        rows=[r for r in d["results"] if r["architecture"]==arch]
        ax.plot([r['padding_tokens'] for r in rows],[r['relative_l2'] for r in rows],marker='o',
                markersize=3,color=color,label=label)
    ax.set_xlabel("Added right-padding tokens");ax.set_ylabel("Relative feature change")
    ax.legend(frameon=False,fontsize=7);ax.grid(alpha=.15);fig.tight_layout()
    fig.savefig(FIG/"padding_diagnostic.pdf");plt.close(fig)


if __name__=="__main__":
    pipeline();lengths();curves();padding()
    print("Created four original vector-PDF figures")
