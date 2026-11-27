"""Render tables and figures exclusively from a completed evaluation artifact."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def export(folder):
    root=Path(folder)
    r=json.loads((root/"evaluation.json").read_text(encoding="utf8"))
    labels=r["labels"]
    matrix=pd.DataFrame(r["confusion"],index=labels,columns=labels)
    matrix.to_csv(root/"confusion.csv")
    fig,ax=plt.subplots(figsize=(5,4),layout="constrained")
    ax.imshow(matrix,cmap="Blues")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j,i,str(matrix.iloc[i,j]),ha="center",va="center")
    ax.set(xticks=range(len(labels)),yticks=range(len(labels)),xticklabels=labels,yticklabels=labels,
           xlabel="Predicted class",ylabel="Independent reference")
    fig.savefig(root/"confusion.svg")
    plt.close(fig)
    cv=pd.DataFrame(r["development_cv"]).explode("fold_macro_f1")
    cv.fold_macro_f1=cv.fold_macro_f1.astype(float)
    table=cv.groupby(["model","features"]).fold_macro_f1.agg(["mean","std"])
    table.to_csv(root/"development-cv.csv")
    table.to_latex(root/"development-cv.tex",float_format="%.3f")
    ax=table["mean"].unstack("features").plot.bar(figsize=(8,4),ylim=(0,1),rot=0)
    ax.set(ylabel="Development CV macro F1",xlabel="")
    ax.figure.tight_layout()
    ax.figure.savefig(root/"feature-comparison.svg")
    plt.close(ax.figure)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("evaluation_directory")
    export(p.parse_args().evaluation_directory)
