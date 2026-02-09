#!/usr/bin/env python

import sys
import getopt
import glob
import os
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt


def main(argv):

    keyword = None

    try:
        opts, args = getopt.getopt(
            argv,
            "hk:",
            ["keyword="]
        )
    except getopt.GetoptError:
        print(f"Usage: {__file__} -k <keyword>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print(f"\n{__file__} [OPTIONS]")
            print("\t-h, --help        Show help")
            print("\t-k, --keyword     Unique keyword used in CSV files")
            sys.exit()
        elif opt in ("-k", "--keyword"):
            keyword = arg

    if keyword is None:
        raise RuntimeError("Keyword must be provided using -k or --keyword")

    print(f"Evaluating results for keyword: {keyword}")

    # ---------- FIND CSV FILES ----------
    csv_files = glob.glob(f"results/*{keyword}*.csv")

    if len(csv_files) == 0:
        raise RuntimeError(f"No CSV files found for keyword '{keyword}'")

    print(f"Found {len(csv_files)} CSV files")

    # ---------- LOAD & AGGREGATE ----------
    df_list = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)

    # ---------- METRICS ----------
    metrics = [
        "train_accuracy", "train_precision", "train_recall", "train_f1",
        "test_accuracy", "test_precision", "test_recall", "test_f1"
    ]

    # ---------- BOXPLOT ----------
    plt.figure(figsize=(12, 6))
    df[metrics].boxplot(rot=45)
    plt.ylabel("Metric value")
    plt.title("Multiclass Classification Performance (Train vs Test)")

    # ---------- SAVE FIGURE ----------
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_name = f"results/boxplot_{keyword}_{timestamp}.png"

    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()

    print(f"Boxplot saved to {fig_name}")


if __name__ == "__main__":
    main(sys.argv[1:])
