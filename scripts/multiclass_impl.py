#!/usr/bin/env python

import sys
import getopt
import os
from datetime import datetime

import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from cpe487587hw import deepl


def main(argv):

    
    data_file = "data/Android_Malware.csv"
    epochs = 100
    lr = 0.001
    keyword = "hw02"

    
    try:
        opts, args = getopt.getopt(
            argv,
            "hf:e:l:k:",
            ["data_file=", "epochs=", "lr=", "keyword="]
        )
    except getopt.GetoptError:
        print(f"Usage: {__file__} -f <data_file> -e <epochs> -l <lr> -k <keyword>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print(f"\n{__file__} [OPTIONS]")
            print("\t-h, --help            Show help")
            print("\t-f, --data_file       Path to CSV file")
            print("\t-e, --epochs          Number of epochs")
            print("\t-l, --lr              Learning rate")
            print("\t-k, --keyword         Unique keyword for CSV")
            sys.exit()
        elif opt in ("-f", "--data_file"):
            data_file = arg
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("-l", "--lr"):
            lr = float(arg)
        elif opt in ("-k", "--keyword"):
            keyword = arg

    print("OPTIONS:")
    print(f"  data_file = {data_file}")
    print(f"  epochs    = {epochs}")
    print(f"  lr        = {lr}")
    print(f"  keyword   = {keyword}")

    # ---------- LOAD DATA ----------
    df = pd.read_csv(data_file)
    df.columns = df.columns.str.strip()
    drop_cols = [
        "Flow ID", "Source IP", "Source Port",
        "Destination IP", "Destination Port",
        "Protocol", "Timestamp"
    ]
    df = df.drop(columns=drop_cols)

    X = df.drop(columns=["Label"])
    y = df["Label"]

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median())

    # ---------- PREPROCESS ----------
    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # ---------- MODEL + TRAINER ----------
    num_classes = len(np.unique(y_train))

    model = deepl.SimpleNN(
        in_features=X_train.shape[1], #coloumns features!
        num_classes=num_classes
    )

    trainer = deepl.ClassTrainer(
        X_train=X_train,
        Y_train=y_train,
        model=model,
        eta=lr,
        epochs=epochs
    )

    # ---------- TRAIN ----------
    trainer.train()

    # ---------- EVALUATE ----------
    metrics = trainer.evaluation(X_test, y_test)

    # ---------- SAVE METRICS ----------
    os.makedirs("results", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"results/metrics_{keyword}_{timestamp}.csv"
    pd.DataFrame([metrics]).to_csv(csv_name, index=False)
    print(f"\nSaved metrics to {csv_name}")
    print("Training complete.")


if __name__ == "__main__":
    main(sys.argv[1:])
