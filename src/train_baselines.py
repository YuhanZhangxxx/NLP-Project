#!/usr/bin/env python
import argparse, json, pathlib, datetime, csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib
import matplotlib.pyplot as plt

def load_splits():
    def read_ids(p): 
        return [x for x in pathlib.Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]
    splits = {
        "train": set(read_ids("data/processed/train.txt")),
        "valid": set(read_ids("data/processed/valid.txt")),
        "test":  set(read_ids("data/processed/test.txt")),
    }
    return splits

def plot_cm(cm, classes, title, out_path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    plt.xlabel("Predicted"); plt.ylabel("True")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def evaluate(y_true, y_pred, labels_order):
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels_order)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels_order, zero_division=0)
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": {lbl: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(support[i])}
                      for i, lbl in enumerate(labels_order)}
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_csv", required=True, help="raw CSV with id,text,label")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--max_df", type=float, default=0.9)
    ap.add_argument("--ngram_max", type=int, default=2)
    ap.add_argument("--nb_alpha", type=float, default=1.0)
    ap.add_argument("--lr_C", type=float, default=1.0)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    assert {"id","text","label"}.issubset(df.columns)

    splits = load_splits()
    mask_train = df["id"].isin(splits["train"])
    mask_valid = df["id"].isin(splits["valid"])
    mask_test  = df["id"].isin(splits["test"])

    df_train = df[mask_train].copy()
    df_valid = df[mask_valid].copy()
    df_test  = df[mask_test].copy()

    labels_order = ["bad","good"]  # fixed ordering

    # TF-IDF vectorizer
    tfidf = TfidfVectorizer(lowercase=True, ngram_range=(1, args.ngram_max),
                            min_df=args.min_df, max_df=args.max_df)

    # Pipelines
    pipe_nb = Pipeline([("tfidf", tfidf),
                        ("clf", MultinomialNB(alpha=args.nb_alpha))])

    pipe_lr = Pipeline([("tfidf", tfidf),
                        ("clf", LogisticRegression(C=args.lr_C, max_iter=200, n_jobs=None, class_weight=None))])

    # Fit on train
    pipe_nb.fit(df_train["text"], df_train["label"])
    pipe_lr.fit(df_train["text"], df_train["label"])

    # Evaluate on valid
    preds_nb_val = pipe_nb.predict(df_valid["text"])
    preds_lr_val = pipe_lr.predict(df_valid["text"])
    m_nb_val = evaluate(df_valid["label"], preds_nb_val, labels_order)
    m_lr_val = evaluate(df_valid["label"], preds_lr_val, labels_order)

    # choose best by macro-f1
    best_model_name, best_pipe, best_valid = ("nb", pipe_nb, m_nb_val) if m_nb_val["macro_f1"] >= m_lr_val["macro_f1"] else ("lr", pipe_lr, m_lr_val)

    # Test once
    preds_nb_test = pipe_nb.predict(df_test["text"])
    preds_lr_test = pipe_lr.predict(df_test["text"])
    m_nb_test = evaluate(df_test["label"], preds_nb_test, labels_order)
    m_lr_test = evaluate(df_test["label"], preds_lr_test, labels_order)

    # Reports
    reports_dir = pathlib.Path("reports")
    reports_dir.mkdir(exist_ok=True, parents=True)

    # metrics.csv
    import csv
    with open(reports_dir/"metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model","split","accuracy","macro_f1"])
        w.writerow(["nb","valid", m_nb_val["accuracy"], m_nb_val["macro_f1"]])
        w.writerow(["lr","valid", m_lr_val["accuracy"], m_lr_val["macro_f1"]])
        w.writerow(["nb","test", m_nb_test["accuracy"], m_nb_test["macro_f1"]])
        w.writerow(["lr","test", m_lr_test["accuracy"], m_lr_test["macro_f1"]])

    # confusion matrices (test)
    for name, preds in [("nb", preds_nb_test), ("lr", preds_lr_test)]:
        cm = confusion_matrix(df_test["label"], preds, labels=labels_order)
        plot_cm(cm, labels_order, f"Confusion Matrix (test) – {name.upper()}", reports_dir/f"confusion_matrix_{name}.png")

    # run_config.json
    cfg = {
        "random_seed": args.seed,
        "tfidf": {"ngram": [1, args.ngram_max], "min_df": args.min_df, "max_df": args.max_df},
        "nb": {"alpha": args.nb_alpha},
        "lr": {"C": args.lr_C, "max_iter": 200},
        "splits": {k: f"data/processed/{k}.txt" for k in ["train","valid","test"]},
        "labels_order": labels_order
    }
    (reports_dir/"run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # Save models
    models_dir = pathlib.Path("models")
    models_dir.mkdir(exist_ok=True, parents=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d")
    nb_path = models_dir/f"song_nb_v1_{stamp}.joblib"
    lr_path = models_dir/f"song_lr_v1_{stamp}.joblib"
    joblib.dump(pipe_nb, nb_path)
    joblib.dump(pipe_lr, lr_path)

    # manifest.json
    manifest = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "train_size": int(len(df_train)),
        "valid_size": int(len(df_valid)),
        "test_size": int(len(df_test)),
        "models": {
            "nb": {"path": str(nb_path), "valid": m_nb_val, "test": m_nb_test},
            "lr": {"path": str(lr_path), "valid": m_lr_val, "test": m_lr_test},
        },
        "best_on_valid": best_model_name
    }
    (models_dir/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Training complete. See reports/metrics.csv and models/manifest.json")

if __name__ == "__main__":
    main()
