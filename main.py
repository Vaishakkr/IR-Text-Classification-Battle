"""
Text Classification Battle - Decision Tree vs KNN vs SVM vs Naive Bayes
Information Retrieval (21CSE377T) - CLA-2, Task 2
Author : K R Vaishak (RA2411026050018), III CSE AIML A

Dataset : BBC News (Greene & Cunningham, ICML 2006) - 2,225 articles, 5 classes
Run     : python main.py
Outputs : results/*.csv, results/*.txt and figures/*.png
"""

import os
import re
import time
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ------------------------------------------------------------------
# Configuration (kept in one place so every model sees the same setup)
# ------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.20
DATA_FILE = os.path.join("data", "bbc-text.csv")
RESULTS_DIR = "results"
FIGURES_DIR = "figures"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

stemmer = PorterStemmer()
STOP_WORDS = set(ENGLISH_STOP_WORDS)  # built into scikit-learn, no download needed


# ------------------------------------------------------------------
# Step 1 : Preprocessing
# ------------------------------------------------------------------
def preprocess(text):
    """lowercase -> keep alphabetic tokens -> drop stop words / 1-2 letter
    tokens -> Porter stemming. Returns a single space-joined string."""
    text = text.lower()
    tokens = re.findall(r"[a-z]+", text)
    kept = [stemmer.stem(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(kept)


def build_vectorizer(max_features=None):
    """The ONE feature representation shared by all four classifiers."""
    return TfidfVectorizer(
        min_df=2,          # ignore terms that appear in only one training document
        max_df=0.95,       # ignore terms present in >95% of documents
        sublinear_tf=True, # tf -> 1 + log(tf), dampens very frequent words
        max_features=max_features,
    )


def build_models():
    """Four classifiers. Hyper-parameters are standard defaults so that the
    comparison is about the algorithm, not about tuning effort."""
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, metric="cosine"),
        "SVM (Linear)": LinearSVC(C=1.0, random_state=SEED),
        "Naive Bayes": MultinomialNB(alpha=1.0),
    }


def evaluate(model, X_tr, y_tr, X_te, y_te):
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_te)
    pred_time = time.perf_counter() - t0

    p, r, f, _ = precision_recall_fscore_support(y_te, y_pred, average="macro", zero_division=0)
    return y_pred, {
        "Train accuracy": accuracy_score(y_tr, model.predict(X_tr)),
        "Accuracy": accuracy_score(y_te, y_pred),
        "Precision (macro)": p,
        "Recall (macro)": r,
        "F1 (macro)": f,
        "Train time (s)": train_time,
        "Predict time (s)": pred_time,
    }


def main():
    log = []

    def say(msg=""):
        print(msg)
        log.append(str(msg))

    say("=" * 70)
    say("TEXT CLASSIFICATION BATTLE : DT vs KNN vs SVM vs NB  (BBC News)")
    say("=" * 70)

    # --------------------------------------------------------------
    # Step 0 : Load and clean dataset
    # --------------------------------------------------------------
    df = pd.read_csv(DATA_FILE)
    say(f"Raw documents loaded          : {len(df)}")
    dup = df.duplicated(subset="text").sum()
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    say(f"Exact duplicate articles removed : {dup}")
    say(f"Documents used                : {len(df)}")

    dist = df["category"].value_counts().rename_axis("Class").reset_index(name="Documents")
    dist["Percent"] = (100 * dist["Documents"] / len(df)).round(2)
    dist.to_csv(os.path.join(RESULTS_DIR, "class_distribution.csv"), index=False)
    imbalance = dist["Documents"].max() / dist["Documents"].min()
    say("\nClass distribution:")
    say(dist.to_string(index=False))
    say(f"Imbalance ratio (largest / smallest) = {imbalance:.2f}")

    df["raw_words"] = df["text"].str.split().str.len()

    # --------------------------------------------------------------
    # Step 1 : Preprocess
    # --------------------------------------------------------------
    df["clean"] = df["text"].apply(preprocess)
    df["clean_words"] = df["clean"].str.split().str.len()
    say(f"\nAvg words / doc before preprocessing : {df['raw_words'].mean():.1f}")
    say(f"Avg tokens / doc after preprocessing : {df['clean_words'].mean():.1f}")
    sample = df.iloc[0]
    with open(os.path.join(RESULTS_DIR, "preprocessing_example.txt"), "w") as fh:
        fh.write(f"CLASS : {sample['category']}\n\nBEFORE (first 60 words):\n")
        fh.write(" ".join(sample["text"].split()[:60]) + "\n\nAFTER (first 40 tokens):\n")
        fh.write(" ".join(sample["clean"].split()[:40]) + "\n")

    # --------------------------------------------------------------
    # Step 2 : Same stratified split for every model
    # --------------------------------------------------------------
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        df["clean"], df["category"], test_size=TEST_SIZE, stratify=df["category"], random_state=SEED
    )
    say(f"\nTrain docs: {len(X_train_txt)}  |  Test docs: {len(X_test_txt)}  (stratified 80/20, seed={SEED})")

    # --------------------------------------------------------------
    # Step 3 : TF-IDF (fit on TRAIN only -> no test leakage)
    # --------------------------------------------------------------
    vec = build_vectorizer()
    X_train = vec.fit_transform(X_train_txt)
    X_test = vec.transform(X_test_txt)
    n_feat = X_train.shape[1]
    sparsity = 100 * (1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1]))
    say(f"TF-IDF vocabulary size: {n_feat}  |  matrix sparsity: {sparsity:.2f}%")

    # --------------------------------------------------------------
    # Step 4 : Train + evaluate the four classifiers
    # --------------------------------------------------------------
    labels = sorted(df["category"].unique())
    rows, preds = [], {}
    say("\n" + "-" * 70)
    for name, model in build_models().items():
        y_pred, m = evaluate(model, X_train, y_train, X_test, y_test)
        preds[name] = y_pred
        rows.append({"Model": name, **m})
        say(f"\n### {name}")
        say(classification_report(y_test, y_pred, digits=4, zero_division=0))

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
    say("-" * 70)
    say("COMPARISON (test set, macro averages)")
    say(results.round(4).to_string(index=False))

    # per-class F1 table
    per_class = {}
    for name, y_pred in preds.items():
        _, _, f, _ = precision_recall_fscore_support(y_test, y_pred, labels=labels, zero_division=0)
        per_class[name] = f
    per_class = pd.DataFrame(per_class, index=labels)
    per_class.to_csv(os.path.join(RESULTS_DIR, "per_class_f1.csv"))
    say("\nPer-class F1:")
    say(per_class.round(4).to_string())

    # --------------------------------------------------------------
    # Step 5 : 5-fold cross-validation (robustness check, train set only)
    # --------------------------------------------------------------
    say("\n5-fold stratified CV on the training set (macro F1):")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_rows = []
    for name, model in build_models().items():
        pipe = make_pipeline(build_vectorizer(), model)  # vectorizer refit inside each fold
        scores = cross_val_score(pipe, X_train_txt, y_train, cv=skf, scoring="f1_macro")
        cv_rows.append({"Model": name, "CV F1 mean": scores.mean(), "CV F1 std": scores.std()})
        say(f"  {name:<14}: {scores.mean():.4f} +/- {scores.std():.4f}")
    pd.DataFrame(cv_rows).to_csv(os.path.join(RESULTS_DIR, "cross_validation.csv"), index=False)

    # --------------------------------------------------------------
    # Step 6 : Figures
    # --------------------------------------------------------------
    metrics = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    x = np.arange(len(results))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, mtr in enumerate(metrics):
        bars = ax.bar(x + (i - 1.5) * w, results[mtr], w, label=mtr, color=colors[i])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004, f"{b.get_height():.3f}",
                    ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(results["Model"])
    ax.set_ylim(0.75, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("BBC News - Test-set performance of the four classifiers")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "01_model_comparison.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (name, y_pred) in zip(axes.ravel(), preds.items()):
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False,
                                                               xticks_rotation=35)
        acc = np.trace(cm) / cm.sum()
        ax.set_title(f"{name}  (acc = {acc:.4f})")
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(
            os.path.join(RESULTS_DIR, f"confusion_{name.split()[0].lower()}.csv"))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "02_confusion_matrices.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    per_class.plot(kind="bar", ax=ax, color=colors, width=0.8)
    ax.set_ylim(0.6, 1.02)
    ax.set_ylabel("F1-score")
    ax.set_title("Per-class F1-score")
    ax.set_xticklabels(labels, rotation=0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "03_per_class_f1.png"), dpi=150)
    plt.close(fig)

    # --------------------------------------------------------------
    # Step 7 : Best model, top terms, misclassified examples
    # --------------------------------------------------------------
    best = results.sort_values(["F1 (macro)", "Accuracy"], ascending=False).iloc[0]["Model"]
    say(f"\nBest model by macro F1 on the test set: {best}")

    terms = np.array(vec.get_feature_names_out())
    svm = LinearSVC(C=1.0, random_state=SEED).fit(X_train, y_train)
    nb = MultinomialNB(alpha=1.0).fit(X_train, y_train)
    dt = DecisionTreeClassifier(random_state=SEED).fit(X_train, y_train)
    with open(os.path.join(RESULTS_DIR, "top_terms.txt"), "w") as fh:
        fh.write("Top 10 SVM weights per class\n")
        for i, c in enumerate(svm.classes_):
            fh.write(f"  {c:<14}: {', '.join(terms[np.argsort(svm.coef_[i])[::-1][:10]])}\n")
        fh.write("\nTop 10 Naive Bayes log-probabilities per class\n")
        for i, c in enumerate(nb.classes_):
            fh.write(f"  {c:<14}: {', '.join(terms[np.argsort(nb.feature_log_prob_[i])[::-1][:10]])}\n")
        fh.write("\nDecision Tree\n")
        fh.write(f"  depth = {dt.get_depth()}, leaves = {dt.get_n_leaves()}, "
                 f"features actually used = {(dt.feature_importances_ > 0).sum()} of {n_feat}\n")
        fh.write(f"  top-10 split features: {', '.join(terms[np.argsort(dt.feature_importances_)[::-1][:10]])}\n")

    test_df = pd.DataFrame({"text": df.loc[X_test_txt.index, "text"].values,
                            "true": y_test.values, "pred": preds[best]})
    wrong = test_df[test_df.true != test_df.pred]
    wrong.assign(snippet=wrong.text.str[:200]).drop(columns="text").to_csv(
        os.path.join(RESULTS_DIR, "misclassified_best_model.csv"), index=False)
    say(f"Misclassified test documents for {best}: {len(wrong)} of {len(test_df)}")

    # --------------------------------------------------------------
    # Step 8 : WHAT-IF EXPERIMENTS
    # --------------------------------------------------------------
    # (A) main what-if : shrink the TF-IDF vocabulary for ALL four models
    say("\n" + "=" * 70)
    say("WHAT-IF (A): How does vocabulary size (max_features) affect each model?")
    sizes = [50, 100, 250, 500, 1000, 2500, 5000, None]
    wi_rows = []
    for mf in sizes:
        v = build_vectorizer(max_features=mf)
        Xtr, Xte = v.fit_transform(X_train_txt), v.transform(X_test_txt)
        for name, model in build_models().items():
            _, m = evaluate(model, Xtr, y_train, Xte, y_test)
            wi_rows.append({"max_features": Xtr.shape[1] if mf is None else mf,
                            "Model": name, "Accuracy": m["Accuracy"], "F1 (macro)": m["F1 (macro)"]})
    wi = pd.DataFrame(wi_rows)
    wi.to_csv(os.path.join(RESULTS_DIR, "whatif_vocab_size.csv"), index=False)
    say(wi.pivot(index="max_features", columns="Model", values="F1 (macro)").round(4).to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    for (name, g), c in zip(wi.groupby("Model", sort=False), colors):
        ax.plot(g["max_features"], g["F1 (macro)"], marker="o", label=name, color=c)
    ax.set_xscale("log")
    ax.set_xlabel("TF-IDF vocabulary size (max_features, log scale)")
    ax.set_ylabel("Macro F1 (test)")
    ax.set_title("What-if A: effect of vocabulary size")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "04_whatif_vocab_size.png"), dpi=150)
    plt.close(fig)

    # (B) supporting what-if : KNN neighbours and distance metric
    say("\nWHAT-IF (B): KNN - number of neighbours k and distance metric")
    k_rows = []
    for metric in ["cosine", "euclidean"]:
        for k in [1, 3, 5, 7, 9, 15, 25, 51]:
            _, m = evaluate(KNeighborsClassifier(n_neighbors=k, metric=metric), X_train, y_train, X_test, y_test)
            k_rows.append({"metric": metric, "k": k, "Accuracy": m["Accuracy"], "F1 (macro)": m["F1 (macro)"]})
    kdf = pd.DataFrame(k_rows)
    kdf.to_csv(os.path.join(RESULTS_DIR, "whatif_knn_k.csv"), index=False)
    say(kdf.pivot(index="k", columns="metric", values="F1 (macro)").round(4).to_string())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (metric, g), c in zip(kdf.groupby("metric"), ["#55A868", "#8172B2"]):
        ax.plot(g["k"], g["F1 (macro)"], marker="o", label=f"{metric} distance", color=c)
    ax.set_xlabel("k (number of neighbours)")
    ax.set_ylabel("Macro F1 (test)")
    ax.set_title("What-if B: KNN neighbours and distance metric")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "05_whatif_knn_k.png"), dpi=150)
    plt.close(fig)

    summary = {"documents": int(len(df)), "duplicates_removed": int(dup), "train": int(len(X_train_txt)),
               "test": int(len(X_test_txt)), "vocabulary": int(n_feat), "best_model": best}
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(RESULTS_DIR, "run_log.txt"), "w") as fh:
        fh.write("\n".join(log) + "\n")
    say("\nDone. Tables in results/, plots in figures/.")


if __name__ == "__main__":
    main()
