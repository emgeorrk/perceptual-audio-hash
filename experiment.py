"""Perceptual audio hashing for IoT sound event detection — ESC-50 experiment.

The idea: instead of transmitting raw audio, a device computes a compact
perceptual hash from MFCC statistics and uses it for sound-event classification.

Pipeline
--------
1. Load WAV at 22050 Hz mono, peak-normalize the waveform.
2. Compute 40 MFCC, take mean + std over time  ->  80-dim feature vector.
3. Per-file z-normalization (zero mean, unit std across the 80 dimensions).
4. Two classifiers:
     - float  : 1-NN with L2 distance over all training vectors.
     - binary : 80-bit hash = sign(class-mean prototype); nearest prototype
                by Hamming distance.
Train = folds 1-3, test = folds 4-5 (standard ESC-50 split).

Run
---
    python3 experiment.py                # ESC-10 and ESC-50, both methods
    python3 experiment.py --config esc50 # only the full set
    python3 experiment.py --subsets      # + averaged 2- and 3-class accuracy
    python3 experiment.py --no-cache     # re-extract features from audio

Requires the ESC-50 dataset in ./ESC-50 (see scripts/download_esc50.sh).
A precomputed feature cache (features_cache.npz) ships with the repo, so the
experiment runs without downloading the 1.7 GB dataset.
"""
import argparse
import itertools
import math
import os

import numpy as np
import pandas as pd

DATASET_DIR = "ESC-50/audio"
META_PATH = "ESC-50/meta/esc50.csv"
CACHE_PATH = "features_cache.npz"

SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 40
FEAT_DIM = 2 * N_MFCC  # mean + std  ->  80

TRAIN_FOLDS = (1, 2, 3)
TEST_FOLDS = (4, 5)


# --------------------------------------------------------------------------- #
# Feature extraction                                                          #
# --------------------------------------------------------------------------- #
def load_audio(path):
    import librosa

    signal, _ = librosa.load(path, sr=SR, mono=True)
    peak = np.max(np.abs(signal))
    return signal / peak if peak > 0 else signal


def compute_features(signal):
    """40 MFCC -> per-coefficient mean and std -> 80-dim vector."""
    import librosa

    mfcc = librosa.feature.mfcc(
        y=signal, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def build_or_load_cache(meta, use_cache=True):
    if use_cache and os.path.exists(CACHE_PATH):
        z = np.load(CACHE_PATH, allow_pickle=True)
        feats = z["feats"]
        if (z["filenames"] == meta["filename"].values).all():
            print(f"Loaded feature cache: {CACHE_PATH}  {feats.shape}")
            return feats
        print("Cache does not match metadata order, recomputing...")

    feats = np.empty((len(meta), FEAT_DIM), dtype=np.float64)
    for i, fn in enumerate(meta["filename"].values):
        feats[i] = compute_features(load_audio(os.path.join(DATASET_DIR, fn)))
        if (i + 1) % 200 == 0:
            print(f"  extracted {i + 1}/{len(meta)}")
    np.savez_compressed(CACHE_PATH, feats=feats, filenames=meta["filename"].values)
    print(f"Saved feature cache: {CACHE_PATH}")
    return feats


# --------------------------------------------------------------------------- #
# Normalization and classifiers                                               #
# --------------------------------------------------------------------------- #
def znorm(matrix):
    """Per-row z-normalization (zero mean, unit std across the 80 dims)."""
    m = np.atleast_2d(matrix).astype(float)
    return (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-8)


def binarize(vec):
    return (vec > 0).astype(np.uint8)


def classify_float(query, refs, labels):
    """1-NN with L2 distance."""
    return labels[np.argmin(np.linalg.norm(refs - query, axis=1))]


def classify_binary(query_hash, prototypes):
    """Nearest class prototype by Hamming distance."""
    return min(
        prototypes, key=lambda c: int(np.sum(query_hash != prototypes[c]))
    )


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #
def run(feats, meta, method):
    """Return (per-class DataFrame, correct, total) for one method."""
    idx = meta.index.values
    X = feats[idx]
    folds = meta["fold"].values
    cats = meta["category"].values

    tr = np.isin(folds, TRAIN_FOLDS)
    te = np.isin(folds, TEST_FOLDS)
    Xtr, ytr = znorm(X[tr]), cats[tr]
    Xte, yte = znorm(X[te]), cats[te]

    if method == "binary":
        prototypes = {
            c: binarize(Xtr[ytr == c].mean(axis=0)) for c in np.unique(ytr)
        }
        preds = [classify_binary(binarize(x), prototypes) for x in Xte]
    else:
        preds = [classify_float(x, Xtr, ytr) for x in Xte]

    res = pd.DataFrame({"true": yte, "correct": [int(p == y) for p, y in zip(preds, yte)]})
    per = (
        res.groupby("true")["correct"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "correct", "count": "total"})
    )
    per["accuracy"] = per["correct"] / per["total"]
    return per.sort_values("accuracy", ascending=False), int(res["correct"].sum()), len(res)


def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion."""
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


# --------------------------------------------------------------------------- #
# Subset analysis (averaged accuracy over class pairs / triples)              #
# --------------------------------------------------------------------------- #
def class_min_matrix(Xte, Xtr, ytr, classes):
    """M[q, k] = min L2 distance from test file q to any train file of class k.

    With this matrix, the float (1-NN, L2) prediction restricted to any subset
    of classes is just argmin over the subset's columns, so subset accuracy can
    be evaluated without recomputing distances per subset.
    """
    M = np.full((Xte.shape[0], len(classes)), np.inf)
    for k, c in enumerate(classes):
        refs = Xtr[ytr == c]
        d = np.linalg.norm(Xte[:, None, :] - refs[None, :, :], axis=2)
        M[:, k] = d.min(axis=1)
    return M


SUBSET_CSV = {2: "results/subsets_pairs.csv", 3: "results/subsets_triples.csv"}


def run_subsets(feats, meta, sizes=(2, 3)):
    """Average float (1-NN, L2) accuracy over all class subsets of each size.

    Same pipeline as the float method: per-file z-norm, train folds 1-3, test
    folds 4-5, nearest training vector restricted to the subset's classes. For
    every subset the accuracy over its test files is computed; we report the
    mean (expected accuracy on an arbitrary subset of that size) plus the spread
    (min, median, share reaching 100%). Per-subset accuracies go to results/.
    """
    X = znorm(feats[meta.index.values])
    folds = meta["fold"].values
    cats = meta["category"].values
    classes = np.unique(cats)
    cls_idx = {c: i for i, c in enumerate(classes)}

    tr = np.isin(folds, TRAIN_FOLDS)
    te = np.isin(folds, TEST_FOLDS)
    M = class_min_matrix(X[te], X[tr], cats[tr], classes)
    yte = np.array([cls_idx[c] for c in cats[te]])
    members = {k: np.where(yte == k)[0] for k in range(len(classes))}
    per_class_tr = int(tr.sum()) // len(classes)
    per_class_te = int(te.sum()) // len(classes)

    os.makedirs("results", exist_ok=True)
    print(f"\n{'config':<12}{'classes':>8}{'train/test':>12}"
          f"{'baseline':>10}{'accuracy':>10}{'x base':>9}")
    print("-" * 61)
    spread = []
    for size in sizes:
        combos, accs = [], []
        for combo in itertools.combinations(range(len(classes)), size):
            sel = np.concatenate([members[k] for k in combo])
            sub = M[sel][:, combo]
            true_col = np.array([combo.index(t) for t in yte[sel]])
            combos.append(tuple(classes[k] for k in combo))
            accs.append(float((sub.argmin(axis=1) == true_col).mean()))
        accs = np.array(accs)
        baseline = 1.0 / size
        mean = accs.mean()
        print(f"{f'{size} classes':<12}{size:>8}"
              f"{f'{size * per_class_tr}/{size * per_class_te}':>12}"
              f"{baseline * 100:>9.1f}%{mean * 100:>9.1f}%{mean / baseline:>8.2f}")
        worst_i = int(accs.argmin())
        n_min = int((accs == accs.min()).sum())
        n100 = int((accs == 1.0).sum())
        spread.append((size, len(accs), accs.min(), combos[worst_i], n_min,
                       float(np.median(accs)), accs.max(), n100))
        pd.DataFrame(
            {"classes": ["+".join(c) for c in combos], "accuracy": accs}
        ).to_csv(SUBSET_CSV[size], index=False)

    print("\nspread:")
    for size, n, mn, worst, n_min, med, mx, n100 in spread:
        name = {2: "pairs", 3: "triples"}.get(size, f"{size}-subsets")
        print(f"  {name:<8} n={n:<6} min {mn * 100:5.1f}% "
              f"({'+'.join(worst)}, {n_min} at min)  median {med * 100:5.1f}%  "
              f"max {mx * 100:5.1f}%  ==100%: {n100} ({n100 / n * 100:.1f}%)")


CONFIGS = {
    "esc10": ("ESC-10", lambda m: m[m["esc10"]].copy()),
    "esc50": ("ESC-50", lambda m: m.copy()),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", choices=["esc10", "esc50", "all"], default="all",
        help="Which subset to evaluate (default: all).",
    )
    parser.add_argument(
        "--method", choices=["float", "binary", "both"], default="both",
        help="Classifier to evaluate (default: both).",
    )
    parser.add_argument("--no-cache", action="store_true", help="Re-extract features from audio.")
    parser.add_argument(
        "--subsets", action="store_true",
        help="Also report averaged accuracy over all 2- and 3-class subsets.",
    )
    args = parser.parse_args()

    meta = pd.read_csv(META_PATH).reset_index(drop=True)
    feats = build_or_load_cache(meta, use_cache=not args.no_cache)

    configs = ["esc10", "esc50"] if args.config == "all" else [args.config]
    methods = ["float", "binary"] if args.method == "both" else [args.method]

    print(f"\n{'config':<10}{'method':<9}{'accuracy':>10}{'95% CI (Wilson)':>22}{'k / n':>12}")
    print("-" * 63)
    os.makedirs("results", exist_ok=True)
    for cfg in configs:
        label, select = CONFIGS[cfg]
        sub = select(meta)
        for method in methods:
            per, k, n = run(feats, sub, method)
            lo, hi = wilson_ci(k, n)
            print(
                f"{label:<10}{method:<9}{k / n * 100:>9.1f}%"
                f"{f'[{lo * 100:.1f}; {hi * 100:.1f}]':>22}{f'{k}/{n}':>12}"
            )
            per.to_csv(f"results/{cfg}_{method}.csv")

    if args.subsets:
        run_subsets(feats, meta)


if __name__ == "__main__":
    main()
