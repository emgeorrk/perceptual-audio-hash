# Perceptual Audio Hashing for IoT Sound Event Detection

**English** | [Русский](README.ru.md)

Reference implementation for a bachelor's thesis on compact perceptual audio
hashing. Instead of transmitting raw audio, an edge device computes a small
fixed-length hash from MFCC statistics and uses it to classify sound events.
This trades a modest drop in accuracy for a ~22 000× smaller representation,
which suits IoT nodes constrained in bandwidth, memory, and energy.

Experiments use the [ESC-50](https://github.com/karolpiczak/ESC-50)
environmental-sound dataset (2000 clips, 50 classes, 5 folds).

## Method

The pipeline that produces every number in the thesis:

1. **Load** each clip at 22 050 Hz mono and peak-normalize the waveform.
2. **Features** — 40 MFCC, summarized by per-coefficient **mean + std** over
   time → an **80-dimensional** vector.
3. **Normalize** — per-file z-normalization (zero mean, unit std across the 80
   dimensions).
4. **Classify**, two variants:
   - **float** — 1-NN with L2 distance over all training vectors
     (80 × `float64` = 640 bytes per reference).
   - **binary** — an **80-bit hash** = `sign` of the per-class mean prototype;
     classify by **Hamming distance** to the nearest prototype (10 bytes).

Train on folds 1–3, test on folds 4–5 (the standard ESC-50 split).

| Constant | Value |
|---|---|
| Sample rate | 22 050 Hz |
| FFT window / hop | 2048 / 512 |
| MFCC | 40 |
| Feature dim | 80 (mean + std) |
| Hash length | 80 bits = 10 bytes |

## Results

Accuracy on the test set (folds 4–5):

| Config | Method | Accuracy | k / n | Random baseline |
|---|---|---:|---:|---:|
| ESC-10 | float (1-NN, L2) | 56.2% | 90/160 | 10.0% |
| ESC-10 | binary (80-bit) | 43.8% | 70/160 | 10.0% |
| ESC-50 | float (1-NN, L2) | 29.1% | 233/800 | 2.0% |
| ESC-50 | binary (80-bit) | **10.4%** | 83/800 | 2.0% |

Binarization costs ~12 points on ESC-10 but ~19 points on ESC-50: a single
80-bit prototype per class stops separating the 50 spectrally similar classes,
yet the binary hash still beats random guessing by ~5× on the full set.

Per-class breakdowns for each run are written to `results/`.

### Subset accuracy

Expected float (1-NN, L2) accuracy on an arbitrary small subset of classes,
averaged over **every** subset of a given size (same train/test split). Run with
`python3 experiment.py --subsets`:

| Subset size | Combinations | Mean accuracy | Random baseline | × baseline |
|---|---:|---:|---:|---:|
| 2 classes | 1225 | 88.2% | 50.0% | 1.76× |
| 3 classes | 19 600 | 80.4% | 33.3% | 2.41× |

Spread over pairs: min 50.0% (least separable, e.g. `clock_tick` vs `cat`),
median 90.6%, and 176/1225 pairs reach 100%. Per-subset accuracies are written
to `results/subsets_pairs.csv` and `results/subsets_triples.csv`.

## Quickstart

The repository ships a precomputed feature cache (`features_cache.npz`) and the
labels file, so the experiment runs **without downloading the 1.7 GB audio**:

```bash
pip install -r requirements.txt
python3 experiment.py                 # ESC-10 and ESC-50, float and binary
```

Other options:

```bash
python3 experiment.py --config esc50  # only the full 50-class set
python3 experiment.py --method binary # only the binary hash
python3 experiment.py --subsets       # + averaged 2- and 3-class accuracy
python3 experiment.py --no-cache      # re-extract features (needs the audio)
```

To re-extract features from the raw audio, download the dataset first:

```bash
bash scripts/download_esc50.sh        # fetches ESC-50/audio (~600 MB zip)
python3 experiment.py --no-cache
```

## Repository layout

```
experiment.py              Full pipeline + evaluation (single entry point)
features_cache.npz         Precomputed 80-dim features for all 2000 clips
ESC-50/meta/esc50.csv      Labels and fold assignments (audio is not committed)
scripts/download_esc50.sh  Fetch the raw audio (no git submodule)
results/                   Per-class and per-subset accuracy CSVs, regenerated on each run
requirements.txt
LICENSE                    MIT (code); ESC-50 is CC BY-NC 3.0, see LICENSE
```

## License and attribution

Code is released under the MIT License. The ESC-50 dataset is the work of
K. J. Piczak and is distributed under CC BY-NC 3.0; see `LICENSE`.

> K. J. Piczak. *ESC: Dataset for Environmental Sound Classification.*
> Proceedings of the 23rd ACM International Conference on Multimedia, 2015.
