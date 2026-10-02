# Business Entity Resolution, ML Challenge 2026

This folder regenerates `matching_results.tsv` and `candidate_pairs.tsv` from the challenge data. The pipeline is: normalise, block, score with two LightGBM models, decide. It uses no pretrained models, no external data and no network access while running. Every dependency has a BSD, MIT, Apache-2.0 or ISC licence.

## 1. Put the data in place

```
dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
dataset/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

`--data-dir` points at the folder that contains `train/` and `test/`.

## 2. Install

```bash
pip install -r requirements.txt
```

## 3. Run

From this folder:

```bash
python src/train.py   --data-dir dataset --work-dir work --n-jobs 4
python src/predict.py --data-dir dataset --work-dir work --out-dir output --n-jobs 4
```

`train.py` saves both models and the tuned decision rule in `work/models/`, plus `diagnostics.txt` with real examples of blocking misses, wrong merges and missed matches. `predict.py` writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`. The candidate file is exactly the set of pairs the model scored.

Both scripts cache their work under `work/`. If a run stops, start it again and it picks up where it left off. If you change blocking or feature code, delete `work/` first, otherwise the old cache gets reused.

Then validate:

```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

France has no labels, so it is worth eyeballing a sample of its matches:

```bash
python src/inspect_matches.py --data-dir dataset --out-dir output --country France --n 15
```

## 4. How our submitted files were produced (Kaggle)

The full data does not fit on a laptop, so our submitted outputs came from a Kaggle CPU notebook (4 cores, about 30 GB RAM, internet on only for `pip`). The data and this code were added as two private Kaggle datasets. The cells were:

```python
DATA = "/kaggle/input/<data-dataset>/dataset"
CODE = "/kaggle/input/<code-dataset>/code/business_entity_resolution"
```

```python
!rm -rf /kaggle/working/ber
!cp -r {CODE} /kaggle/working/ber
!pip install -q anyascii "rapidfuzz>=3.6" lightgbm
```

```python
!rm -rf /tmp/work
%cd /kaggle/working/ber
!python src/train.py --data-dir {DATA} --work-dir /tmp/work --n-jobs 4 2>&1 | tee /kaggle/working/train_log.txt
```

```python
!python src/predict.py --data-dir {DATA} --work-dir /tmp/work --out-dir /kaggle/working/output --n-jobs 4 2>&1 | tee /kaggle/working/predict_log.txt
```

`/tmp` is used for the work directory because it has more free disk than `/kaggle/working`. Kaggle already ships numpy, pandas, scipy, scikit-learn and pyarrow, so only the three missing libraries are installed there. The whole run takes about 6 to 7 hours, so we ran it with Save & Run All (Commit).

## 5. Resources

| | Full data |
|---|---|
| RAM | 10 to 14 GB at peak |
| Disk for `--work-dir` | about 15 GB |
| Time on 4 cores | about 6 to 7 hours for train + predict |

Every log line prints the peak memory so far. If memory runs short, add these to both commands:

```bash
--work-budget 10000000 --feature-chunk 400000 --train-s1 150000
```

## 6. Settings used for the submission

All defaults, nothing overridden on the command line:

| Flag | Value | Meaning |
|---|---|---|
| `--top-k` | 40 | candidates kept per Source 1 record |
| `--df-cap` | 3000 | blocking keys shared by more records than this are ignored |
| `--train-s1` / `--holdout-s1` | 300000 / 120000 | Source 1 records used to fit the models / to tune and score the decision rule |
| `--unseen-policy` | `rate_match` | for countries not seen in training (France), raise the threshold if matches per record exceed the true training rate; never lower it |
| `--unseen-strict-house` | 0 | 1 = in unseen countries, never merge when both house numbers exist and differ |

## 7. Reading the training log

- `blocking recall by rank`: share of true matches inside the top-k candidates. If it is still rising at the last rank, a bigger `--top-k` will help.
- `HOLD-OUT macro F0.5`: score on Source 1 records the models never trained on, using a threshold tuned on a different half of the hold-out. Our final run printed 0.9566.
- `transfer check`: the model retrained with one country left out, as a rough guide to how it behaves on France.

## 8. Code map

| File | Role |
|---|---|
| `src/config.py` | command-line flags and defaults |
| `src/normalize.py` | transliteration, accent folding, digit-for-letter fixes, legal-form and abbreviation canonicalisation |
| `src/data.py` | chunked TSV reading, parallel normalisation, parquet cache |
| `src/blocking.py` | hashed blocking keys and IDF-weighted sparse top-k candidate search |
| `src/features.py` | about 50 pairwise features |
| `src/featurize.py` | per-country blocking and features, written to disk |
| `src/stage2.py` | group and competition features from stage-1 scores |
| `src/models.py` | LightGBM settings and helpers |
| `src/decode.py` | thresholds, exclusive assignment, macro F0.5 |
| `src/train.py` | training, hold-out tuning, transfer check, diagnostics |
| `src/predict.py` | test inference; writes both output files |
| `src/inspect_matches.py` | prints predicted matches side by side for review |
