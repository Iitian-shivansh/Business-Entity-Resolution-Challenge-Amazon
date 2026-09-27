# Business Entity Resolution — ML Challenge 2026

Two-stage LightGBM matcher with inverted-index blocking. No pretrained models, no
external data, no network access at run time. Every dependency is BSD/MIT/Apache/ISC.

## 1. Setup

```bash
pip install -r requirements.txt
```

Put the data here (paths are configurable with `--data-dir`):

```
dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
dataset/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

## 2. Run

From this folder:

```bash
python src/train.py   --data-dir dataset --work-dir work --n-jobs 4
python src/predict.py --data-dir dataset --work-dir work --out-dir output --n-jobs 4
```

`train.py` writes the models and the tuned decision rule to `work/models/`.
`predict.py` writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
Both steps are resumable: normalised data and per-country features are cached under
`work/`, and a finished country is skipped on a re-run. Delete `work/` (or the
relevant country folder) after changing blocking or feature code.

Validate, then eyeball the unlabelled country:

```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
python src/inspect_matches.py --data-dir dataset --out-dir output --country France
```

## 3. Resources (full data, ~1.7M test S1 records, ~10M S2/S3 records)

| | estimate |
|---|---|
| RAM | 10–14 GB peak |
| Disk for `--work-dir` | ~15 GB (features stored as float16) |
| Time, 4 cores | train ~1–1.5 h, predict ~30–45 min |

Every log line shows the peak memory so far (`[peak X GB]`). If memory is tight:

```bash
--work-budget 10000000 --feature-chunk 400000 --train-s1 150000 --top-k 25
```

On Kaggle, point `--work-dir` at `/tmp/work` (more disk than `/kaggle/working`) and
`--out-dir` at `/kaggle/working/output`.

## 4. Reading the training log

* `blocking recall by rank` — share of true matches inside the top-k candidates.
  If it is still rising at the last value, increase `--top-k`.
* `HOLD-OUT macro F0.5 (report half)` — honest score on Source 1 records the models
  never saw, with the threshold tuned on a different half.
* `transfer check [country]` — the model retrained without that country. It
  approximates how the pipeline behaves on France, which has no labels.

## 5. Options that matter

| flag | default | meaning |
|---|---|---|
| `--top-k` | 30 | candidates per S1 record kept by blocking |
| `--df-cap` | 1000 | blocking keys shared by more records are ignored |
| `--train-s1` / `--holdout-s1` | 300000 / 120000 | S1 records used to fit / to tune and score |
| `--unseen-policy` | `rate_match` | for countries not in training (France): raise the threshold until matches per S1 do not exceed the training hold-out rate; `same` disables it |
| `--unseen-strict-house` | 0 | 1 = in unseen countries, never merge when both house numbers exist and differ |

## 6. Code map

| file | role |
|---|---|
| `src/normalize.py` | transliteration, abbreviation and legal-form canonicalisation, typo-tolerant keys |
| `src/data.py` | chunked TSV reading, parallel normalisation, parquet cache, per-country loading |
| `src/blocking.py` | hashed blocking keys, IDF-weighted sparse top-k candidate generation |
| `src/features.py` | ~50 pairwise features (rapidfuzz, IDF overlap, rare-token overlap, numbers) |
| `src/featurize.py` | per-country blocking + features written to disk |
| `src/stage2.py` | competition / group features built from stage-1 scores |
| `src/decode.py` | exclusive assignment, thresholds, macro F0.5 |
| `src/models.py` | LightGBM settings and helpers |
| `src/train.py` | training, hold-out tuning, transfer check |
| `src/predict.py` | test inference, writes both output files |
| `src/inspect_matches.py` | prints predicted matches for manual review |
