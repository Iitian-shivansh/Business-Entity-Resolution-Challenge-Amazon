"""LightGBM helpers shared by training and prediction."""
import os

import lightgbm as lgb
import numpy as np

from data import load_country, log
from features import FEATURE_NAMES
from stage2 import S2_NAMES, Stage2Context, extract_raw

# Absolute blocking score depends on how many records a country has, so it does not
# transfer to an unseen country. Only its relative form and rank are used.
STAGE1_EXCLUDE = {"blk_score"}
STAGE1_COLS = [i for i, f in enumerate(FEATURE_NAMES) if f not in STAGE1_EXCLUDE]
STAGE1_NAMES = [FEATURE_NAMES[i] for i in STAGE1_COLS]

STAGE1_PARAMS = dict(
    objective="binary", learning_rate=0.06, num_leaves=255, min_data_in_leaf=150,
    feature_fraction=0.85, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    max_bin=255, verbose=-1,
)
STAGE1_ROUNDS = 800
STAGE2_PARAMS = dict(
    objective="binary", learning_rate=0.04, num_leaves=127, min_data_in_leaf=150,
    feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    verbose=-1,
)
STAGE2_ROUNDS = 600


def stage1_matrix(X, rows=None):
    if rows is None:
        return np.asarray(X[:, STAGE1_COLS], np.float32)
    return np.asarray(X[rows][:, STAGE1_COLS], np.float32)


def predict_stage1(models, X, chunk=2_000_000, n_jobs=4):
    n = X.shape[0]
    out = np.zeros(n, np.float32)
    for a in range(0, n, chunk):
        m = np.asarray(X[a:a + chunk][:, STAGE1_COLS], np.float32)
        out[a:a + chunk] = np.mean(
            [mdl.predict(m, num_threads=n_jobs) for mdl in models], axis=0)
    return out


def load_b_text(cache_dir, split, country):
    """Candidate-side name/address arrays (S2 then S3, same order as featurize)."""
    parts = [load_country(cache_dir, split, k, country) for k in (2, 3)]
    names = np.concatenate([p["name_core"].to_numpy(dtype=object) for p in parts])
    addrs = np.concatenate([p["addr"].to_numpy(dtype=object) for p in parts])
    return names, addrs


def stage2_predict_all(model2, cand, p1, X, n_s1, n_b, b_names, b_addrs,
                       chunk=2_000_000, n_jobs=4):
    ctx = Stage2Context(cand, p1, n_s1, n_b)
    raw = extract_raw(X)
    n = len(p1)
    p2 = np.zeros(n, np.float32)
    for a in range(0, n, chunk):
        idx = np.arange(a, min(a + chunk, n))
        p2[idx] = model2.predict(ctx.features(idx, raw, b_names, b_addrs), num_threads=n_jobs)
    return p2, ctx, raw


def save_models(model_dir, stage1_models, stage2_model):
    os.makedirs(model_dir, exist_ok=True)
    for k, m in enumerate(stage1_models):
        m.save_model(os.path.join(model_dir, f"stage1_fold{k}.txt"))
    stage2_model.save_model(os.path.join(model_dir, "stage2.txt"))


def load_models(model_dir):
    s1 = []
    k = 0
    while os.path.exists(os.path.join(model_dir, f"stage1_fold{k}.txt")):
        s1.append(lgb.Booster(model_file=os.path.join(model_dir, f"stage1_fold{k}.txt")))
        k += 1
    if not s1:
        raise FileNotFoundError(f"no stage-1 models in {model_dir}; run train.py first")
    s2 = lgb.Booster(model_file=os.path.join(model_dir, "stage2.txt"))
    log(f"loaded {len(s1)} stage-1 models + stage-2 model from {model_dir}")
    return s1, s2
