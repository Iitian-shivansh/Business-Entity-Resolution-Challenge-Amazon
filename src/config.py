"""Default settings. Every value can be overridden from the command line."""
import argparse
import os
import re

DEFAULTS = dict(
    data_dir="dataset",          # contains train/ and test/
    work_dir="work",             # caches, features, models (needs ~10 GB free disk)
    out_dir="output",
    n_jobs=4,                    # processes for normalisation / threads for LightGBM
    # blocking
    top_k=40,                    # candidates kept per Source 1 record
    df_cap=3000,                 # blocking keys shared by more records than this are ignored
    work_budget=30_000_000,      # max estimated sparse-product size per chunk (memory knob)
    feature_chunk=1_000_000,     # pairs per feature chunk (memory knob)
    # training
    train_s1=300_000,            # Source 1 records used to fit the models
    holdout_s1=120_000,          # Source 1 records held out for tuning + honest scoring
    folds=3,
    transfer_check=1,            # 1 = also measure leave-one-country-out generalisation
    seed=42,
    # prediction
    unseen_policy="rate_match",  # for countries absent from training: rate_match | same
    unseen_strict_house=0,       # 1 = in unseen countries, never merge on conflicting house numbers
)


def get_args(description):
    p = argparse.ArgumentParser(description=description)
    for k, v in DEFAULTS.items():
        p.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    return p.parse_args()


def slug(label):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", label) or "unknown"


def country_dir(work_dir, split, country):
    d = os.path.join(work_dir, split, slug(country))
    os.makedirs(d, exist_ok=True)
    return d
