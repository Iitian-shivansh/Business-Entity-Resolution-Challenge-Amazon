"""Blocking + pairwise features for every country of a split, written to disk.

Output per country (in work_dir/<split>/<country>/):
    s1_ids.npy, b_ids.npy   entity ids, indexed by the integer ids below
    cand.npz                s1 (int64), b (int32), score, rank, top  -- sorted by s1
    X.f16                   float16 memmap, n_pairs x N_FEAT
    meta.json               sizes
"""
import json
import os
import time

import numpy as np

from blocking import generate_candidates
from config import country_dir
from data import list_countries, load_country_records, log
from features import N_FEAT, TokenIDF, compute_features


def featurize_country(args, split, country):
    d = country_dir(args.work_dir, split, country)
    meta_path = os.path.join(d, "meta.json")
    if os.path.exists(meta_path):
        log(f"[{split}/{country}] features already on disk, skipping")
        return json.load(open(meta_path))
    t0 = time.time()
    s1, b = load_country_records(args.work_dir + "/cache", split, country)
    np.save(os.path.join(d, "s1_ids.npy"), s1.entity_id.astype(str))
    np.save(os.path.join(d, "b_ids.npy"), b.entity_id.astype(str))
    if s1.n == 0 or b.n == 0:
        cand = {k: np.empty(0, dt) for k, dt in
                (("s1", np.int64), ("b", np.int32), ("score", np.float32),
                 ("rank", np.int16), ("top", np.float32))}
    else:
        cand = generate_candidates(s1, b, top_k=args.top_k, df_cap=args.df_cap,
                                   work_budget=args.work_budget)
    np.savez(os.path.join(d, "cand.npz"), **cand)
    n_pairs = len(cand["s1"])
    X = np.lib.format.open_memmap(os.path.join(d, "X.npy"), mode="w+",
                                  dtype=np.float16, shape=(max(n_pairs, 1), N_FEAT))
    if n_pairs:
        idf = TokenIDF(s1, b)
        step = args.feature_chunk
        for start in range(0, n_pairs, step):
            sl = slice(start, min(start + step, n_pairs))
            X[sl] = compute_features(s1, b, cand, idf, sl).astype(np.float16)
            log(f"  [{split}/{country}] features {sl.stop:,}/{n_pairs:,}")
        del idf
    X.flush()
    del X
    meta = dict(country=country, n_s1=int(s1.n), n_b=int(b.n), n_pairs=int(n_pairs))
    json.dump(meta, open(meta_path, "w"))
    log(f"[{split}/{country}] done in {time.time() - t0:.0f}s: {meta}")
    return meta


def featurize_split(args, split):
    countries = list_countries(args.work_dir + "/cache", split)
    log(f"{split}: countries = {countries}")
    return {c: featurize_country(args, split, c) for c in countries}


def load_country_arrays(args, split, country):
    d = country_dir(args.work_dir, split, country)
    cand = dict(np.load(os.path.join(d, "cand.npz")))
    X = np.load(os.path.join(d, "X.npy"), mmap_mode="r")
    s1_ids = np.load(os.path.join(d, "s1_ids.npy"))
    b_ids = np.load(os.path.join(d, "b_ids.npy"))
    return cand, X, s1_ids, b_ids
