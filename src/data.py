"""Reading the TSV sources, normalising them once, and caching to parquet.

The raw files are ~400-500 MB each. They are read in chunks, normalised in parallel
and written to one parquet file per source. Every later stage reads the parquet cache
one country at a time, so peak memory is bounded by the largest country, not by the
whole file.
"""
import csv
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from normalize import normalize_address, normalize_name, skeleton

READ_KW = dict(
    sep="\t", dtype=str, keep_default_na=False, na_filter=False,
    quoting=csv.QUOTE_NONE, engine="c", encoding="utf-8", encoding_errors="replace",
)
COLS = ["entity_id", "business_name", "business_address", "country"]

FLAG_NONASCII_NAME = 1
FLAG_WEB_NAME = 2
FLAG_EMPTY_ADDR = 4


try:
    import resource

    def _peak_gb():
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / (1024 ** 3) if os.uname().sysname == "Darwin" else kb / (1024 ** 2)
except ImportError:                      # Windows
    def _peak_gb():
        return float("nan")


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + f"[peak {_peak_gb():.1f} GB] " + msg, flush=True)


def _norm_chunk(args):
    names, addrs = args
    out_full, out_core, out_skel, out_addr, out_nums, out_flags = [], [], [], [], [], []
    for n, a in zip(names, addrs):
        full, core = normalize_name(n)
        ad, nums = normalize_address(a)
        flags = 0
        if not n.isascii():
            flags |= FLAG_NONASCII_NAME
        low = n.lower()
        if "www" in low or ".com" in low or low.startswith("#") or ".in" in low or ".fr" in low:
            flags |= FLAG_WEB_NAME
        if not ad:
            flags |= FLAG_EMPTY_ADDR
        out_full.append(full)
        out_core.append(core)
        out_skel.append(" ".join(skeleton(t) for t in core.split()))
        out_addr.append(ad)
        out_nums.append(nums)
        out_flags.append(flags)
    return out_full, out_core, out_skel, out_addr, out_nums, out_flags


def source_path(data_dir, split, k):
    return os.path.join(data_dir, split, f"{split}_source{k}.tsv")


def cache_path(cache_dir, split, k):
    return os.path.join(cache_dir, f"{split}_source{k}.parquet")


def prepare_source(data_dir, cache_dir, split, k, n_jobs=4, chunk_rows=400_000):
    """Normalise one source file into a parquet cache (skipped if it already exists)."""
    out = cache_path(cache_dir, split, k)
    if os.path.exists(out):
        log(f"cache exists: {out}")
        return out
    os.makedirs(cache_dir, exist_ok=True)
    src = source_path(data_dir, split, k)
    log(f"normalising {src}")
    tmp = out + ".tmp"
    writer = None
    total = 0
    reader = pd.read_csv(src, usecols=COLS, chunksize=chunk_rows, **READ_KW)
    with Pool(n_jobs) as pool:
        for df in reader:
            df["country"] = df["country"].str.strip()
            names = df["business_name"].tolist()
            addrs = df["business_address"].tolist()
            step = max(1, len(names) // (n_jobs * 4) + 1)
            parts = pool.map(
                _norm_chunk,
                [(names[i:i + step], addrs[i:i + step]) for i in range(0, len(names), step)],
            )
            cols = list(zip(*parts))
            flat = [sum(c, []) for c in cols]
            table = pa.table({
                "entity_id": pa.array(df["entity_id"].str.strip().tolist(), pa.string()),
                "country": pa.array(df["country"].tolist(), pa.string()),
                "name_full": pa.array(flat[0], pa.string()),
                "name_core": pa.array(flat[1], pa.string()),
                "name_skel": pa.array(flat[2], pa.string()),
                "addr": pa.array(flat[3], pa.string()),
                "nums": pa.array(flat[4], pa.string()),
                "flags": pa.array(np.asarray(flat[5], dtype=np.int8)),
            })
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
            total += len(df)
            log(f"  {total:,} rows")
    if writer is not None:
        writer.close()
    os.replace(tmp, out)
    return out


def prepare_split(data_dir, cache_dir, split, n_jobs=4):
    for k in (1, 2, 3):
        prepare_source(data_dir, cache_dir, split, k, n_jobs=n_jobs)


def list_countries(cache_dir, split):
    """Every country label present in any source of the split (open set)."""
    labels = set()
    for k in (1, 2, 3):
        t = pq.read_table(cache_path(cache_dir, split, k), columns=["country"])
        labels |= set(t.column("country").unique().to_pylist())
    return sorted(labels)


def load_country(cache_dir, split, k, country):
    t = pq.read_table(cache_path(cache_dir, split, k), filters=[("country", "==", country)])
    return t.to_pandas()


class Records:
    """Column-oriented view of normalised records (numpy object arrays)."""

    FIELDS = ("entity_id", "name_full", "name_core", "name_skel", "addr", "nums")

    def __init__(self, df):
        for f in self.FIELDS:
            setattr(self, f, df[f].to_numpy(dtype=object))
        self.flags = df["flags"].to_numpy(dtype=np.int8)
        self.n = len(df)


def load_country_records(cache_dir, split, country):
    """Return (S1 records, S2+S3 records) for one country."""
    s1 = load_country(cache_dir, split, 1, country)
    b = pd.concat(
        [load_country(cache_dir, split, 2, country), load_country(cache_dir, split, 3, country)],
        ignore_index=True,
    )
    return Records(s1), Records(b)


def read_ground_truth(data_dir):
    path = os.path.join(data_dir, "train", "train_ground_truth.tsv")
    gt = pd.read_csv(path, **READ_KW)
    gt.columns = [c.strip() for c in gt.columns]
    return dict(zip(gt["source1_entity_id"].str.strip(), gt["matched_entity_ids"].str.strip()))
