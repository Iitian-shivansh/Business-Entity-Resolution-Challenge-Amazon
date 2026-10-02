"""Candidate generation: IDF-weighted inverted-index blocking on hashed keys.

Each record is turned into a bag of blocking keys (typo-tolerant name tokens,
order-free name-token pairs, a joined-name prefix for web-handle names, address
words, ZIP/PIN codes, and house-number x street-word combinations). Keys that are too
common (document frequency above `df_cap`) are dropped, so the remaining keys are
rare and the sparse product S1 x (S2+S3)^T stays cheap. For every Source 1 record we
keep the top-K Source 2/3 records by summed IDF of shared keys.

The work is chunked by an estimate of the product size, so memory stays flat no
matter how large the country is.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

from data import log
from normalize import ADDR_GENERIC, skeleton

N_FEATURES = 2 ** 24


def blocking_keys(doc):
    core, skel, addr = doc
    nt = [t for t in skel.split() if len(t) >= 2][:6]
    keys = ["n" + t for t in nt]
    ctoks = core.split()
    joined = "".join(ctoks)
    if len(joined) >= 6:
        keys.append("j" + joined[:9])
        keys.append("J" + joined)
    # concatenations of adjacent words, and long single words, so that
    # "pediatric dentistry" meets "pediatricdentistry.com"
    for i in range(min(len(ctoks), 5) - 1):
        keys.append("c" + ctoks[i] + ctoks[i + 1])
    for t in ctoks[:5]:
        if len(t) >= 10:
            keys.append("c" + t)
    first4 = nt[:4]
    for i in range(len(first4)):
        for j in range(i + 1, len(first4)):
            a, b = sorted((first4[i], first4[j]))
            keys.append("p" + a + "_" + b)
    toks = addr.split()
    words, nums, seen = [], [], set()
    for t in toks:
        if t.isdigit():
            if t not in seen and t != "0":
                nums.append(t)
                seen.add(t)
        elif len(t) >= 3 and t not in ADDR_GENERIC:
            w = skeleton(t)
            if w not in seen:
                words.append(w)
                seen.add(w)
    keys += ["a" + w for w in words[:8]]
    keys += ["z" + n for n in nums if len(n) >= 5]
    for n in nums[:2]:
        for w in words[:6]:
            keys.append("x" + n + "_" + w)
    if nums:
        for t in nt[:2]:
            keys.append("y" + t + "_" + nums[0])
    # Fuzzy keys built from 4-letter prefixes of the phonetic skeleton. They survive
    # transliteration drift ("infrastructure" / "inphrastrkcr" -> "infr") and are made
    # rare by pairing them with a second name prefix or with an address number.
    pre = [t[:4] for t in nt[:3] if len(t) >= 3]
    for i in range(len(pre)):
        for j in range(i + 1, len(pre)):
            a, b = sorted((pre[i], pre[j]))
            keys.append("P" + a + "_" + b)
    for n in nums[:3]:
        for q in pre:
            keys.append("q" + n + "_" + q)
    return keys


def _vectorizer():
    return HashingVectorizer(
        analyzer=blocking_keys, n_features=N_FEATURES, alternate_sign=False,
        norm=None, binary=True, dtype=np.float32,
    )


def key_matrix_chunks(recs, chunk=500_000):
    """Hashed key matrix built in row chunks (keeps the peak memory of hashing low)."""
    hv = _vectorizer()
    parts = []
    for a in range(0, recs.n, chunk):
        z = min(a + chunk, recs.n)
        m = hv.transform(zip(recs.name_core[a:z], recs.name_skel[a:z], recs.addr[a:z])).tocsr()
        m.sum_duplicates()
        m.data[:] = 1.0
        parts.append(m)
    return parts


def _col_counts(parts):
    df = np.zeros(N_FEATURES, dtype=np.int64)
    for m in parts:
        df += np.bincount(m.indices, minlength=N_FEATURES)
    return df


def _prune_stack(parts, keep):
    for m in parts:
        m.data *= keep[m.indices]
        m.eliminate_zeros()
    out = sp.vstack(parts, format="csr") if len(parts) > 1 else parts[0]
    parts.clear()
    return out


def _topk_from_product(M, k, row_offset):
    """Top-k columns per row of CSR matrix M (vectorised)."""
    counts = np.diff(M.indptr)
    if M.nnz == 0:
        e = np.empty(0, dtype=np.int64)
        return e, e.astype(np.int32), e.astype(np.float32), e.astype(np.int16)
    rows = np.repeat(np.arange(M.shape[0], dtype=np.int64), counts)
    order = np.lexsort((-M.data, rows))
    rank = np.arange(M.nnz, dtype=np.int64) - M.indptr[rows[order]]
    keep = rank < k
    sel = order[keep]
    return (rows[sel] + row_offset, M.indices[sel].astype(np.int32),
            M.data[sel].astype(np.float32), rank[keep].astype(np.int16))


def generate_candidates(s1, b, top_k=30, df_cap=1000, min_score=0.0, work_budget=40_000_000):
    """Return dict of arrays: s1 (int64), b (int32), score, rank."""
    log(f"  blocking: {s1.n:,} S1 vs {b.n:,} S2/S3 records")
    A_parts = key_matrix_chunks(s1)
    B_parts = key_matrix_chunks(b)
    df = _col_counts(A_parts) + _col_counts(B_parts)
    n_docs = s1.n + b.n
    keep = (df > 0) & (df <= df_cap)
    idf = np.zeros(N_FEATURES, dtype=np.float32)
    idf[keep] = np.log((n_docs + 1.0) / (df[keep] + 1.0)).astype(np.float32)

    # drop over-common keys (this is also what keeps memory low)
    keep_f = keep.astype(np.float32)
    B = _prune_stack(B_parts, keep_f)
    BT = B.T.tocsr()
    del B
    A = _prune_stack(A_parts, keep_f)
    A.data = idf[A.indices]
    log(f"  blocking keys: S1 nnz={A.nnz:,}  S2/S3 nnz={BT.nnz:,}  kept keys={int(keep.sum()):,}")

    # estimated product size per S1 row = sum of df over its kept keys
    df_kept = np.where(keep, df, 0).astype(np.float64)
    Abin = A.copy()
    Abin.data[:] = 1.0
    work = Abin @ df_kept
    del Abin

    out = {"s1": [], "b": [], "score": [], "rank": []}
    start = 0
    cum = np.cumsum(work)
    n_chunks = 0
    while start < s1.n:
        base = cum[start - 1] if start > 0 else 0.0
        end = int(np.searchsorted(cum, base + work_budget, side="right"))
        end = min(max(end, start + 1), s1.n)
        M = A[start:end] @ BT
        if min_score > 0:
            M.data[M.data < min_score] = 0
            M.eliminate_zeros()
        r, c, d, rk = _topk_from_product(M, top_k, start)
        out["s1"].append(r)
        out["b"].append(c)
        out["score"].append(d)
        out["rank"].append(rk)
        start = end
        n_chunks += 1
    res = {k: np.concatenate(v) for k, v in out.items()}
    top = np.zeros(s1.n, dtype=np.float32)
    np.maximum.at(top, res["s1"], res["score"])
    res["top"] = top[res["s1"]]
    log(f"  blocking done: {len(res['s1']):,} candidate pairs in {n_chunks} chunks "
        f"({len(res['s1']) / max(1, s1.n):.1f} per S1)")
    return res
