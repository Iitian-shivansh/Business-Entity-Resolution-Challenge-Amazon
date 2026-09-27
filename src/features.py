"""Pairwise similarity features for (Source 1, candidate) pairs.

All string scorers come from rapidfuzz.process.cpdist, which runs element-wise in
C++ across all cores. Token-weight features use per-country IDF computed from the
data itself, so nothing depends on the country label (France works the same way as
the training countries).
"""
import numpy as np
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import HashingVectorizer
import jellyfish

from data import FLAG_EMPTY_ADDR, FLAG_NONASCII_NAME, FLAG_WEB_NAME

N_FEATURES = 2 ** 22

FEATURE_NAMES = [
    # blocking
    "blk_score", "blk_rank", "blk_score_rel",
    # names
    "n_ratio", "n_tsort", "n_tset", "n_partial", "n_jw", "n_phonetic_jacc", "n_acronym",
    "n_core_ratio", "n_core_tset", "n_join_ratio", "n_join_partial", "n_skel_ratio",
    "n_skel_tset", "n_idf_cos", "n_idf_cov1", "n_idf_cov2", "n_len1", "n_len2",
    "n_first_eq",
    # addresses
    "a_ratio", "a_tsort", "a_tset", "a_partial", "a_idf_cos", "a_idf_cov1", "a_idf_cov2",
    "a_len1", "a_len2", "num_jacc", "num_cov1", "house_eq", "zip_eq",
    # flags
    "b_is_s3", "b_nonascii", "b_web", "b_empty_addr", "s1_empty_addr",
    # "specific" tokens: rarer than SPEC_FRAC of the country's records. Common words
    # (city names, "club", "services") drop out, so these say whether the *distinctive*
    # part of a name/address agrees. Relative to country size, so they transfer.
    "n_spec_n1", "n_spec_n2", "n_spec_inter", "n_spec_jacc", "n_spec_cov1",
    "a_spec_n1", "a_spec_n2", "a_spec_inter", "a_spec_jacc", "a_spec_cov1",
    "n_info1", "a_info1",
    # cross / ratio features
    "n_len_ratio", "a_len_ratio", "na_diff", "num_count_diff",
]
SPEC_FRAC = 0.001
N_FEAT = len(FEATURE_NAMES)


def _split_tokens(s):
    return s.split()


class TokenIDF:
    """IDF-weighted, L2-normalised token vectors for names and addresses."""

    def __init__(self, s1, b):
        hv = HashingVectorizer(analyzer=_split_tokens, n_features=N_FEATURES,
                               alternate_sign=False, norm=None, binary=True,
                               dtype=np.float32)
        self.name1 = hv.transform(s1.name_core).tocsr()
        self.name2 = hv.transform(b.name_core).tocsr()
        self.addr1 = hv.transform(s1.addr).tocsr()
        self.addr2 = hv.transform(b.addr).tocsr()
        n = s1.n + b.n
        for a1, a2 in (("name1", "name2"), ("addr1", "addr2")):
            m1, m2 = getattr(self, a1), getattr(self, a2)
            df = (np.bincount(m1.indices, minlength=N_FEATURES)
                  + np.bincount(m2.indices, minlength=N_FEATURES))
            spec = (df > 0) & (df <= max(2.0, SPEC_FRAC * n))
            for src, dst in ((a1, a1 + "_spec"), (a2, a2 + "_spec")):
                m = getattr(self, src).copy()
                m.data = spec[m.indices].astype(np.float32)
                m.eliminate_zeros()
                setattr(self, dst, m)
            idf = np.log((n + 1.0) / (df + 1.0)).astype(np.float32) + 1.0
            for m in (m1, m2):
                m.data = idf[m.indices]
            # informativeness of the S1 text: mean idf of its tokens relative to the
            # maximum possible idf in this country (scale-free)
            info = _row_sums(m1) / np.maximum(np.diff(m1.indptr), 1) / np.log(n + 1.0)
            setattr(self, a1 + "_info", info.astype(np.float32))
        self.name1_norm = _row_norms(self.name1)
        self.name2_norm = _row_norms(self.name2)
        self.addr1_norm = _row_norms(self.addr1)
        self.addr2_norm = _row_norms(self.addr2)
        self.name1_sum = _row_sums(self.name1)
        self.name2_sum = _row_sums(self.name2)
        self.addr1_sum = _row_sums(self.addr1)
        self.addr2_sum = _row_sums(self.addr2)


def _row_norms(m):
    return np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel()).astype(np.float32)


def _row_sums(m):
    return np.asarray(m.sum(axis=1)).ravel().astype(np.float32)


def _pair_dot(m1, m2, i, j):
    """Row-wise dot product m1[i[k]] . m2[j[k]] for all k (weights squared-free overlap)."""
    a = m1[i]
    b = m2[j]
    b.data[:] = 1.0                      # overlap weight counted once, from the S1 side
    return np.asarray(a.multiply(b).sum(axis=1)).ravel().astype(np.float32)


def _pair_cos(m1, m2, i, j, n1, n2):
    a = m1[i]
    b = m2[j]
    dot = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    den = n1[i] * n2[j]
    return np.where(den > 0, dot / np.maximum(den, 1e-9), 0).astype(np.float32)


def _cp(a, b, scorer, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw) / 100.0


def _num_features(n1, n2):
    jac = np.zeros(len(n1), np.float32)
    cov = np.zeros(len(n1), np.float32)
    for k, (x, y) in enumerate(zip(n1, n2)):
        if not x or not y:
            jac[k] = -1.0
            cov[k] = -1.0
            continue
        sx, sy = set(x.split()), set(y.split())
        inter = len(sx & sy)
        jac[k] = inter / len(sx | sy)
        cov[k] = inter / len(sx)
    return jac, cov


def _first_num(addr):
    for t in addr.split():
        if t.isdigit():
            return t
    return ""


def _long_num(nums):
    best = ""
    for t in nums.split():
        if len(t) in (5, 6) and len(t) > len(best):
            best = t
    return best


def _phonetic_jaccard(n1, n2):
    """Jaccard similarity of the metaphone codes of the first two tokens."""
    out = np.zeros(len(n1), np.float32)
    for k, (x, y) in enumerate(zip(n1, n2)):
        if not x or not y:
            continue
        cx = {jellyfish.metaphone(t) for t in x.split()[:2] if len(t) >= 2}
        cy = {jellyfish.metaphone(t) for t in y.split()[:2] if len(t) >= 2}
        cx.discard("")
        cy.discard("")
        if cx and cy:
            inter = len(cx & cy)
            out[k] = inter / len(cx | cy)
    return out


def _acronym_eq(n1, n2):
    """Check if the initials of one name exactly match the other name."""
    out = np.zeros(len(n1), np.float32)
    for k, (x, y) in enumerate(zip(n1, n2)):
        if not x or not y:
            continue
        
        x_init = "".join([w[0] for w in x.split() if w])
        y_init = "".join([w[0] for w in y.split() if w])
        
        x_clean = x.replace(" ", "")
        y_clean = y.replace(" ", "")
        
        if (len(x_init) >= 2 and x_init == y_clean) or (len(y_init) >= 2 and y_init == x_clean):
            out[k] = 1.0
    return out


def _eq_or_missing(x, y):
    out = np.zeros(len(x), np.float32)
    for k, (a, c) in enumerate(zip(x, y)):
        if a and c:
            out[k] = 1.0 if a == c else -1.0
    return out


def compute_features(s1, b, cand, idf, pair_slice=None):
    """Feature matrix (float32, n_pairs x N_FEAT) for candidate pairs in `pair_slice`."""
    sl = pair_slice if pair_slice is not None else slice(None)
    i = cand["s1"][sl]
    j = cand["b"][sl]
    n = len(i)
    X = np.zeros((n, N_FEAT), dtype=np.float32)
    col = {name: k for k, name in enumerate(FEATURE_NAMES)}

    X[:, col["blk_score"]] = cand["score"][sl]
    X[:, col["blk_rank"]] = cand["rank"][sl]
    X[:, col["blk_score_rel"]] = cand["score"][sl] / np.maximum(cand["top"][sl], 1e-6)

    nf1, nf2 = s1.name_full[i], b.name_full[j]
    nc1, nc2 = s1.name_core[i], b.name_core[j]
    ns1, ns2 = s1.name_skel[i], b.name_skel[j]
    X[:, col["n_ratio"]] = _cp(nf1, nf2, fuzz.ratio)
    X[:, col["n_tsort"]] = _cp(nf1, nf2, fuzz.token_sort_ratio)
    X[:, col["n_tset"]] = _cp(nf1, nf2, fuzz.token_set_ratio)
    X[:, col["n_partial"]] = _cp(nc1, nc2, fuzz.partial_ratio)
    X[:, col["n_jw"]] = process.cpdist(nc1, nc2, scorer=JaroWinkler.similarity,
                                       workers=-1, dtype=np.float32)
    X[:, col["n_phonetic_jacc"]] = _phonetic_jaccard(nc1, nc2)
    X[:, col["n_acronym"]] = _acronym_eq(nc1, nc2)
    X[:, col["n_core_ratio"]] = _cp(nc1, nc2, fuzz.ratio)
    X[:, col["n_core_tset"]] = _cp(nc1, nc2, fuzz.token_set_ratio)
    j1 = np.array([x.replace(" ", "") for x in nc1], dtype=object)
    j2 = np.array([x.replace(" ", "") for x in nc2], dtype=object)
    X[:, col["n_join_ratio"]] = _cp(j1, j2, fuzz.ratio)
    X[:, col["n_join_partial"]] = _cp(j1, j2, fuzz.partial_ratio)
    X[:, col["n_skel_ratio"]] = _cp(ns1, ns2, fuzz.token_sort_ratio)
    X[:, col["n_skel_tset"]] = _cp(ns1, ns2, fuzz.token_set_ratio)
    X[:, col["n_idf_cos"]] = _pair_cos(idf.name1, idf.name2, i, j, idf.name1_norm, idf.name2_norm)
    ov = _pair_dot(idf.name1, idf.name2, i, j)
    X[:, col["n_idf_cov1"]] = ov / np.maximum(idf.name1_sum[i], 1e-6)
    ov2 = _pair_dot(idf.name2, idf.name1, j, i)
    X[:, col["n_idf_cov2"]] = ov2 / np.maximum(idf.name2_sum[j], 1e-6)
    X[:, col["n_len1"]] = [len(x) for x in nc1]
    X[:, col["n_len2"]] = [len(x) for x in nc2]
    X[:, col["n_first_eq"]] = [
        1.0 if (x and y and x.split()[0] == y.split()[0]) else 0.0 for x, y in zip(ns1, ns2)
    ]

    a1, a2 = s1.addr[i], b.addr[j]
    X[:, col["a_ratio"]] = _cp(a1, a2, fuzz.ratio)
    X[:, col["a_tsort"]] = _cp(a1, a2, fuzz.token_sort_ratio)
    X[:, col["a_tset"]] = _cp(a1, a2, fuzz.token_set_ratio)
    X[:, col["a_partial"]] = _cp(a1, a2, fuzz.partial_ratio)
    X[:, col["a_idf_cos"]] = _pair_cos(idf.addr1, idf.addr2, i, j, idf.addr1_norm, idf.addr2_norm)
    ov = _pair_dot(idf.addr1, idf.addr2, i, j)
    X[:, col["a_idf_cov1"]] = ov / np.maximum(idf.addr1_sum[i], 1e-6)
    ov2 = _pair_dot(idf.addr2, idf.addr1, j, i)
    X[:, col["a_idf_cov2"]] = ov2 / np.maximum(idf.addr2_sum[j], 1e-6)
    X[:, col["a_len1"]] = [len(x) for x in a1]
    X[:, col["a_len2"]] = [len(x) for x in a2]
    nu1, nu2 = s1.nums[i], b.nums[j]
    jac, cov = _num_features(nu1, nu2)
    X[:, col["num_jacc"]] = jac
    X[:, col["num_cov1"]] = cov
    X[:, col["house_eq"]] = _eq_or_missing([_first_num(x) for x in a1], [_first_num(x) for x in a2])
    X[:, col["zip_eq"]] = _eq_or_missing([_long_num(x) for x in nu1], [_long_num(x) for x in nu2])

    bid = b.entity_id[j]
    X[:, col["b_is_s3"]] = [1.0 if e.startswith("S3") else 0.0 for e in bid]
    fb = b.flags[j]
    X[:, col["b_nonascii"]] = (fb & FLAG_NONASCII_NAME) > 0
    X[:, col["b_web"]] = (fb & FLAG_WEB_NAME) > 0
    X[:, col["b_empty_addr"]] = (fb & FLAG_EMPTY_ADDR) > 0
    X[:, col["s1_empty_addr"]] = (s1.flags[i] & FLAG_EMPTY_ADDR) > 0

    for pre, m1, m2 in (("n", idf.name1_spec, idf.name2_spec),
                        ("a", idf.addr1_spec, idf.addr2_spec)):
        c1 = np.diff(m1.indptr)[i].astype(np.float32)
        c2 = np.diff(m2.indptr)[j].astype(np.float32)
        inter = np.asarray(m1[i].multiply(m2[j]).sum(axis=1)).ravel().astype(np.float32)
        X[:, col[pre + "_spec_n1"]] = c1
        X[:, col[pre + "_spec_n2"]] = c2
        X[:, col[pre + "_spec_inter"]] = inter
        X[:, col[pre + "_spec_jacc"]] = np.where(c1 + c2 - inter > 0,
                                                 inter / np.maximum(c1 + c2 - inter, 1), -1)
        X[:, col[pre + "_spec_cov1"]] = np.where(c1 > 0, inter / np.maximum(c1, 1), -1)
    X[:, col["n_info1"]] = idf.name1_info[i]
    X[:, col["a_info1"]] = idf.addr1_info[i]

    # --- new cross / ratio features (near-zero compute cost) ---
    n_l1 = np.array([max(len(x), 1) for x in nc1], np.float32)
    n_l2 = np.array([max(len(x), 1) for x in nc2], np.float32)
    X[:, col["n_len_ratio"]] = np.minimum(n_l1, n_l2) / np.maximum(n_l1, n_l2)

    a_l1 = np.array([max(len(x), 1) for x in a1], np.float32)
    a_l2 = np.array([max(len(x), 1) for x in a2], np.float32)
    X[:, col["a_len_ratio"]] = np.minimum(a_l1, a_l2) / np.maximum(a_l1, a_l2)

    X[:, col["na_diff"]] = X[:, col["n_tset"]] - X[:, col["a_tset"]]

    nc1_count = np.array([len(x.split()) if x else 0 for x in nu1], np.float32)
    nc2_count = np.array([len(x.split()) if x else 0 for x in nu2], np.float32)
    X[:, col["num_count_diff"]] = np.abs(nc1_count - nc2_count)

    return X
