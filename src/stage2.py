"""Stage-2 features: how a pair's stage-1 probability compares with its neighbours.

Two kinds of context are added to each (S1, candidate) pair:
  * S1 side  - rank / max / sum of stage-1 scores among that S1's candidates.
  * candidate side - the best stage-1 score the same S2/S3 record gets against ANY
    other S1 record. Each S2/S3 record belongs to at most one S1 entity, so a
    candidate that fits another S1 better is very likely a false merge.
  * support - similarity of the candidate to the S1's strongest other candidate
    (duplicates of one business look like each other).
"""
import numpy as np
from rapidfuzz import fuzz, process

from features import FEATURE_NAMES

RAW_KEEP = ["n_tset", "n_idf_cos", "a_tset", "a_idf_cos", "blk_rank", "blk_score_rel",
            "house_eq", "b_empty_addr"]
RAW_IDX = [FEATURE_NAMES.index(f) for f in RAW_KEEP]

S2_NAMES = (["p1", "p1_rank_s1", "p1_max_s1", "p1_rel_s1", "p1_gap_s1", "p1_sum_s1",
             "n_cand_s1", "n_strong_s1", "p1_other_s1_for_b", "p1_margin_b", "p1_rank_b",
             "n_s1_for_b", "sup_name", "sup_addr", "sup_p1"] + RAW_KEEP)


def _group_top2(keys, vals, n_groups):
    """Per group: max, second max, and each element's rank (0 = best)."""
    order = np.lexsort((-vals, keys))
    sk = keys[order]
    first = np.ones(len(order), bool)
    first[1:] = sk[1:] != sk[:-1]
    start_idx = np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))
    rank_sorted = np.arange(len(order)) - start_idx
    rank = np.empty(len(order), np.int32)
    rank[order] = rank_sorted
    top1 = np.full(n_groups, -1.0, np.float32)
    top2 = np.full(n_groups, -1.0, np.float32)
    arg1 = np.full(n_groups, -1, np.int64)
    sel1 = order[rank_sorted == 0]
    top1[keys[sel1]] = vals[sel1]
    arg1[keys[sel1]] = sel1
    sel2 = order[rank_sorted == 1]
    top2[keys[sel2]] = vals[sel2]
    return top1, top2, arg1, rank


def extract_raw(X, chunk=2_000_000):
    """Pull the few stage-1 columns stage 2 needs out of the on-disk feature matrix."""
    n = X.shape[0]
    out = np.empty((n, len(RAW_IDX)), np.float16)
    for a in range(0, n, chunk):
        out[a:a + chunk] = X[a:a + chunk][:, RAW_IDX]
    return out


class Stage2Context:
    """Group statistics over ALL pairs of a country (so every S1 competes)."""

    def __init__(self, cand, p1, n_s1, n_b):
        self.s1 = cand["s1"]
        self.b = cand["b"].astype(np.int64)
        self.p1 = p1 = p1.astype(np.float32)
        s1, b = self.s1, self.b
        self.top1_s, self.top2_s, self.arg1_s, self.rank_s = _group_top2(s1, p1, n_s1)
        self.top1_b, self.top2_b, _, self.rank_b = _group_top2(b, p1, n_b)
        self.sum_s = np.bincount(s1, weights=p1, minlength=n_s1).astype(np.float32)
        self.cnt_s = np.bincount(s1, minlength=n_s1).astype(np.float32)
        self.strong_s = np.bincount(s1, weights=(p1 > 0.5), minlength=n_s1).astype(np.float32)
        self.cnt_b = np.bincount(b, minlength=n_b).astype(np.float32)
        self.arg2_s = np.full(n_s1, -1, np.int64)
        sel2 = np.nonzero(self.rank_s == 1)[0]
        self.arg2_s[s1[sel2]] = sel2

    def features(self, idx, raw_small, b_names, b_addrs):
        s1, b, p1 = self.s1, self.b, self.p1
        si, bi, pp = s1[idx], b[idx], p1[idx]
        rs = self.rank_s[idx]
        other_b = np.where(self.rank_b[idx] == 0, self.top2_b[bi], self.top1_b[bi])
        gap = np.where(rs == 0, pp - np.maximum(self.top2_s[si], 0), pp - self.top1_s[si])

        ref = np.where(rs == 0, self.arg2_s[si], self.arg1_s[si])
        has_ref = ref >= 0
        ref_b = np.where(has_ref, b[np.maximum(ref, 0)], 0)
        sup_p1 = np.where(has_ref, p1[np.maximum(ref, 0)], 0).astype(np.float32)
        sup_name = np.zeros(len(idx), np.float32)
        sup_addr = np.zeros(len(idx), np.float32)
        if has_ref.any():
            h = np.nonzero(has_ref)[0]
            sup_name[h] = process.cpdist(b_names[bi[h]], b_names[ref_b[h]],
                                         scorer=fuzz.token_set_ratio, workers=-1,
                                         dtype=np.float32) / 100.0
            sup_addr[h] = process.cpdist(b_addrs[bi[h]], b_addrs[ref_b[h]],
                                         scorer=fuzz.token_set_ratio, workers=-1,
                                         dtype=np.float32) / 100.0
        cols = [
            pp, rs.astype(np.float32), self.top1_s[si],
            pp / np.maximum(self.top1_s[si], 1e-6), gap,
            self.sum_s[si], self.cnt_s[si], self.strong_s[si], np.maximum(other_b, 0),
            pp - np.maximum(other_b, 0), self.rank_b[idx].astype(np.float32),
            self.cnt_b[bi], sup_name, sup_addr, sup_p1,
        ]
        return np.column_stack(cols + [raw_small[idx].astype(np.float32)]).astype(np.float32)
