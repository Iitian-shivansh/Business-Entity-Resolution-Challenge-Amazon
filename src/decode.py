"""Turning pair probabilities into match lists, and scoring them with macro F0.5."""
import numpy as np


def exclusive_best(s1, b, p):
    """True for pairs where this S1 is the best-scoring S1 for the candidate record.

    Every S2/S3 record belongs to at most one Source 1 entity, so a record may only be
    given to the S1 that wants it most.
    """
    order = np.lexsort((-p, b))
    first = np.ones(len(order), bool)
    first[1:] = b[order][1:] != b[order][:-1]
    keep = np.zeros(len(p), bool)
    keep[order[first]] = True
    return keep


def decode(s1, b, p, threshold, exclusive=True, min_top=None, excl_mask=None):
    """Selected pair mask.

    threshold : minimum probability for a pair to be kept
    exclusive : a candidate record may only go to the S1 that scores it highest
    min_top   : if set, an S1 keeps matches only when its best pair clears this value
                (a stricter gate against false merges on singletons)
    excl_mask : precomputed exclusive_best(s1, b, p), to avoid recomputing it
    """
    sel = p >= threshold
    if exclusive:
        sel &= excl_mask if excl_mask is not None else exclusive_best(s1, b, p)
    if min_top is not None and min_top > threshold and len(p):
        top = np.zeros(int(s1.max()) + 1, np.float32)
        np.maximum.at(top, s1, np.where(sel, p, 0).astype(np.float32))
        sel &= top[s1] >= min_top
    return sel


def macro_f05(pred_sets, true_sets):
    """pred_sets/true_sets: dict s1_id -> set. Averaged over the keys of true_sets."""
    total = 0.0
    for k, truth in true_sets.items():
        pred = pred_sets.get(k, set())
        if not truth and not pred:
            total += 1.0
            continue
        if not truth or not pred:
            continue
        tp = len(pred & truth)
        total += 1.25 * tp / (len(pred) + 0.25 * len(truth))
    return total / max(1, len(true_sets))


def per_s1_scores(pair_s1, pair_true, sel, n_true, eval_s1):
    """Fast vectorised macro F0.5 for integer-indexed S1 records.

    pair_s1  : S1 index of every candidate pair
    pair_true: bool, pair is a true match
    sel      : bool, pair is predicted
    n_true   : per-S1 number of true matches (including ones missed by blocking)
    eval_s1  : S1 indices to average over
    """
    n = len(n_true)
    tp = np.bincount(pair_s1, weights=(sel & pair_true), minlength=n)
    npred = np.bincount(pair_s1, weights=sel, minlength=n)
    f = np.where(
        (n_true == 0) & (npred == 0), 1.0,
        np.where((n_true == 0) | (npred == 0), 0.0,
                 1.25 * tp / np.maximum(npred + 0.25 * n_true, 1e-9)))
    return f[eval_s1].mean(), f
