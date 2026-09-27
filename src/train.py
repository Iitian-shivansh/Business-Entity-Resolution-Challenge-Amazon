"""Train the two-stage matcher on the training split and report honest hold-out scores.

    python src/train.py --data-dir dataset --work-dir work

Steps
 1. normalise train sources -> parquet cache
 2. per country: blocking + pairwise features -> disk
 3. split Source 1 records into train / hold-out / rest
 4. stage 1: LightGBM on pairwise features (K-fold by S1 -> out-of-fold scores)
 5. stage 2: LightGBM on stage-1 scores + group/competition features
 6. tune the decision threshold for macro F0.5 on half of the hold-out,
    report on the other half
"""
import json
import os
import time

import lightgbm as lgb
import numpy as np

from config import get_args
from data import list_countries, log, prepare_split, read_ground_truth
from decode import decode, exclusive_best, per_s1_scores
from featurize import featurize_split, load_country_arrays
from features import FEATURE_NAMES
from models import (STAGE1_NAMES, STAGE1_PARAMS, STAGE1_ROUNDS, STAGE2_PARAMS,
                    STAGE2_ROUNDS, S2_NAMES, load_b_text, predict_stage1, save_models,
                    stage1_matrix, stage2_predict_all)
from stage2 import Stage2Context, extract_raw

ROLE_REST, ROLE_TRAIN, ROLE_HOLD = 0, 1, 2
HOUSE_COL = FEATURE_NAMES.index("house_eq")


def labels_for_country(cand, s1_ids, b_ids, gt):
    b_index = {e: i for i, e in enumerate(b_ids)}
    n_b = len(b_ids)
    n_true = np.zeros(len(s1_ids), np.int32)
    keys = []
    for i, e in enumerate(s1_ids):
        m = gt.get(e, "")
        if not m:
            continue
        ms = [x for x in m.split(",") if x]
        n_true[i] = len(ms)
        for x in ms:
            j = b_index.get(x)
            if j is not None:
                keys.append(i * n_b + j)
    true_keys = np.unique(np.asarray(keys, np.int64))
    pair_keys = cand["s1"].astype(np.int64) * n_b + cand["b"].astype(np.int64)
    y = np.isin(pair_keys, true_keys)
    return y, n_true


def transfer_check(args, C, H, countries, best_p1):
    """Leave-one-country-out: how much does stage 1 lose on a country it never saw?

    This is the closest thing we have to measuring the France situation. The stage-1
    rule tuned in-distribution is applied to a model trained without the country.
    """
    _, _, thr, min_top, exclusive = best_p1
    out = {}
    params = dict(STAGE1_PARAMS, num_threads=args.n_jobs, seed=args.seed)
    for c in countries:
        Xs, ys = [], []
        for o in countries:
            if o == c:
                continue
            d = C[o]
            rows = np.nonzero(d["role"][d["cand"]["s1"]] == ROLE_TRAIN)[0]
            Xs.append(stage1_matrix(d["X"], rows))
            ys.append(d["y"][rows])
        m = lgb.train(params, lgb.Dataset(np.vstack(Xs), np.concatenate(ys),
                                          feature_name=STAGE1_NAMES), STAGE1_ROUNDS)
        del Xs
        d, h = C[c], H[c]
        hp = np.nonzero(d["role"][d["cand"]["s1"]] == ROLE_HOLD)[0]
        p_unseen = m.predict(stage1_matrix(d["X"], hp), num_threads=args.n_jobs).astype(np.float32)
        ev = np.nonzero(d["role"] == ROLE_HOLD)[0]
        res = {}
        for name, p in (("seen", h["p1"]), ("unseen", p_unseen)):
            ex = exclusive_best(h["s1"], h["b"], p)
            sel = decode(h["s1"], h["b"], p, thr, exclusive=exclusive, min_top=min_top,
                         excl_mask=ex)
            res[name] = float(per_s1_scores(h["s1"], h["y"], sel, d["n_true"], ev)[0])
            res[name + "_pred_per_s1"] = float(sel.sum() / max(1, len(ev)))
        # what an oracle-free "rate matching" threshold would have achieved
        target = res["seen_pred_per_s1"]
        ex = exclusive_best(h["s1"], h["b"], p_unseen)
        for t in np.arange(thr, 0.999, 0.005):
            sel = decode(h["s1"], h["b"], p_unseen, t, exclusive=exclusive,
                         min_top=min_top, excl_mask=ex)
            if sel.sum() / max(1, len(ev)) <= target:
                break
        res["unseen_rate_matched"] = float(per_s1_scores(h["s1"], h["y"], sel, d["n_true"], ev)[0])
        # the stricter rule used for unseen countries: no match when both records
        # carry a house number and the numbers disagree
        house = np.asarray(d["X"][hp][:, HOUSE_COL], np.float32)
        sel_h = sel & (house != -1)
        res["unseen_rate_matched_strict_house"] = float(
            per_s1_scores(h["s1"], h["y"], sel_h, d["n_true"], ev)[0])
        res["rate_matched_threshold"] = float(t)
        out[c] = res
        log(f"  transfer check [{c}]: seen={res['seen']:.4f}  unseen={res['unseen']:.4f}  "
            f"unseen+rate-match={res['unseen_rate_matched']:.4f} (t={t:.3f})  "
            f"+strict-house={res['unseen_rate_matched_strict_house']:.4f}  "
            f"pred/S1 seen={res['seen_pred_per_s1']:.2f} unseen={res['unseen_pred_per_s1']:.2f}")
    return out


def main():
    args = get_args("Train the entity-resolution matcher")
    t0 = time.time()
    cache = os.path.join(args.work_dir, "cache")
    model_dir = os.path.join(args.work_dir, "models")
    prepare_split(args.data_dir, cache, "train", n_jobs=args.n_jobs)
    featurize_split(args, "train")
    gt = read_ground_truth(args.data_dir)
    countries = list_countries(cache, "train")
    rng = np.random.default_rng(args.seed)

    # ---------------- labels + role assignment ----------------
    C = {}
    for c in countries:
        cand, X, s1_ids, b_ids = load_country_arrays(args, "train", c)
        y, n_true = labels_for_country(cand, s1_ids, b_ids, gt)
        C[c] = dict(cand=cand, X=X, n_s1=len(s1_ids), n_b=len(b_ids), y=y, n_true=n_true)
        rec = y.sum() / max(1, n_true.sum())
        log(f"[{c}] pairs={len(y):,}  positives={int(y.sum()):,}  "
            f"blocking recall={rec:.4f}  singletons={np.mean(n_true == 0):.3f}")
        # recall by candidate rank: if it is still climbing at the last value, a larger
        # --top-k would find more matches
        ks = [k for k in (5, 10, 15, 20, 25, 30, 40, 50) if k < args.top_k] + [args.top_k]
        curve = "  ".join(f"@{k}={y[cand['rank'] < k].sum() / max(1, n_true.sum()):.4f}"
                          for k in ks)
        log(f"[{c}] blocking recall by rank: {curve}")

    total_s1 = sum(v["n_s1"] for v in C.values())
    perm = rng.permutation(total_s1)
    role_all = np.zeros(total_s1, np.int8)
    n_tr = min(args.train_s1, int(total_s1 * 0.6))
    n_ho = min(args.holdout_s1, total_s1 - n_tr)
    role_all[perm[:n_tr]] = ROLE_TRAIN
    role_all[perm[n_tr:n_tr + n_ho]] = ROLE_HOLD
    fold_all = rng.integers(0, args.folds, total_s1)
    half_all = rng.integers(0, 2, total_s1)       # hold-out: 0 = tune, 1 = report
    off = 0
    for c in countries:
        n = C[c]["n_s1"]
        C[c]["role"] = role_all[off:off + n]
        C[c]["fold"] = fold_all[off:off + n]
        C[c]["half"] = half_all[off:off + n]
        off += n

    # ---------------- stage 1 ----------------
    Xs, ys, fs, where = [], [], [], []
    for c in countries:
        d = C[c]
        rows = np.nonzero(d["role"][d["cand"]["s1"]] == ROLE_TRAIN)[0]
        Xs.append(stage1_matrix(d["X"], rows))
        ys.append(d["y"][rows])
        fs.append(d["fold"][d["cand"]["s1"][rows]])
        where.append((c, rows))
    X1, y1, f1 = np.vstack(Xs), np.concatenate(ys), np.concatenate(fs)
    del Xs
    log(f"stage 1: {len(y1):,} training pairs, positive rate {y1.mean():.3f}")
    oof = np.zeros(len(y1), np.float32)
    models1 = []
    params = dict(STAGE1_PARAMS, num_threads=args.n_jobs, seed=args.seed)
    for k in range(args.folds):
        tr, va = f1 != k, f1 == k
        ds = lgb.Dataset(X1[tr], y1[tr], feature_name=STAGE1_NAMES, free_raw_data=True)
        m = lgb.train(params, ds, STAGE1_ROUNDS)
        oof[va] = m.predict(X1[va], num_threads=args.n_jobs)
        models1.append(m)
        acc = ((oof[va] > 0.5) == y1[va]).mean()
        log(f"  stage 1 fold {k}: accuracy@0.5 = {acc:.4f}")
    imp = sorted(zip(models1[0].feature_importance("gain"), STAGE1_NAMES), reverse=True)
    log("  top stage-1 features: " + ", ".join(f"{n}" for _, n in imp[:12]))
    del X1

    pos = 0
    log("stage-1 prediction on all pairs (per country)...")
    for c, rows in where:
        d = C[c]
        log(f"  [{c}] predicting stage 1 on {len(d['y']):,} pairs...")
        p1 = predict_stage1(models1, d["X"], n_jobs=args.n_jobs)
        p1[rows] = oof[pos:pos + len(rows)]
        pos += len(rows)
        d["p1"] = p1
        log(f"  [{c}] stage-1 prediction done")

    # ---------------- stage 2 ----------------
    log("building stage-2 features...")
    X2s, y2s = [], []
    for c in countries:
        d = C[c]
        log(f"  [{c}] loading text for stage 2...")
        d["b_names"], d["b_addrs"] = load_b_text(cache, "train", c)
        log(f"  [{c}] computing stage-2 context + features...")
        ctx = Stage2Context(d["cand"], d["p1"], d["n_s1"], d["n_b"])
        raw = extract_raw(d["X"])
        rows = np.nonzero(d["role"][d["cand"]["s1"]] == ROLE_TRAIN)[0]
        X2s.append(ctx.features(rows, raw, d["b_names"], d["b_addrs"]))
        y2s.append(d["y"][rows])
        del ctx, raw
        log(f"  [{c}] stage-2 features done")
    X2, y2 = np.vstack(X2s), np.concatenate(y2s)
    del X2s
    log(f"stage 2: {len(y2):,} training pairs")
    params2 = dict(STAGE2_PARAMS, num_threads=args.n_jobs, seed=args.seed)
    model2 = lgb.train(params2, lgb.Dataset(X2, y2, feature_name=S2_NAMES), STAGE2_ROUNDS)
    imp = sorted(zip(model2.feature_importance("gain"), S2_NAMES), reverse=True)
    log("  top stage-2 features: " + ", ".join(f"{n}" for _, n in imp[:10]))
    del X2
    save_models(model_dir, models1, model2)

    for c in countries:
        d = C[c]
        d["p2"], _, _ = stage2_predict_all(model2, d["cand"], d["p1"], d["X"], d["n_s1"],
                                           d["n_b"], d["b_names"], d["b_addrs"],
                                           n_jobs=args.n_jobs)
        del d["b_names"], d["b_addrs"]

    # ---------------- decision tuning on hold-out ----------------
    # Exclusivity is decided over ALL pairs (every S1 competes for a record); the
    # scoring itself only looks at hold-out S1 records.
    H = {}
    for c in countries:
        d = C[c]
        s1, b = d["cand"]["s1"], d["cand"]["b"].astype(np.int64)
        hp = np.nonzero(d["role"][s1] == ROLE_HOLD)[0]
        H[c] = dict(s1=s1[hp], b=b[hp], y=d["y"][hp],
                    p1=d["p1"][hp], p2=d["p2"][hp],
                    ex_p1=exclusive_best(s1, b, d["p1"])[hp],
                    ex_p2=exclusive_best(s1, b, d["p2"])[hp])

    def evaluate(prob_key, thr, min_top, exclusive, half, cs=None):
        tot, cnt = 0.0, 0
        for c in (cs or countries):
            d, h = C[c], H[c]
            sel = decode(h["s1"], h["b"], h[prob_key], thr, exclusive=exclusive,
                         min_top=min_top, excl_mask=h["ex_" + prob_key])
            mask = d["role"] == ROLE_HOLD
            if half is not None:
                mask &= d["half"] == half
            ev = np.nonzero(mask)[0]
            if len(ev) == 0:
                continue
            f, _ = per_s1_scores(h["s1"], h["y"], sel, d["n_true"], ev)
            tot += f * len(ev)
            cnt += len(ev)
        return tot / max(cnt, 1)

    grid_t = np.round(np.arange(0.30, 0.96, 0.025), 3)
    best = None
    best_p1 = None
    for key in ("p1", "p2"):
        for exclusive in (False, True):
            for thr in grid_t:
                for min_top in (None, 0.7, 0.8, 0.9):
                    if min_top is not None and min_top <= thr:
                        continue
                    f = evaluate(key, thr, min_top, exclusive, half=0)
                    if best is None or f > best[0]:
                        best = (f, key, float(thr), min_top, exclusive)
                    if key == "p1" and (best_p1 is None or f > best_p1[0]):
                        best_p1 = (f, key, float(thr), min_top, exclusive)
        log(f"  best so far after {key}: F0.5(tune)={best[0]:.4f} {best[1:]}")
    f_tune, key, thr, min_top, exclusive = best
    f_rep = evaluate(key, thr, min_top, exclusive, half=1)
    base = evaluate("p1", 0.5, None, False, half=1)
    log(f"HOLD-OUT macro F0.5 (report half) = {f_rep:.4f}   "
        f"[stage-1 @0.5 baseline = {base:.4f}]")
    per_country = {}
    for c in countries:
        per_country[c] = evaluate(key, thr, min_top, exclusive, half=None, cs=[c])
        log(f"  [{c}] hold-out macro F0.5 = {per_country[c]:.4f}")
    # how many matches the rule hands out per S1 record on the hold-out; used to
    # sanity-check (and, for unseen countries, calibrate) the test predictions
    n_sel, n_ev = 0, 0
    for c in countries:
        d, h = C[c], H[c]
        sel = decode(h["s1"], h["b"], h[key], thr, exclusive=exclusive,
                     min_top=min_top, excl_mask=h["ex_" + key])
        n_sel += int(sel.sum())
        n_ev += int((d["role"] == ROLE_HOLD).sum())
    pred_per_s1 = n_sel / max(1, n_ev)
    true_per_s1 = float(np.mean(np.concatenate(
        [C[c]["n_true"][C[c]["role"] == ROLE_HOLD] for c in countries])))
    log(f"hold-out: {pred_per_s1:.3f} predicted vs {true_per_s1:.3f} true matches per S1")

    transfer = {}
    if args.transfer_check and len(countries) > 1:
        transfer = transfer_check(args, C, H, countries, best_p1)
    decision = dict(prob=key, threshold=thr, min_top=min_top, exclusive=exclusive,
                    holdout_f05_tune=f_tune, holdout_f05_report=f_rep,
                    per_country=per_country, top_k=args.top_k, df_cap=args.df_cap,
                    train_countries=countries, holdout_pred_per_s1=pred_per_s1,
                    holdout_true_per_s1=true_per_s1, transfer_check=transfer)
    json.dump(decision, open(os.path.join(model_dir, "decision.json"), "w"), indent=2)
    log(f"saved decision rule: {decision}")
    log(f"training finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
