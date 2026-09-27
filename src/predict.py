"""Run the trained matcher on the test split and write both submission files.

    python src/predict.py --data-dir dataset --work-dir work --out-dir output

Writes
    output/candidate_pairs.tsv   every candidate the model scored (blocking output)
    output/matching_results.tsv  the final matches (subset of the candidates)
Every Source 1 test entity gets exactly one row in both files, in input order.
"""
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq

from config import get_args
from data import cache_path, list_countries, log, prepare_split
from decode import decode, exclusive_best
from features import FEATURE_NAMES

HOUSE_COL = FEATURE_NAMES.index("house_eq")
from featurize import featurize_split, load_country_arrays
from models import load_b_text, load_models, predict_stage1, stage2_predict_all


def main():
    args = get_args("Predict matches for the test split")
    t0 = time.time()
    cache = os.path.join(args.work_dir, "cache")
    model_dir = os.path.join(args.work_dir, "models")
    models1, model2 = load_models(model_dir)
    rule = json.load(open(os.path.join(model_dir, "decision.json")))
    log(f"decision rule: prob={rule['prob']} threshold={rule['threshold']} "
        f"min_top={rule['min_top']} exclusive={rule['exclusive']}")
    # blocking settings must match the ones the model was trained with
    args.top_k, args.df_cap = rule["top_k"], rule["df_cap"]

    prepare_split(args.data_dir, cache, "test", n_jobs=args.n_jobs)
    featurize_split(args, "test")

    os.makedirs(args.out_dir, exist_ok=True)
    f_cand = open(os.path.join(args.out_dir, "candidate_pairs.tsv"), "w",
                  encoding="utf-8", newline="\n")
    f_match = open(os.path.join(args.out_dir, "matching_results.tsv"), "w",
                   encoding="utf-8", newline="\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    stats, written = {}, 0
    for c in list_countries(cache, "test"):
        cand, X, s1_ids, b_ids = load_country_arrays(args, "test", c)
        n_s1, n_pairs = len(s1_ids), len(cand["s1"])
        s1, b = cand["s1"], cand["b"].astype(np.int64)
        if n_pairs:
            p1 = predict_stage1(models1, X, n_jobs=args.n_jobs)
            if rule["prob"] == "p2":
                b_names, b_addrs = load_b_text(cache, "test", c)
                p, _, _ = stage2_predict_all(model2, cand, p1, X, n_s1, len(b_ids),
                                             b_names, b_addrs, n_jobs=args.n_jobs)
                del b_names, b_addrs
            else:
                p = p1
            thr = rule["threshold"]
            ex = exclusive_best(s1, b, p) if rule["exclusive"] else None
            sel = decode(s1, b, p, thr, exclusive=rule["exclusive"],
                         min_top=rule["min_top"], excl_mask=ex)
            unseen = c not in rule.get("train_countries", [c])
            house_ok = None
            if unseen and args.unseen_strict_house:
                # No labels for this country, so be stricter: when both records carry a
                # house number and the numbers disagree, do not merge.
                house = np.concatenate([np.asarray(X[a:a + 2_000_000][:, HOUSE_COL], np.float32)
                                        for a in range(0, n_pairs, 2_000_000)])
                house_ok = house != -1
                sel &= house_ok
            if unseen and args.unseen_policy == "rate_match":
                # The model never saw this country. If it hands out more matches per
                # S1 than it does on the training hold-out, it is over-confident here:
                # raise the threshold (never lower it) until the rates agree.
                target = rule["holdout_pred_per_s1"]
                log(f"[{c}] unseen country: {sel.sum() / n_s1:.3f} matches/S1 at "
                    f"t={thr:.3f} (training hold-out: {target:.3f})")
                while sel.sum() / n_s1 > target and thr < 0.995:
                    thr = round(thr + 0.005, 3)
                    sel = decode(s1, b, p, thr, exclusive=rule["exclusive"],
                                 min_top=rule["min_top"], excl_mask=ex)
                    if house_ok is not None:
                        sel &= house_ok
                log(f"[{c}] rate-matched threshold = {thr:.3f} "
                    f"({sel.sum() / n_s1:.3f} matches/S1)")
        else:
            p = np.zeros(0, np.float32)
            sel = np.zeros(0, bool)

        # pairs are sorted by s1: one contiguous block per S1 record
        starts = np.searchsorted(s1, np.arange(n_s1), side="left")
        ends = np.searchsorted(s1, np.arange(n_s1), side="right")
        for i in range(n_s1):
            e = s1_ids[i]
            a, z = starts[i], ends[i]
            if a == z:
                f_cand.write(e + "\t\n")
                f_match.write(e + "\t\n")
                continue
            f_cand.write(e + "\t" + ",".join(b_ids[b[a:z]]) + "\n")
            chosen = np.arange(a, z)[sel[a:z]]
            chosen = chosen[np.argsort(-p[chosen], kind="stable")]   # best match first
            f_match.write(e + "\t" + ",".join(b_ids[b[chosen]]) + "\n")
        written += n_s1
        n_matched = int(np.unique(s1[sel]).size) if n_pairs else 0
        stats[c] = dict(s1=int(n_s1), pairs=int(n_pairs), matched_s1=n_matched,
                        matches=int(sel.sum()),
                        threshold=float(thr) if n_pairs else None)
        log(f"[{c}] {stats[c]}  ({n_matched / max(1, n_s1):.1%} of S1 get a match, "
            f"{sel.sum() / max(1, n_matched):.2f} matches each)")
        del cand, X, p, sel
    f_cand.close()
    f_match.close()

    n_expected = pq.read_metadata(cache_path(cache, "test", 1)).num_rows
    if written != n_expected:
        raise RuntimeError(f"wrote {written} rows but test_source1 has {n_expected}")
    json.dump(stats, open(os.path.join(args.out_dir, "prediction_stats.json"), "w"), indent=2)
    log(f"wrote {written:,} rows to {args.out_dir}/ in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
