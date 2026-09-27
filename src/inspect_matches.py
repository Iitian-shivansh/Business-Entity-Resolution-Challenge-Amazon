"""Print a random sample of predicted matches for one country, side by side.

    python src/inspect_matches.py --data-dir dataset --out-dir output --country France

Use it after predict.py to eyeball precision, especially for France, which has no
training labels. Every line you judge wrong is a false merge the metric punishes.
"""
import argparse
import csv
import random

import pandas as pd

R = dict(sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default="dataset")
p.add_argument("--out-dir", default="output")
p.add_argument("--country", default="France")
p.add_argument("--n", type=int, default=25)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

s1 = pd.read_csv(f"{a.data_dir}/test/test_source1.tsv", **R)
s1 = s1[s1.country.str.strip() == a.country].set_index("entity_id")
res = pd.read_csv(f"{a.out_dir}/matching_results.tsv", **R)
res = res[res.source1_entity_id.isin(s1.index) & (res.matched_entity_ids != "")]
print(f"{a.country}: {len(res):,} of {len(s1):,} S1 records have matches")
pick = res.sample(min(a.n, len(res)), random_state=a.seed)
need = {m for x in pick.matched_entity_ids for m in x.split(",")}
b = pd.concat([pd.read_csv(f"{a.data_dir}/test/test_source{k}.tsv", **R) for k in (2, 3)])
b = b[b.entity_id.isin(need)].set_index("entity_id")
for _, r in pick.iterrows():
    x = s1.loc[r.source1_entity_id]
    print(f"\nS1  {x.business_name}  |  {x.business_address}")
    for m in r.matched_entity_ids.split(","):
        y = b.loc[m]
        print(f"    {m[:2]}  {y.business_name}  |  {y.business_address}")
