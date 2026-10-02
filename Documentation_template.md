# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** Komalika Das
**Submission Date:** [Date]

---

## 1. Executive Summary

Our pipeline normalises every record (transliterating Indic scripts, folding French accents, canonicalising legal forms and address abbreviations), generates candidates with an IDF-weighted inverted index over hashed blocking keys, and scores them with two LightGBM models. The second model sees how each candidate scores against every other Source 1 entity that retrieved it, which lets it refuse merges where a record clearly belongs to someone else. On a held-out set of training Source 1 records the final version scores a macro F0.5 of **0.9566** (US 0.9686, India 0.9375).

---

## 2. Methodology

### 2.1 Problem Analysis

**Scale.** Train has 2.2M Source 1, 5.0M Source 2 and 5.3M Source 3 records. Test has 1.7M, 4.9M and 5.1M. Anything quadratic was ruled out from the start, so every step works on sparse matrices in chunks.

**How matches are structured.** About 5.6% of training Source 1 entities are singletons. The rest have between 1 and 11 matches, 3.47 on average. Every Source 2/3 id appears in at most one ground-truth row, so a record never belongs to two Source 1 entities. A match always carries the same country label as its Source 1 entity. These two facts shaped the whole design: blocking runs per country, and the model is allowed to treat candidates as competing for a single owner.

**Name noise we found.**
- Native scripts are common. Roughly 24% of Indian and 7% of US Source 2/3 names are written in Devanagari, Gujarati, Kannada, Tamil and other scripts.
- 3 to 4% of names are web handles or domains, such as `aksharfisheries.com` or `#blockeu`.
- Some names carry injected ids (`(ID: 43535)`, `& #97816`), repeated words, shuffled word order or character typos.
- Digits replace letters inside words: `Po1yesters`, `5ARL`, `br0s`.
- Transliteration drifts a long way from the English spelling: `Sree Infrastructure` turns up as `sri inphrastrkcr`, `Maa Consultants` as `ma knsltents`.
- A small share of true matches have a name that has nothing to do with the entity. Only the address links them.

**Address noise we found.**
- 3.5 to 4% of addresses are empty, and others hold placeholders (`null`, `<NULL>`, `N/A`).
- States appear in native script (`राजस्थान`) or as codes.
- House numbers are padded (`0045`), decorated (`##222`, `130.`) or truncated (`3601` becomes `360`).
- Components are reordered, and city names pick up suffixes like `CDP`, `Borough` or `City`.
- French records use `R.` for rue, `N°`, `BIS` and accented characters.

**France.** France has no training labels, and French business names are generic: a city, a common word and a legal form (`Nantes Amis SARL`). Many different businesses share most of their name tokens and the city. We treated this as the biggest generalisation risk in the challenge.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage classifier + constrained decoding

**Core Innovation:** Competition features in the second stage. A candidate is judged against the Source 1 entity that retrieved it and against every other Source 1 entity that retrieved the same record. If the record fits another business better, the model learns to drop it. Combined with a decision rule tuned directly on macro F0.5 (singletons included), this is where most of the precision comes from.

Nothing in the pipeline uses the country label as a feature, and no external data, API or pretrained model is involved. France goes through exactly the same code as US and India.

---

## 3. Candidate Generation (Blocking)

**Blocking keys used.** Each record is turned into a set of string keys, hashed into 2^24 buckets:
- name-token skeletons (first letter plus consonants, repeats collapsed), so small typos and vowel changes still collide;
- unordered pairs of the first four name tokens;
- the joined core name, both as a 9-character prefix and in full;
- concatenations of adjacent name words, so `pediatric dentistry` meets `pediatricdentistry.com`;
- 4-letter sound prefixes of name words, paired with each other or with a house number. These catch heavy transliteration drift (`knsltents` against `consultants`). The sound keys treat a hard `c` as `k`;
- address-word skeletons and 5/6-digit ZIP and PIN codes;
- house number combined with a street word, and name token combined with house number.

Before any key is built, digits that stand in for letters inside names are corrected (`po1yesters` → `polyesters`).

**Scoring candidates.** Keys shared by more than 3,000 records are dropped as too common. A candidate's score is the sum of the IDF weights of the keys it shares with the Source 1 record, computed as a sparse matrix product of Source 1 against Source 2+3, one country at a time. The product runs in chunks sized from an estimate of its non-zeros, which keeps memory flat on the full data. The 40 highest-scoring candidates per Source 1 record go forward.

**Candidate pairs generated:** [total from `predict_log.txt`, the "blocking done" lines], at most 40 per Source 1 record.

**How we made sure true matches were not lost.**
- Several independent key families cover each record. A typo in the name, an empty address or a reordered address each leave other keys intact.
- Transliteration and digit correction run before key generation.
- Training logs recall at every candidate rank (5, 10, 15 ... 40). In our first full run recall was still rising at rank 30, so we widened the net to 40 candidates and raised the frequency cap from 1,000 to 3,000.

| Blocking recall (training data) | First full run (k=30) | Final run (k=40) |
|---|---|---|
| India | 86.4% | 90.0% |
| US | 93.3% | 94.7% |

India gained the most from the sound-prefix keys, because most of its missed matches were transliteration drift.

---

## 4. Matching Model

**Features used (about 50 per pair):**

- **Name features.** Levenshtein ratio, token-sort, token-set and partial ratio, and Jaro-Winkler on the full name and on the core name (legal forms removed). The same ratios on the space-free joined name, which handles web handles. Token-set ratio on phonetic skeletons. IDF-weighted cosine and coverage in both directions. Overlap of "specific" tokens, meaning tokens that appear in under 0.1% of the country's records, so city names and generic words carry no weight. Name informativeness, the mean IDF scaled by country size.
- **Address features.** The same string scorers, IDF cosine and coverage, and specific-token overlap. Jaccard and coverage of the number sets. House-number agreement and ZIP/PIN agreement, coded as -1 (conflict), 0 (missing) and 1 (equal).
- **Other.** Blocking rank and blocking score relative to the record's best candidate. The raw score is left out because it grows with country size. The candidate's source (S2 or S3), and flags for native script, web-style names and empty addresses.

**Model type:**

- **Stage 1:** LightGBM (MIT licence), 500 trees with 127 leaves, trained on the candidate pairs of about 300,000 training Source 1 records. Out-of-fold scores come from 3-fold cross-validation grouped by Source 1 record, so stage 2 never trains on scores the model produced for pairs it had already seen.
- **Stage 2:** a second LightGBM on the stage-1 score plus group features. From the Source 1 side: the candidate's rank, the top score, the gap to the top score, and the sum and count of scores. From the candidate side: the best score the same record gets from any other Source 1 entity, the margin to that score, and how many Source 1 entities retrieved it. Support features measure how similar the candidate is to the Source 1 entity's strongest other candidate.

Both models are small gradient-boosted tree ensembles, far inside the licence and 8B-parameter limits.

**Threshold selection method:** We held out 120,000 training Source 1 records and split them in half. On the first half we grid-searched which probability to use (stage 1 or stage 2), the threshold, whether to enforce exclusive assignment (a record goes only to its best-scoring Source 1 entity), and an optional minimum top score per entity, all to maximise macro F0.5 with singletons counted. The second half was used only to report the score. The chosen threshold was 0.65.

**France (unseen country).** The model applies to France unchanged. As a safety check, if France ends up with more predicted matches per Source 1 record than the true training rate of 3.47, the threshold is raised until the two agree. It is never lowered. In our first full run this check compared against the predicted training rate instead, which is held down by blocking misses, and it pushed France's threshold from 0.65 to 0.985. Spot checks showed the French matches it threw away were mostly correct. After the fix, France stays at 0.65 and receives 3.38 matches per record.

---

## 5. Results & Error Analysis

**F_0.5 score (macro), held-out training Source 1 records:**

| | First full run | Final run |
|---|---|---|
| Overall | 0.9446 | **0.9566** |
| US | 0.963 | 0.9686 |
| India | 0.917 | 0.9375 |

France has no labels, so it has no hold-out score. We reviewed random samples of French predictions by hand with `inspect_matches.py` and they looked clean.

The classifier was rarely the bottleneck. In the first run it predicted 3.07 matches per Source 1 record, while blocking only delivered 3.1 true matches per record against a true average of 3.47. Nearly every true match that reached the model was found, so most of the remaining loss sits in blocking.

**Common false positives (wrong merges):**
- Different businesses with generic or near-identical names in the same city, at close but different house numbers (`9720` against `9725`, `108` against `10`).
- Different businesses in the same building where the candidate's name looks like noise.
- Both patterns are more frequent in France, because French names share so many tokens.

**Common false negatives (missed matches):**
- Records crowded out of their Source 1 entity's top 40 by look-alikes, even though their own best Source 1 match was the correct one. For example, `om knstrksn | 129 gorakhpur` never reached `om construction, 129 ... gorakhpur`.
- True duplicates with truncated house numbers (`3601` → `360`). The model only knows "same" or "different" for house numbers, so it cannot tell a truncation apart from a neighbour.
- Candidates with an unrelated name and an empty address. Nothing in the record links them to the entity, so no method can recover these.
- Very heavy transliteration drift (`imjiniyring` for `engineering`).

---

## 6. Conclusion

Most of the score came from blocking, not from the classifier: each round of better normalisation and wider keys raised the recall ceiling, and the two-stage model with competition features turned that ceiling into precision-heavy F0.5. The hardest part was France, where we had no labels; we handled it by keeping every feature country-agnostic and making the safety threshold one-directional. With more time we would add reverse search in blocking (letting each Source 2/3 record pull in its own best Source 1 entities) and finer house-number features, both aimed directly at the misses listed above.

---

## Appendix

### A. Code Artefacts

Everything lives in `code/business_entity_resolution/`. `README.md` there has the exact commands; in short:

```bash
pip install -r requirements.txt
python src/train.py   --data-dir dataset --work-dir work --n-jobs 4
python src/predict.py --data-dir dataset --work-dir work --out-dir output --n-jobs 4
```

`train.py` builds the cache, blocks and featurises the training data, trains both stages and tunes the decision rule. `predict.py` repeats blocking and featurisation on the test data, applies the rule and writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`. The candidate file holds exactly the pairs the model scored, so every matched id also appears there.

| File | What it does |
|---|---|
| `src/normalize.py` | transliteration, accent folding, digit-for-letter fixes, legal-form and abbreviation canonicalisation |
| `src/data.py` | chunked TSV reading, parallel normalisation, parquet cache |
| `src/blocking.py` | blocking keys and IDF-weighted sparse top-k candidate search |
| `src/features.py`, `src/featurize.py` | pairwise features, written to disk per country |
| `src/stage2.py` | group and competition features |
| `src/models.py` | LightGBM settings and helpers |
| `src/decode.py` | thresholds, exclusive assignment, macro F0.5 |
| `src/train.py`, `src/predict.py` | entry points |
| `src/inspect_matches.py` | prints predicted matches side by side for manual review |

### B. Additional Results

Our final outputs were produced on a Kaggle CPU notebook (4 cores, about 30 GB RAM). Training and prediction together took about 6 to 7 hours. Both output files have 1,732,545 lines: one per test Source 1 record plus the header.

Version history on the full data:

| Change | India recall | US recall | Hold-out F0.5 |
|---|---|---|---|
| First full run: k=30, frequency cap 1,000 | 86.4% | 93.3% | 0.9446 |
| Sound-prefix keys, `c`→`k`, digit fixes in names, k=40, cap 3,000, France threshold fix | 90.0% | 94.7% | 0.9566 |
