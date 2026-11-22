# MS3 Summary — Padel Predictor

**Noah Rodriguez** · [github.com/Noah-Rod/padel-predictor](https://github.com/Noah-Rod/padel-predictor)

## 1 What I achieved in this milestone

**What works now that did not before.** At MS2 the training pipeline ran, but it promoted its first model unconditionally and judged later ones on a comparison that flattered them. Now every run is held to the same bar:

- **Training on a schedule and on new data.** After every successful feature run and on Mondays, on 6,556 labelled matches from February 2023. Every run is a W&B run with parameters, metrics, baselines and the promotion reason; every candidate is a versioned artifact, and only a promoted one gets the `champion` alias.
- **A champion is judged on matches it has never seen.** Backfilling 2023 and 2024 moved the test set onto matches champion v2 was trained on. Each model records its last training match, and both are compared only after it (`b7e529e`).
- **The provider's Elo is the bar for serving.** It is the first champion of an empty registry, and every candidate has to beat it on the matches the provider rates (`4b3eda3`, `5ea6cb8`, `304b4d5`).
- **Experiments on the training split.** `padel ablation` and `padel training-window`, a manual workflow, every variant a W&B run (`e27e541`, `d61124c`).

**Key decisions, and why.** *The success criterion counts only where the provider has a rating*: a 0.5 for an unrated match is not the provider's prediction (E1). *One publication rule for ranking snapshots*: label date plus a lag, applied when features are built, so backfilled and live snapshots count from the same moment (A1). *Missing players are read one by one, not by paging the full ranking lists*: one request returns a player's points and Elo for a whole window (E1). *An unranked player stays at 0 points*: I tried "missing" and the model got worse (A3).

**Look at first.** (1) `compare_with_champion` and `beats_provider_elo_where_rated` in `training/pipeline.py`. (2) `publication_times` in `common/features.py` and the tests under "One publication rule" in `tests/test_features.py`. (3) The latest *Feature pipeline* run in the Actions tab: its summary ends with the baselines on the rated matches and when each weekly ranking was first seen.

**Results** (runs of 22.11.2026, Plus plan; test set: 1,311 matches from 2026-06-26; log-loss gaps with 95 % bootstrap intervals, negative is better):

| | 1,024 test matches with a provider rating (78 %) | All 1,311 test matches |
| --- | --- | --- |
| Model | 0.492 | 0.487 (accuracy 77.4 %) |
| Provider Elo | 0.502 (in-house Elo 0.520) | 0.544 (unrated at 0.5) |
| Model − provider | **−0.011 [−0.032, +0.011]: not met** | [−0.078, −0.036], flattered by the 0.5 fill |

The model is ahead of the provider on average, but the interval includes zero: **the success criterion is not met.** No candidate has replaced champion v2 since the 2023 backfill; the latest was 0.001 ahead of it, short of the 0.002 a promotion needs.

| Variant | Log loss | vs provider, rated matches | vs the full model |
| --- | ---: | ---: | ---: |
| All features, all history | 0.4874 | −0.0106 [−0.0319, +0.0105] | – |
| without partnership | 0.4880 | −0.0125 [−0.0331, +0.0088] | +0.0006 [−0.0062, +0.0071] |
| without form | 0.4856 | −0.0138 [−0.0343, +0.0065] | −0.0018 [−0.0053, +0.0015] |
| without rest | 0.4869 | −0.0113 [−0.0320, +0.0091] | −0.0005 [−0.0037, +0.0025] |
| fitted on the last 30 months | 0.4833 | −0.0172 [−0.0377, +0.0040] | −0.0041 [−0.0098, +0.0019] |
| fitted on the last 12 months | 0.4944 | −0.0070 [−0.0300, +0.0155] | +0.0070 [−0.0026, +0.0170] |

*What partnership, form and rest add (MS1 point E3):* nothing I can measure. Every interval against the full model includes zero; the model is carried by the in-house Elo, the ranking points and the tier. I keep the three features until MS4 and decide on more test matches. *Does old history hurt?* No window (30, 24, 18, 12, 6 months) differs reliably from all history, so that is not what holds the candidates back.

**What it cost.** A3 as proposed went to the live store untested and made the model worse for one run before I reverted it. Two backfills died 40 minutes in on Hopsworks errors the retry loop did not recognise (`f5c1a84`), and the first live run of the window experiment failed on timestamps without a time zone (`eecf091`).

## 2 Response to MS2 reviews

Labels as in the editorial summary; A2 and B2 are answered under E1, where the editors placed them.

- **E1 · The success criterion is measured on a flattered comparison — Partly** (also A2, B2). The interval on the rated matches is the primary number: in the table above, the training summary, the W&B tag `success_criterion_met`, the experiment tables and `padel data-report`, which also scores the in-house Elo on the same matches (`5ea6cb8`). *Differently than proposed:* instead of paging the full ranking lists, every player missing from the top 200 is read from `/players/{id}/rankings`, one request per player and window (`723d90c`). Re-reading June to November 2026 raised the rated share of the test set from 63 % to 78 %; the rest are players the provider has no Elo for. *Open:* the history before June 2026 is being re-read in ranges, so until then `rank_points_diff` (B2) lacks those players in older training rows. *Where:* `training/dataset.py`, `features/sources.py`, `features/report.py`.
- **A1 · Rankings count as "published" at different times in backfill and live — Partly.** `publication_times` dates every snapshot at its label date plus `features.rankings_publication_lag_days`, whenever it was fetched, in the module training and inference share (`0c69806`). Features are rebuilt from the raw store on every run, so the rule reached all history without rewriting the store. A test stores the same snapshot as backfilled and as seen live and expects identical rows. As reviewer A suggested, the store keeps when a live run first saw each snapshot, and every feature run lists these delays. So far the rankings dated 09.11 and 16.11 were there at the daily run on their own Monday. *Open:* the lag is still the assumed seven days; I lower it once two more weeks confirm a delay of under a day. *Where:* `common/features.py`, `features/report.py`, `config/settings.yaml`.
- **E2 · Course material and received reviews in the public repository — Partly.** Both folders are out of the tree and git-ignored (`6adcf20`), and `main`, `ms1` and `ms2` were rewritten without them; the files are kept locally. The rewrite changed the hashes from 01.10 on, so those cited in the MS2 summary no longer resolve on `main`. *Not done, on purpose:* the merged feature branches and pull request #1 still show the files. They document how the project was built, and rewriting eleven merged branches would break more references than it protects.
- **A3 · An unranked player gets 0 points — Rejected, after trying it.** I implemented it (`c48a8e8`) and ran it on the live store: on the same test matches the log loss went from 0.492 to 0.518, and champion v2, trained on the 0, from 0.493 to 0.520; after the revert (`c914878`) it was 0.492 again. The feature is one difference per match, so a missing value cannot say *which* pair is unknown, while a 0 keeps the direction, and a pair outside the ranking lists is nearly always the weaker one. The explicit flag would be a seventh model input for a case the per-player reads keep shrinking; I did not add it. *Where:* `common/features.py` (`features_for`), `tests/test_features.py`.
- **B1 · The first trained model becomes champion without beating anything — Changed.** An empty registry starts with the provider's Elo as champion (`4b3eda3`). My live registry already held trained champions, so every candidate that beats a trained champion now also has to beat the provider's Elo on the rated matches (`304b4d5`). Champion v2 predates the rule and stays until a candidate passes both: demoting it would serve 0.5 for the matches the provider does not rate. *Where:* `training/pipeline.py`, `training/baseline.py`.
- **E3 · Let the real schedule accumulate — Partly.** The feature workflow no longer falls back: without `PADEL_SOURCE` or a hosted store its first step fails the run (`2b92ab7`). The README has a badge per workflow. The cron has run on real data into Hopsworks every day since 05.11, each run ending with the data report (GitHub starts the 04:15 UTC cron around 12:30): 18 green runs by 22.11, four weeks by the deadline. *Where:* `.github/workflows/feature-pipeline.yml`, README.
