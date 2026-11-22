# MS3 Summary — Padel Predictor

**Noah Rodriguez** · [github.com/Noah-Rod/padel-predictor](https://github.com/Noah-Rod/padel-predictor)

> Working draft for `docs/ms3_summary.pdf` (due 03.12.2026, at most 2 pages).
> Section 2 answers every MS2 point as of 22.11. Before exporting: refresh the
> numbers in section 1 after the ranking backfill, add the ablation and
> training-window tables, state the measured ranking delay under A1, and delete
> this block.

## 1 What I achieved in this milestone

**What works now that did not before.** At MS2 the training pipeline ran, but it promoted the first model unconditionally and judged every later one on a comparison that flattered it. Now a model is trained, tracked and promoted against a bar that is the same for every run:

- **Training runs on a schedule and on new data.** After every successful feature run and on Mondays, on the features of 6,554 labelled matches from February 2023 to November 2026. Every run is a W&B run with its parameters, metrics, baselines and the promotion reason; every candidate is a versioned model artifact and only a promoted one gets the `champion` alias.
- **A champion is judged on matches it has never seen.** The test set is the newest fifth of the store, so backfilling 2023 and 2024 moved it back onto matches champion v2 had been trained on. Each model records its last training match, and candidate and champion are compared only after it (`b7e529e`).
- **The provider's Elo is the bar for serving.** It is the first champion of an empty registry, and any candidate has to beat it on the matches the provider has a rating for before it is served (`4b3eda3`, `5ea6cb8`, `304b4d5`).
- **Experiments on the training split.** `padel ablation` drops partnership, form and rest in turn; `padel training-window` fits on the last 30 to 6 months only. Both score every variant against the provider's Elo on the rated matches and against the full model (`e27e541`, `d61124c`).

**Key decisions, and why.**

- *The success criterion is measured only where the provider has a rating.* A 0.5 for an unrated match is not the provider's prediction, and scoring against it flattered the model (E1).
- *An unranked player stays at 0 points.* I tried leaving the value missing, as reviewer A proposed, and the model got worse; with one difference per match, only the 0 says which pair is the unknown one (A3).
- *One publication rule for ranking snapshots.* The label date plus a lag, applied when the features are built, so a backfilled snapshot and one seen live count from the same moment (A1).
- *Read missing players one by one instead of paging the full ranking lists.* One request returns a player's weekly points and Elo for a whole window; the full lists cost one request per 50 players, per list, category and week (E1).

**Look at first.** (1) `compare_with_champion` and `beats_provider_elo_where_rated` in `training/pipeline.py`: who a candidate is compared with, and on which matches. (2) `publication_times` and `attach_rankings` in `common/features.py` with the tests under "One publication rule" in `tests/test_features.py`. (3) The latest *Feature pipeline* run in the Actions tab: its summary ends with the baselines on the rated matches and the table of when each weekly ranking was first seen.

**Results.** Training run of 22.11.2026 on the code of this milestone (W&B, Plus plan; 5,243 training matches), before the ranking backfill of E1:

| | 891 test matches with a provider rating (68 %) | All 1,311 test matches (from 2026-06-26) |
| --- | --- | --- |
| Model: log loss | 0.494 | 0.492 (accuracy 77.2 %) |
| Provider Elo: log loss | 0.506 (in-house Elo: 0.508) | 0.566 (unrated at 0.5) |
| Model − provider, 95 % CI | **−0.012 [−0.035, +0.011]: not met** | [−0.094, −0.053], flattered by the 0.5 fill |

The model is ahead of the provider on the rated matches on average, but the interval still includes zero: **the success criterion is not met.** No candidate has replaced champion v2 since the 2023 backfill; the latest was 0.001 behind it on the 1,295 test matches v2 has not seen.

## 2 Response to MS2 reviews

Labels as in the editorial summary. A2 and B2 are answered under E1, where the editors placed them.

- **E1 · The success criterion is measured on a flattered comparison — Partly** (also A2, B2). *Done:* the interval on the rated matches is the primary number: in the table above, in the training summary (`criterion met=`), in the W&B tag `success_criterion_met`, in the experiment tables and in `padel data-report`, which also scores the in-house Elo on the same matches (`5ea6cb8`). The gap on all matches is still logged, labelled as flattered. *Differently than proposed:* I do not page through the full ranking lists. Every player of the fetched matches who is missing from the top 200 is read from `/players/{id}/rankings`, one request per player and window (`723d90c`); that reaches the same players at a fraction of the requests. *Open:* this has run daily since 17.11, which raised the coverage of the test set from 63 % to 68 %, but the history before it has not been read again. Until that backfill has run, the bar and `rank_points_diff` (B2) still lack those players. *Where:* `training/dataset.py` (`covered_comparison`, `rated_by_provider`), `features/sources.py` (`_missing_player_series`), `features/report.py`.
- **A1 · Rankings count as "published" at different times in backfill and live — Partly.** *Done:* the join time no longer depends on when a snapshot was fetched. `publication_times` dates every snapshot at its label date plus `features.rankings_publication_lag_days`, in the one module training and inference share, so live rows and backfilled rows follow the same rule (`0c69806`). The features are rebuilt from the raw store on every run, so the rule reached all history without rewriting the store. Tests: the same snapshot stored as backfilled and as seen live gives identical feature rows. *Open:* the lag is still the assumed seven days. As reviewer A suggested, the store keeps the time a live run first saw each snapshot, and every feature run lists these delays and warns if one exceeds the lag. So far: the rankings dated 09.11 and 16.11 were both there at the daily run on their own Monday (12:51 and 12:44 UTC), and the one dated 02.11 at the first real run on 04.11. That points to a delay of under a day, so seven days makes every match use rankings a week older than it has to; I lower the lag once two more weeks confirm it. *Where:* `common/features.py`, `features/report.py` (`ranking_delays`), `config/settings.yaml`.
- **E2 · Course material and received reviews in the public repository — Partly.** `docs/Guidelines/` and `docs/reviews/` are out of the tree and git-ignored (`6adcf20`), and `main` and the tags `ms1` and `ms2` were rewritten without them on 21.11; the files are kept locally. The rewrite changed the commit hashes from 01.10 onwards, so hashes cited in the MS2 summary no longer resolve on `main`. *Not done, on purpose:* the merged feature branches and pull request #1 still show the files. I left them as they are: they document how the project was built, and rewriting eleven merged branches would break more references than it protects.
- **A3 · An unranked player gets 0 points — Rejected, after trying it.** I implemented the proposal (`c48a8e8`: `rank_points_diff` missing unless all four players have points) and ran it on the live store on 22.11. On the same 1,311 test matches the log loss went from 0.492 to 0.518, and from 0.495 to 0.501 on the matches the provider rates; champion v2, trained on the 0, dropped from 0.493 to 0.520. After the revert the same run gave 0.492 again. The reason: the feature is one difference per match, so a missing value cannot say *which* pair is the unknown one, while a 0 keeps the direction, and a pair outside the ranking lists is nearly always the weaker one. That is the signal the model lost. I reverted it (`c914878`) and recorded the decision at the fill, in the README and in a test. The explicit flag would separate "not listed" from "listed with 0 points", but as a seventh model input for a case that the per-player ranking reads (E1) keep shrinking; I did not add it. *Where:* `common/features.py` (`features_for`), `tests/test_features.py`.
- **B1 · The first trained model becomes champion without beating anything — Changed.** An empty registry starts with the provider's Elo as champion, and a trained model replaces it only by beating it on the rated test matches (`4b3eda3`, `5ea6cb8`). My live registry already held trained champions, so that alone changed nothing there: every candidate that beats a trained champion now also has to beat the provider's Elo on the rated matches (`304b4d5`). The champion in place, v2, was promoted before this rule; it stays until a candidate passes both comparisons, because demoting it would serve 0.5 for the third of matches the provider does not rate. *Where:* `training/pipeline.py`, `training/baseline.py`.
- **E3 · Let the real schedule accumulate — Partly.** The feature workflow no longer falls back: without `PADEL_SOURCE`, or with a store that is not hosted, its first step fails the run (`2b92ab7`). The README has a badge per scheduled workflow. The cron has run on real data into Hopsworks every day since 05.11 (17 green runs by 21.11, each ending with the data report; GitHub starts the 04:15 UTC cron around 12:30). *Open:* that is two and a half weeks; by the MS3 deadline it is four. *Where:* `.github/workflows/feature-pipeline.yml`, README.
