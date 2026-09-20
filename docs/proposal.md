# Project Proposal — Padel Predictor: pre-match win probabilities for the Premier Padel / FIP tour

**Noah Rodriguez** · [github.com/Noah-Rod/padel-predictor](https://github.com/Noah-Rod/padel-predictor) (public)

## 1 Problem statement

For every scheduled **main-draw match** on the **Premier Padel** tour and the **FIP Platinum and Gold** tiers, men's and women's draws, predict the **probability that team A beats team B**. Predictions are produced **daily for all fixtures up to seven days ahead**, i.e. from the moment a draw is published until the match starts, and are served to **fans and fantasy-league players** who fill in brackets before play begins.

Scope: the 2026/27 season live, backfilled with completed draws from January 2023. Qualifying rounds and Silver, Rise and Promotion events are excluded: too many players with no history.

Success criterion, on an out-of-time test set (the most recent 20 % of matches, cut at a tournament boundary):

- **log loss lower than the Elo baseline**, i.e. the Elo expectation P(A) = 1 / (1 + 10^((R_B − R_A)/400)) computed from the same replayed player ratings; and
- **accuracy ≥ 68 %, and higher than the rank-favourite baseline** (the team with more FIP ranking points wins).

A candidate that fails either check is not promoted; the current champion keeps serving.

## 2 Originality & motivation

I play padel and follow the Premier Padel tour. The useful version of this system is the one I would open on a Wednesday to pick a bracket before Thursday's round of 16.

Checked against the mlops-lab.ch project lists and the KTH ID2223 showcases 2022/23 to 2025/26 (134 projects): eleven are sports or game predictors (football ×3, Fantasy Premier League, NHL ×2, chess, Clash Royale, League of Legends ×2, transfer values) and none covers padel, tennis or any doubles sport. What makes this different from the tennis predictors it resembles: **padel is doubles-only and pairs split and re-form every few months, so there is no stable team entity to rate.** The system rates individual players (Elo replayed match by match, team = mean of its two players) and adds an explicit **partnership** feature (matches this exact pair has played together), so a new pairing of two strong players is neither "unknown" nor an established team. The MLOps-side twist is a **point-in-time ranking join**: the FIP ranking is re-published weekly, and attaching today's ranking to a 2024 match is the leak most sports predictors quietly contain.

## 3 Data source & features

**Source.** [padelapi.org](https://padelapi.org) (REST, JSON, API key; run by Fantasy Padel Tour): tournaments → draws → matches → players, plus rankings, with consistent player IDs across Premier Padel and the FIP Tour. Free tier: 50 000 requests/month, 10/min; schedules, results of the last six months and the current ranking. The paid tier adds the full archive (complete draws for Premier Padel and FIP from 2023) and ranking history; I subscribe for the backfill. A daily 7-day window costs under 50 requests, so the live pipeline stays inside the free quota.

**Update frequency and volume.** The feed updates during tournaments, which run Wednesday to Sunday almost every week of the season. Estimated 3 000 to 4 000 labelled main-draw matches per season across both categories, roughly 10 000 since 2023, growing by 60 to 120 per tournament week. Should the archive prove shallower than documented, history starts six months before the poller and the test set grows with it; the proposal is stated so that this degrades the numbers, not the system.

**Fallback.** Scraping the results and rankings pages of padelfip.com (same federation, same player names) behind the same `MatchSource` interface. Its terms of use are checked before any scraping and reported in the MS2 summary. Second line: the last good feature-store snapshot plus the deterministic synthetic source that CI already runs.

**Label.** `team_a_won` ∈ {0, 1} for completed matches, from the source's winner field. Walkovers, retirements and withdrawals are excluded from training and evaluation (they still update rest days). Unplayed fixtures carry a null label and are what inference scores. Team A / B is assigned by a hash of the pair names, never by seeding, so the label is balanced by construction (≈ 50/50) and no rare-class handling is needed; every row is also mirrored (swap teams, negate differences, flip label) to enforce symmetry. Nothing in the feature row is derived from the result: set scores, duration and match status are ingested for filtering only and never joined into features.

**Features** (team A minus team B unless stated), each computed from state strictly before the match:
`elo_diff` (player Elo, K = 24, initial 1500), `rank_points_diff` (FIP points from the last ranking published before the match date), `form_diff` (win rate over each player's last five matches), `rest_days_diff` (days since the team's freshest player last played, capped at 60), `partnership_diff` (matches the pair has played together), `tier_level` (Major 4 … Gold 1), `round`.

**Leakage controls.** Out-of-time split, never inside a tournament. Elo, form, rest and partnership state is updated only *after* a match's feature row is built; a unit test asserts that flipping a match's winner does not change its own features. Rankings are joined as-of the match date, never the current one. Re-ingested matches are de-duplicated on `match_id`, keeping the newest record, so a fixture that later completes becomes one labelled row.

## 4 System design

![FTI architecture](architecture.svg)

Three pipelines share one feature-definition module, `src/padel_predictor/common/features.py`, so training and inference cannot drift.

| Layer | Choice | Why |
| --- | --- | --- |
| Feature store | **Hopsworks** (serverless) | Feature groups with event time and a feature view for as-of joins; survives ephemeral CI runners. Parquet on GCS with the same schema stays as the CI/offline backend. |
| Experiment tracking & registry | **Weights & Biases** | Runs, metrics, model versions with a `champion` alias and the weekly monitoring charts in one hosted tool. |
| Orchestration | **GitHub Actions** (cron + `workflow_dispatch`) | Feature daily 04:15 UTC, training weekly Monday, inference daily 06:00 UTC; nothing to host. |
| Serving | **Google Cloud Run** | Streamlit UI reading the daily predictions table; scales to zero. |
| Monitoring | Inference pipeline → W&B | Rolling weekly log loss and accuracy on newly finished matches, shown in the UI. |

**Stretch, explicitly optional, only after the core is live:** FastAPI `/predict` for ad-hoc matchups; Evidently drift reports on `elo_diff` and `rank_points_diff`; a Grafana dashboard; Terraform for the bucket and the Cloud Run service.

The repository is public; CI runs lint and the test suite on every push; milestones are tagged `ms1` to `ms4`.
