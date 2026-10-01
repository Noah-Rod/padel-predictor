# Project Proposal — Padel Predictor: pre-match win probabilities for the Premier Padel / FIP tour

**Noah Rodriguez** · [github.com/Noah-Rod/padel-predictor](https://github.com/Noah-Rod/padel-predictor) (public)

## 1 Problem statement

For every scheduled **main-draw match** on the **Premier Padel** tour (Major, P1, P2, Finals) and, where the feed covers them, the **FIP Platinum and Gold** tiers, men's and women's draws, predict the **probability that team A beats team B**, for **padel enthusiasts** who want to know who is favoured, and by how much, before the day's matches start.

Horizon: **every morning**, the system predicts every upcoming match whose two pairs are already known. First-round matches are known from the day the draw is published (up to **seven days ahead**); later rounds are known once the previous round is decided, usually the day before. Until a match starts, its prediction is refreshed every morning with the latest Elo, form and rest. Evaluation and monitoring score **only the last prediction made before the match starts**.

Scope: the 2026/27 season live, backfilled with completed draws from January 2023. Qualifying rounds and Silver, Rise and Promotion events are excluded (too many players with no history).

Success criterion: on an out-of-time test set (the most recent 20 % of matches, cut at a tournament boundary), the model's **log loss must be lower than the baseline's**. The baseline is **the data provider's own Elo**: padelapi.org publishes a per-player `elo` and the win probability P(A) = 1 / (1 + 10^((R_B − R_A)/400)), which it reports at 71.8 % accuracy and 0.517 log loss on 7 500 matches. If the model only matches the provider's rating, it has added nothing.

The criterion defines success; it never blocks serving. The provider's Elo is registered as the first champion, so predictions are live from day one. A trained candidate replaces the champion only if its log loss on the same test set is lower; if none ever does, the system keeps serving the provider's Elo.

## 2 Originality & motivation

I play padel and follow the Premier Padel tour. The useful version of this system is the one I would open on a Wednesday to see who is favoured in each of Thursday's round-of-16 matches.

Checked against mlops-lab.ch (FS26: 15 projects, three sport-performance forecasters for golf, high jump and kiteboarding, no head-to-head match predictor) and the KTH ID2223 showcases 2022/23 to 2025/26 (134 projects, eleven sports or game predictors: football ×3, Fantasy Premier League, NHL ×2, chess, Clash Royale, League of Legends ×2, transfer values). None covers padel, tennis or any doubles sport. What makes this different from the tennis predictors it resembles, and from the provider's own Elo: **padel is doubles-only and pairs split and re-form every few months, so there is no stable team entity to rate.** A rating answers "how strong are these four players"; this system asks what the rating misses: how long the pair has played together (`partnership`), current form, rest, tier and the official ranking as it stood on match day. The bar is set by the provider's Elo, which already averages two players into a pair, so the project only succeeds if those signals carry information beyond a rating. The MLOps-side twist is a **point-in-time ranking join**: rankings and Elo are re-published weekly, and attaching today's values to a 2024 match is the leak most sports predictors quietly contain.

## 3 Data source & features

**Source.** [padelapi.org](https://padelapi.org) (REST, JSON, bearer token; run by Fantasy Padel Tour): seasons → tournaments → matches, plus players and rankings, with consistent player IDs across Premier Padel and the FIP Tour. Matches are read per tournament with an `updated_after` filter; each carries `status`, `draw` (main / qualy), `scheduled_at`, round and the four player IDs. I have subscribed to the paid tier: the free tier only has the last six months of results and the current ranking, while the paid tier adds the full archive (complete draws for Premier Padel and FIP from 2023) and the **ranking-history endpoint** (weekly snapshots of official points and provider Elo per player). A daily sync of the running tournaments costs under 50 requests, well inside even the free tier's 2 000/day.

**Update frequency and volume.** The feed updates during tournaments, which usually run Wednesday to Sunday. Estimated 3 000 to 4 000 labelled main-draw matches per season across both categories, roughly 10 000 since 2023, growing by 60 to 120 per tournament week. Should the archive prove shallower, history starts six months before the poller and the test set grows with it: this degrades the numbers, not the system.

**Outages.** The data comes from an official API, not scraping, so no second source is needed. If padelapi.org is down, the next daily run catches up through `updated_after` and the UI keeps serving the last predictions.

**Label.** `team_a_won` ∈ {0, 1}, from the source's winner field, for matches with `status = finished` only; `walkover`, `retired` and `bye` are excluded from training and evaluation. Team A / B is assigned by a hash of the pair names, never by seeding, so the label is balanced by construction (≈ 50/50) and no rare-class handling is needed. Nothing in the feature row is derived from the result: set scores, duration and status are used for filtering only and never joined into features.

**Initial features** (team A minus team B unless stated), each computed from state strictly before the match: `elo_diff` (in-house player Elo, replayed match by match), `provider_elo_diff` (padelapi.org Elo from the last weekly snapshot before the match), `rank_points_diff` (official FIP points, same as-of rule), `form_diff` (recent win rate per player), `rest_days_diff`, `partnership_diff` (matches the pair has played together), `tier_level`, `round`. The set is extended after MS2 once ablations show what carries signal; candidates are head-to-head record, margin-weighted Elo and days into the tournament.

**Leakage controls.** Out-of-time split, never inside a tournament. Elo, form, rest and partnership state is updated only *after* a match's feature row is built; a unit test asserts that flipping a match's winner does not change its own features. Rankings and provider Elo are joined as-of the match date, never the current values. Re-ingested matches are de-duplicated on `match_id`, keeping the newest record, so a fixture that later completes becomes one labelled row.

## 4 System design

![FTI architecture](architecture.svg)

Planned stack, revisited at each milestone with the review feedback. Three pipelines share one feature-definition module, `src/padel_predictor/common/features.py`, so training and inference cannot drift.

| Layer | Choice | Why |
| --- | --- | --- |
| Feature store | **Hopsworks** (serverless) | Feature groups with event time and a feature view for as-of joins; hosted, so the data survives between GitHub Actions runs, which start empty every time. |
| Experiment tracking & registry | **Weights & Biases** | Runs, metrics, model versions with a `champion` alias and the weekly monitoring charts in one hosted tool. |
| Orchestration | **GitHub Actions** (cron + `workflow_dispatch`) | Feature daily 04:15 UTC, training weekly Monday, inference daily 06:00 UTC; nothing to host. |
| Serving | **Google Cloud Run** | Streamlit UI reading the daily predictions table; scales to zero. |
| Monitoring | Inference pipeline → W&B | Rolling weekly log loss and accuracy on newly finished matches (last pre-match prediction each), shown in the UI. |

**Stretch, explicitly optional, only after the core is live:** FastAPI `/predict` for ad-hoc matchups; Evidently drift reports on `elo_diff` and `rank_points_diff`; a Grafana dashboard; Terraform for the bucket and the Cloud Run service. The repository is public, CI runs lint and tests on every push, and milestones are tagged `ms1` to `ms4`.
