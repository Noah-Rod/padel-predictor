# 🎾 Padel Predictor

**Predicts the winner of a professional padel match before it is played** — the
probability that team A beats team B, for every fixture up to seven days ahead,
refreshed daily.

Built for HSLU **MLOps HS26 (I.BA_MLOPS)** as three automated pipelines: a
feature pipeline that ingests live results, a training pipeline that retrains
and promotes models, and an inference pipeline that serves predictions.

| | |
| --- | --- |
| 🌐 **Live app** | _Deployed at MS4 — link goes here_ |
| 🎬 **Video pitch** | _Recorded at MS4 — link goes here_ |
| 📄 **Docs** | [`docs/`](docs/) — proposal and milestone summaries |

---

## Architecture

![FTI architecture](docs/architecture.svg)

| Pipeline | Trigger | What it does |
| --- | --- | --- |
| **1. Feature** | daily cron 04:15 UTC, plus backfill on demand | Fetches results and fixtures, rebuilds features, writes a dated snapshot to the feature store |
| **2. Training** | weekly cron, plus on demand | Reads the feature store, trains, evaluates against an Elo baseline, registers and promotes to `@champion` if better |
| **3. Inference** | daily cron 06:00 UTC, plus on demand from the UI | Loads the champion model, predicts upcoming fixtures, stores the predictions |

### What the model sees

Six team-difference features, all replayed in chronological order so a row only
ever sees matches that finished earlier:

`elo_diff` · `rank_points_diff` · `form_diff` · `rest_days_diff` ·
`partnership_diff` · `tier_level`

Two design decisions carry most of the weight:

- **One feature definition.** Training and inference both import
  `build_feature_frame` from [`src/padel_predictor/common/features.py`](src/padel_predictor/common/features.py).
  There is no second implementation anywhere, so training and serving cannot drift apart.
- **No leakage by construction.** A match's own result is folded into the rolling
  state only *after* its feature row is built. A unit test asserts it: flipping a
  match's winner must not change that match's features.

---

## Clone and run

Requires **Python 3.11+** and [uv](https://docs.astral.sh/uv/). No API key is
needed: the default `sample` source generates deterministic synthetic matches
offline, so a fresh clone runs the whole system end to end.

```bash
git clone https://github.com/Noah-Rod/padel-predictor.git
cd padel-predictor
cp .env.example .env          # edit only if you use a live source
uv sync --all-extras

# 1. Feature pipeline: load history, then keep it current
uv run padel backfill --start 2025-09-01 --end 2026-09-01
uv run padel feature-pipeline

# 2. Training pipeline: train, evaluate, promote to @champion
uv run padel training-pipeline

# 3. Inference pipeline: predict the upcoming fixtures
uv run padel inference-pipeline

# Serve it
uv run streamlit run ui/app.py        # UI  on http://localhost:8501
uv run padel serve                    # API on http://localhost:8000/docs
```

A single prediction from the command line:

```bash
uv run padel matchup --team-a "Player 01,Player 02" --team-b "Player 03,Player 04"
```

`make help` lists the same steps as short targets.

### With Docker

```bash
cp .env.example .env
docker compose up --build            # mlflow + api + ui
docker compose run --rm pipelines padel backfill --start 2025-09-01 --end 2026-09-01
docker compose run --rm pipelines padel training-pipeline
```

| Service | URL |
| --- | --- |
| Streamlit UI | http://localhost:8501 |
| API docs | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |

---

## Configuration

Two separate sources, so no secret is ever committed:

| Where | What | Committed |
| --- | --- | --- |
| [`config/settings.yaml`](config/settings.yaml) | Pipeline behaviour: feature parameters, model hyper-parameters, promotion rules | yes |
| `.env` (see [`.env.example`](.env.example)) | Secrets and environment URIs: API keys, feature store, tracking server | **no** |

In GitHub Actions the same names come from repository **variables** (non-secret,
e.g. `PADEL_FEATURE_STORE_URI`) and **secrets** (`PADEL_API_KEY`, `GCP_SA_KEY`).

### Going live

The scheduled workflows run on ephemeral runners, so anything written to local
disk is discarded when the job ends. For the pipelines to actually accumulate
state, set these repository variables:

| Variable | Example |
| --- | --- |
| `PADEL_FEATURE_STORE_URI` | `gs://your-bucket/padel-predictor` |
| `MLFLOW_TRACKING_URI` | `https://your-mlflow-host` |
| `PADEL_SOURCE` | `http` |
| `PADEL_API_BASE_URL` | the live source's base URL |

The feature and training workflows emit a warning when these are still local.

---

## Repository layout

```
padel-predictor/
├── src/padel_predictor/
│   ├── common/        config, storage, schemas, shared feature definitions
│   ├── features/      1. ingestion adapters, feature pipeline, backfill
│   ├── training/      2. dataset split, MLflow registry, training pipeline
│   ├── inference/     3. predictor, batch pipeline, FastAPI app
│   └── cli.py         one command per pipeline
├── ui/                Streamlit front end
├── config/            settings.yaml (no secrets)
├── tests/             unit tests, run in CI on 3.11 and 3.12
├── notebooks/         exploration only — pipelines never import from here
├── docs/              proposal, milestone summaries, architecture diagram
├── .github/workflows/ ci · feature · training · inference · deploy
├── Dockerfile · docker-compose.yml
└── pyproject.toml · uv.lock · requirements.txt
```

## Development

```bash
uv run pytest                  # unit tests
uv run ruff check .            # lint
uv run ruff format .           # format
```

CI runs lint, format check and the test suite on Python 3.11 and 3.12, and
builds the Docker image, on every push and pull request.

## Milestones

Tagged in git as they are reached:

| Tag | Milestone | Due |
| --- | --- | --- |
| `ms1` | Project proposal | 01.10.2026 |
| `ms2` | Feature pipeline | 05.11.2026 |
| `ms3` | Training pipeline | 03.12.2026 |
| `ms4` | Live system + video | 10.01.2027 |

```bash
git tag ms2 && git push --tags
```

## Licence

MIT — see [`LICENSE`](LICENSE).
