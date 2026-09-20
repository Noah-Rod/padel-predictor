# Project conventions

Instructions for anyone — or any tool — contributing to this repository.

## Authorship

- All commits are authored by the repository owner, using their own name and
  email address as configured in `git config user.name` / `user.email`.
- Do not add co-author, generated-by, or session trailers to commit messages.

## Tooling references

- No assistant or AI-tool branding anywhere in the repository: not in commit
  messages, README, `docs/`, code comments, workflow files, or pull request
  descriptions. Write everything in the project's own voice.

## Repository rules (MLOps HS26)

- Milestones are tagged: `git tag ms2 && git push --tags`.
- No secrets in the repo or its history — keys live in `.env` (gitignored) and
  GitHub Actions secrets. A leaked key must be rotated, not just deleted.
- No data, model files or `mlruns/` in git.
- Pipelines never import from `notebooks/`.
- Feature definitions live in exactly one place, `src/padel_predictor/common/features.py`,
  and are used by both training and inference to avoid training–serving skew.
- The repo stays public and reachable for the whole semester.
