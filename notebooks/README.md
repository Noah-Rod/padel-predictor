# notebooks/

Exploration only — data checks, feature sanity plots, model comparisons.

**Pipelines never import from here.** Anything a pipeline needs belongs in
`src/padel_predictor/`, where it is covered by tests and reviewed like the rest
of the code. A notebook that has become load-bearing is a bug: move the code
into the package and have the notebook import it.

Clear outputs before committing, so diffs stay readable:

```bash
uv run jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```
