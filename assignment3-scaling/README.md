# CS336 Spring 2026 Assignment 3: Scaling

For a full description of the assignment, see the assignment handout at
[cs336_assignment3_scaling.pdf](./cs336_assignment3_scaling.pdf).

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## For students

Install uv

```sh
uv sync
```

Set `A3_API_KEY` to your 8-digit student ID:

```sh
export A3_API_KEY=06123456
```

The hosted training API is available at:

```text
http://hyperturing.stanford.edu:8000
```

Click here for the [docs](http://hyperturing.stanford.edu:8000/docs) and [dashboard](http://hyperturing.stanford.edu:8000/dashboard).

See [`./examples/client_example.ipynb`](./examples/client_example.ipynb) for an
example of submitting and inspecting training runs.

## Offline scaling tools

This working copy also includes local scaling utilities under
`cs336_scaling/offline/`. They are useful before submitting jobs to the hosted API
or before renting a GPU.

Fit a compute scaling law from the bundled IsoFLOPs fixture:

```sh
uv run python -m scripts.fit_scaling_law --predict-compute 3e20
```

Generate an IsoFLOPs experiment grid:

```sh
uv run python -m scripts.make_scaling_grid \
  --params 1e7 3e7 1e8 \
  --compute 1e17 3e17 1e18
```

Ask for a compute-budget recommendation:

```sh
uv run python -m scripts.recommend_budget --compute-budget 1e18
```

Export a config that the A1 training stack can run:

```sh
uv run python -m scripts.export_a1_config \
  --compute-budget 1e18 \
  --train-bin artifacts/tinystories/train.bin \
  --valid-bin artifacts/tinystories/valid.bin \
  --out-dir runs/scale_1e18 \
  --output ../assignment1-basics/configs/scale_1e18.json
```

The generated specs use the same systems defaults as the A1/A2 training stack:
SDPA attention, autocast mixed precision, `torch.compile`, and fused-optimizer
readiness. Treat the output as a planning artifact, then validate with real runs
and update the fit from observed validation losses.

## For non-students

Install dependencies:

```sh
uv sync --extra server
```

To download tokenized data:

```sh
uv run modal run scripts/1_download_tokenized_data.py
```

To run training directly:

```sh
uv run cs336_scaling/training/run.py
```

To run the API and dispatcher, set:

```sh
DATABASE_URL_PROD="postgresql://..."
DATABASE_URL_DEV="postgresql://..."
INTERNAL_API_KEY="SOMEKEY"
```

Then run:

```sh
DB_ENV=prod uv run fastapi run &
DB_ENV=prod uv run dispatcher &
```
