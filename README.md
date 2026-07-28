# CMAPSS Predictive Maintenance — Modularized Pipeline

Modularized from the ALICE Club Session 2 (Track B) notebook.

## File map

| File | Role |
|---|---|
| `config.py` | Constants (paths, hyperparameters, column names) |
| `data_loader.py` | Training data loading + engine split (training-only); `dataset_file_to_dataframe` for uploaded files, `records_to_dataframe` for raw JSON rows (both serving-only) |
| `feature_engineering.py` | Sensor selection, rolling features, final feature list |
| `preprocessing.py` | Scaling (fit on train, transform everywhere else) |
| `models.py` | Baseline + 4 model trainers, model save/load |
| `evaluation.py` | Metric scorecard + results table |
| `explainability.py` | Built-in + permutation feature importance |
| `visualization.py` | Notebook/report plots only — **not used by the backend** |
| `risk.py` | RUL → risk level → recommendation (pure logic) |
| `inference.py` | Serving logic — `load_artifacts()`, `run_inference()` (JSON rows), `score_dataset()` (uploaded file, all cycles) |
| `train_pipeline.py` | Offline script: trains everything, saves artifacts to `ARTIFACTS_DIR` |
| `main.py` | **The FastAPI app.** Wires HTTP requests to the modules above — no model logic lives here |

## Run order

1. Put `train_FD001.txt` under `./data/CMAPSSData/` (or set `CMAPSS_DATA_DIR`).
2. `pip install -r requirements.txt`
3. `python train_pipeline.py` — trains all models and writes to `./artifacts/`:
   - `model.joblib`, `scaler.joblib`, `keep_sensors.json`, `feature_cols.json`
4. `uvicorn main:app --reload` — starts the API at `http://127.0.0.1:8000`.

## API endpoints

| Method & path | Purpose |
|---|---|
| `GET /api/health` | Check the API is up and the model loaded successfully |
| `GET /api/model-info` | Feature columns + sensors the model was trained on |
| `POST /api/predict-dataset` | **The main endpoint.** Upload a dataset file (see below) → get RUL, risk, current cycle, and per-cycle history for every engine in it |
| `GET /api/feature-importance?top_n=10` | Global feature importance from the trained model |

Interactive docs (try requests straight from the browser) are auto-served at
`http://127.0.0.1:8000/docs` once the server is running.

### `POST /api/predict-dataset` — request

Multipart form upload, field name `file`. Accepts either:
- the raw NASA CMAPSS format (whitespace-separated `.txt`, no header — same
  as `train_FD001.txt` / `test_FD001.txt`), or
- a `.csv` with a header row using the column names below.

No RUL column is needed or used — predicting RUL is the whole point.

### `POST /api/predict-dataset` — response

```json
{
  "engine_count": 5,
  "engines": [
    {
      "engine_id": 1,
      "current_cycle": 50,
      "predicted_rul": 48.4,
      "risk": "Medium",
      "recommendation": "Schedule a closer check soon.",
      "history": [
        { "cycle": 1, "predicted_rul": 52.3 },
        { "cycle": 2, "predicted_rul": 50.0 }
      ]
    }
  ]
}
```

`history` is every cycle in the uploaded file for that engine (not just the
latest) — enough to draw a degradation trend chart per engine on the
frontend.

## Expected dataset columns

Whichever file format you upload, it needs these columns (see `COL_NAMES`
in `config.py`):

```
unit_number, time_in_cycles, op_setting_1, op_setting_2, op_setting_3,
sensor_01, sensor_02, ... sensor_21
```

No RUL column — that's what's being predicted.

Each `unit_number` needs **at least `ROLL_WINDOW` (8) consecutive cycles**
in the file — rolling features can't be computed from a single row.
`score_dataset()` returns one result per engine, using its most recent
cycle for the headline `predicted_rul` / `risk`, and every cycle for
`history`.

## Notes for the frontend build

- CORS is wide open (`allow_origins=["*"]`) for local development. Narrow
  it to your frontend's actual origin in `main.py` before deploying anywhere.
- The frontend never needs to know about rolling windows, scalers, or
  feature columns — `POST /api/predict-dataset` handles all of that. It only
  needs to send a file and render the JSON response.
- `GET /api/model-info` is there if the frontend wants to show which sensors
  the model actually uses, without hardcoding that list.
