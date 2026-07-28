"""
FastAPI backend for the CMAPSS predictive-maintenance pipeline.

This file does NOT reimplement any model or feature-engineering logic — it
only wires HTTP requests to the modularized files:

    upload  -> data_loader.dataset_file_to_dataframe()
    score   -> inference.score_dataset()
    explain -> explainability.get_builtin_importances()

Run:
    pip install -r requirements.txt
    python train_pipeline.py          # once, to produce ./artifacts/
    uvicorn main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import inference
from data_loader import dataset_file_to_dataframe
from explainability import get_builtin_importances

# ---------------------------------------------------------------------------
# Artifacts (model, scaler, feature list) are loaded ONCE at startup, not on
# every request — training already happened offline via train_pipeline.py.
# ---------------------------------------------------------------------------
artifacts: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        artifacts.update(inference.load_artifacts())
        print(f"Model loaded — {len(artifacts['feature_cols'])} feature columns, "
              f"{len(artifacts['keep_sensors'])} sensors.")
    except FileNotFoundError as e:
        print(f'WARNING: could not load artifacts ({e}).')
        print('Run "python train_pipeline.py" first, then restart this API.')
    yield
    artifacts.clear()


app = FastAPI(title='CMAPSS Predictive Maintenance API', lifespan=lifespan)

# The frontend will likely run on a different origin (e.g. localhost:3000)
# during development. Narrow allow_origins to your actual frontend's URL
# before deploying anywhere real.
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


def _ensure_model_loaded():
    if not artifacts:
        raise HTTPException(
            status_code=503,
            detail='Model artifacts are not loaded. Run train_pipeline.py, then restart the API.',
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get('/api/health')
def health():
    """Frontend can poll this to check the API is up and the model is ready."""
    return {'status': 'ok', 'model_loaded': bool(artifacts)}


@app.get('/api/model-info')
def model_info():
    """Metadata the frontend can use to build its UI (which sensors matter, etc.)."""
    _ensure_model_loaded()
    return {
        'feature_columns': artifacts['feature_cols'],
        'sensors_used': artifacts['keep_sensors'],
    }


@app.post('/api/predict-dataset')
async def predict_dataset(file: UploadFile = File(...)):
    """
    The core endpoint: the user selects a dataset file on their own PC
    (raw CMAPSS .txt, or a .csv with the same columns), the frontend uploads
    it here, and this returns predicted RUL, risk level, current cycle count,
    and a per-cycle RUL history (for a trend chart) — one entry per engine
    found in the file.
    """
    _ensure_model_loaded()

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail='Uploaded file is empty.')

    try:
        df = dataset_file_to_dataframe(raw_bytes, filename=file.filename or '')
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        results = inference.score_dataset(df, artifacts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Prediction failed: {e}')

    return {'engine_count': len(results), 'engines': results}


@app.get('/api/feature-importance')
def feature_importance(top_n: int = 10):
    """
    Global feature importance from the trained model itself — needs no new
    data, so it's available as soon as the model is loaded.
    """
    _ensure_model_loaded()
    importances = get_builtin_importances(artifacts['model'], artifacts['feature_cols'])
    top = importances.head(top_n)
    return {'features': [{'feature': k, 'importance': float(v)} for k, v in top.items()]}
