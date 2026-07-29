# Fly High — Frontend (Streamlit)

Streamlit console for the CMAPSS Predictive Maintenance FastAPI backend
(`Rezastra/Fly_high`).

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Start the backend first (from the backend repo):

```bash
uvicorn main:app --reload
```

Then, in this folder:

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). Set the
"API base URL" in the sidebar if your backend isn't on the default
`http://127.0.0.1:8000`.

## Files

| File | Purpose |
|---|---|
| `app.py` | The whole Streamlit app |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Dark theme so native widgets match the custom CSS |