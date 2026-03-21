# Phase 4.1 Deployment Notes

## Train

```bash
python "Phase 4.1/phase41_train_pipeline.py" --input "Phase 4/results/phase4_dataset.csv" --results-dir "Phase 4.1/results" --artifacts-dir "Phase 4.1/artifacts" --drop-no-result
```

## Start API

```bash
cd "Phase 4.1"
uvicorn phase41_fastapi_app:app --host 0.0.0.0 --port 8003 --reload
```

## Endpoints
- `GET /health`
- `GET /meta`
- `POST /predict`
