# Phase 4 Deployment Notes

## 1) Build dataset from YAML

```bash
python "Phase 4/phase4_extract_dataset.py" --yaml-dir ipl_male --output "Phase 4/results/phase4_dataset.csv"
```

## 2) Train Phase 4

```bash
python "Phase 4/phase4_train_pipeline.py" --input "Phase 4/results/phase4_dataset.csv" --drop-no-result
```

## 3) Start Phase 4 API

```bash
cd "Phase 4"
uvicorn phase4_fastapi_app:app --host 0.0.0.0 --port 8002 --reload
```

## Endpoints
- `GET /health`
- `GET /meta`
- `POST /predict`

## Payload template
Use `Phase 4/results/phase4_sample_prediction_payload.json`.
