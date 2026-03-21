# Phase 3 Deployment Notes

## Train Phase 3

```bash
python phase3_train_pipeline.py
```

## Start Phase 3 API

```bash
uvicorn phase3_fastapi_app:app --host 0.0.0.0 --port 8001 --reload
```

## Endpoints

- `GET /health`
- `GET /meta`
- `POST /predict`

## Quick API Smoke Test

Use `phase_3_result/phase3_sample_prediction_payload.json` as payload body under the `features` field.
