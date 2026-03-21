# Phase 4.1 Deployment Notes

## Train

```bash
python phases/phase_4_1/phase41_train_pipeline.py --input phases/phase_4/results/phase4_dataset.csv --results-dir phases/phase_4_1/results --artifacts-dir phases/phase_4_1/artifacts --drop-no-result
```

## Start API

```bash
cd phases/phase_4_1
uvicorn phase41_fastapi_app:app --host 0.0.0.0 --port 8003 --reload
```

## Endpoints
- `GET /health`
- `GET /meta`
- `POST /predict`
