# Phase 4 Deployment Notes

## 1) Build dataset from YAML

```bash
python phases/phase_4/phase4_extract_dataset.py --yaml-dir data/raw/ipl_male --output phases/phase_4/results/phase4_dataset.csv
```

## 2) Train Phase 4

```bash
python phases/phase_4/phase4_train_pipeline.py --input phases/phase_4/results/phase4_dataset.csv --drop-no-result
```

## Notes
- Phase 4 API file was removed during cleanup.
- Final production API is maintained in `phases/phase_4_1/phase41_fastapi_app.py`.

## Payload template
Use `phases/phase_4/results/phase4_sample_prediction_payload.json`.
