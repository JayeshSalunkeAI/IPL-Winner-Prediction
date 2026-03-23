from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

import joblib
import pandas as pd


class ModelRuntime:
    def __init__(self, model_path: Path):
        repo_root = model_path.parents[3]
        phase5_module_dir = repo_root / "phases" / "phase_5"
        if str(phase5_module_dir) not in sys.path:
            sys.path.insert(0, str(phase5_module_dir))

        bundle = joblib.load(model_path)
        self.bundle = bundle

        if "model_pipeline" in bundle:
            self.model = bundle["model_pipeline"]
        elif "model" in bundle and isinstance(bundle["model"], dict):
            model_info = bundle["model"]
            if model_info.get("kind") == "single_pipeline":
                self.model = model_info["pipeline"]
            else:
                raise ValueError("Seed ensemble artifacts are not yet supported in this runtime")
        else:
            raise ValueError("Unsupported model artifact format")

        self.label_encoder = bundle["label_encoder"]
        self.feature_columns = bundle["feature_columns"]

    def predict(self, feature_row: dict[str, Any]) -> tuple[str, float, dict[str, float]]:
        x = pd.DataFrame([{c: feature_row.get(c, 0.0) for c in self.feature_columns}])
        proba = self.model.predict_proba(x)[0]
        pred_idx = int(proba.argmax())
        pred_label = str(self.label_encoder.inverse_transform([pred_idx])[0])
        confidence = float(proba[pred_idx])

        probs = {
            str(cls): float(p)
            for cls, p in zip(self.label_encoder.classes_.tolist(), proba.tolist())
        }
        return pred_label, confidence, probs
