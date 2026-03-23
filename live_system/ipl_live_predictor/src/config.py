from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    db_path: Path
    model_path: Path
    dummy_feed_path: Path
    api_base_url: str


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]

settings = Settings(
    root_dir=PROJECT_ROOT,
    db_path=PROJECT_ROOT / "state" / "live_state.db",
    model_path=REPO_ROOT / "phases" / "phase_5" / "artifacts" / "phase5_ipl_winner_best_pipeline.joblib",
    dummy_feed_path=PROJECT_ROOT / "data" / "dummy_matches.json",
    api_base_url=os.getenv("IPL_API_BASE_URL", ""),
)
