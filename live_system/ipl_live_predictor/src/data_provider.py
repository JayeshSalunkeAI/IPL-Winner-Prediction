from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DummyAPIProvider:
    def __init__(self, dummy_feed_path: Path):
        self.dummy_feed_path = dummy_feed_path

    def fetch(self) -> dict[str, Any]:
        if not self.dummy_feed_path.exists():
            return {"matches": []}
        with self.dummy_feed_path.open("r", encoding="utf-8") as f:
            return json.load(f)
