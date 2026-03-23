from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PreMatchPayload:
    match_id: str
    date: str
    venue: str
    team1: str
    team2: str
    toss_winner: str
    toss_decision: str
    team1_players: list[str]
    team2_players: list[str]


@dataclass
class PostMatchPayload:
    match_id: str
    actual_winner: str
    team1_runs: float
    team2_runs: float
    team1_wkts_lost: float
    team2_wkts_lost: float
    team1_balls: float
    team2_balls: float
    team1_powerplay_runs: float
    team2_powerplay_runs: float
    team1_death_runs: float
    team2_death_runs: float
    batting_first: str
    batting_second: str


@dataclass
class PredictionResult:
    predicted_winner: str
    confidence: float
    class_probabilities: dict[str, float]
    feature_row: dict[str, Any]
