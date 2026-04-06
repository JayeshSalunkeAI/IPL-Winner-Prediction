import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import joblib
import pandas as pd
import urllib.request
from ops_db import (
    default_db_path,
    fetch_recent_team_form,
    fetch_recent_venue_stats,
    init_db,
    save_prediction,
)


API_BASE = "https://api.cricapi.com/v1"


TEAM_ALIASES = {
    "RCB": "Royal Challengers Bengaluru",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "PBKS": "Punjab Kings",
    "KXIP": "Punjab Kings",
    "DC": "Delhi Capitals",
    "DD": "Delhi Capitals",
    "MI": "Mumbai Indians",
    "KKR": "Kolkata Knight Riders",
    "CSK": "Chennai Super Kings",
    "GT": "Gujarat Titans",
    "LSG": "Lucknow Super Giants",
    "SRH": "Sunrisers Hyderabad",
    "RR": "Rajasthan Royals",
}


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_gmt_datetime(dt_text: str) -> Optional[datetime]:
    if not dt_text:
        return None
    text = dt_text.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def normalize_team_name(name: str) -> str:
    if not name:
        return name
    n = name.strip()
    return TEAM_ALIASES.get(n, n)


def normalize_toss_choice(choice: str) -> str:
    if not choice:
        return "field"
    c = choice.strip().lower()
    if c in {"bowl", "field", "chase"}:
        return "field"
    if c in {"bat", "batting"}:
        return "bat"
    return "field"


def normalize_player_key(name: str) -> str:
    if not name:
        return ""
    return "".join(ch.lower() for ch in name if ch.isalnum())


def http_get_json(url: str) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


class CricApiClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _url(self, endpoint: str, **params: Any) -> str:
        query = {"apikey": self.api_key}
        for k, v in params.items():
            if v is None:
                continue
            query[k] = v
        return f"{API_BASE}/{endpoint}?{urlencode(query)}"

    def series(self, search: str) -> Dict[str, Any]:
        return self._checked(http_get_json(self._url("series", offset=0, search=search)), "series")

    def series_info(self, series_id: str) -> Dict[str, Any]:
        return self._checked(http_get_json(self._url("series_info", id=series_id)), "series_info")

    def match_info(self, match_id: str) -> Dict[str, Any]:
        return self._checked(http_get_json(self._url("match_info", id=match_id)), "match_info")

    def match_squad(self, match_id: str) -> Dict[str, Any]:
        return self._checked(http_get_json(self._url("match_squad", id=match_id)), "match_squad")

    @staticmethod
    def _checked(payload: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
        status = payload.get("status")
        if status == "failure":
            reason = payload.get("reason", "Unknown API failure")
            raise RuntimeError(f"API {endpoint} failed: {reason}")
        return payload


@dataclass
class MatchCandidate:
    match_id: str
    name: str
    team1: str
    team2: str
    venue: str
    start_dt_utc: Optional[datetime]
    started: bool
    ended: bool


class HistoricalFeatureBuilder:
    def __init__(self, training_csv: Path):
        self.df = pd.read_csv(training_csv)
        self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce", utc=True)
        self.df = self.df.sort_values("date")

        self.team_names = sorted(self.df["team1"].dropna().unique().tolist())
        self.stadium_names = sorted(self.df["stadium"].dropna().unique().tolist())

        self.team_profiles = self._build_team_profiles()
        self.venue_profiles = self._build_venue_profiles()

    def _winner_series(self) -> pd.Series:
        # target=1 means team1 won; target=0 means team2 won
        return self.df.apply(
            lambda r: r["team1"] if int(r["target"]) == 1 else r["team2"], axis=1
        )

    def _build_team_profiles(self) -> Dict[str, Dict[str, float]]:
        rows = []
        for _, r in self.df.iterrows():
            rows.append(
                {
                    "team": r["team1"],
                    "bat_avg": r["t1_bat_avg"],
                    "bat_sr": r["t1_bat_sr"],
                    "bowl_eco": r["t1_bowl_eco"],
                    "bowl_sr": r["t1_bowl_sr"],
                }
            )
            rows.append(
                {
                    "team": r["team2"],
                    "bat_avg": r["t2_bat_avg"],
                    "bat_sr": r["t2_bat_sr"],
                    "bowl_eco": r["t2_bowl_eco"],
                    "bowl_sr": r["t2_bowl_sr"],
                }
            )
        tdf = pd.DataFrame(rows)
        grouped = tdf.groupby("team", dropna=True).mean(numeric_only=True)
        out = {}
        for team, row in grouped.iterrows():
            out[str(team)] = {
                "bat_avg": float(row["bat_avg"]),
                "bat_sr": float(row["bat_sr"]),
                "bowl_eco": float(row["bowl_eco"]),
                "bowl_sr": float(row["bowl_sr"]),
            }
        return out

    def _build_venue_profiles(self) -> Dict[str, Dict[str, Any]]:
        winners = self._winner_series()
        tmp = self.df.copy()
        tmp["winner"] = winners
        out = {}

        for venue, g in tmp.groupby("stadium", dropna=True):
            toss_adv = (g["toss_winner"] == g["winner"]).mean() if len(g) else 0.5
            pitch_mode = "Batting"
            bounce_mode = "Medium"
            if "pitch_type" in g and g["pitch_type"].notna().any():
                pitch_mode = str(g["pitch_type"].mode().iloc[0])
            if "bounce_and_carry" in g and g["bounce_and_carry"].notna().any():
                bounce_mode = str(g["bounce_and_carry"].mode().iloc[0])

            out[str(venue)] = {
                "venue_score_prior": float(g["venue_score_prior"].mean()),
                "venue_chase_winrate_prior": float(g["venue_chase_winrate_prior"].mean()),
                "toss_advantage": float(toss_adv),
                "pitch_type": pitch_mode,
                "bounce_and_carry": bounce_mode,
            }
        return out

    def map_team(self, name: str) -> str:
        norm = normalize_team_name(name)
        if norm in self.team_names:
            return norm
        candidates = get_close_matches(norm, self.team_names, n=1, cutoff=0.6)
        return candidates[0] if candidates else norm

    def map_stadium(self, venue: str) -> str:
        if venue in self.stadium_names:
            return venue
        candidates = get_close_matches(venue, self.stadium_names, n=1, cutoff=0.45)
        return candidates[0] if candidates else venue

    def recent_form(self, team: str, ref_dt: Optional[datetime], n: int = 5) -> float:
        t = self.map_team(team)
        data = self.df.copy()
        if ref_dt is not None:
            data = data[data["date"] < ref_dt]

        if data.empty:
            return 0.5

        winners = data.apply(
            lambda r: r["team1"] if int(r["target"]) == 1 else r["team2"], axis=1
        )
        played = data[(data["team1"] == t) | (data["team2"] == t)].tail(n)
        if played.empty:
            return 0.5

        played_winners = winners.loc[played.index]
        return float((played_winners == t).mean())

    def team_profile(self, team: str) -> Dict[str, float]:
        t = self.map_team(team)
        return self.team_profiles.get(
            t,
            {"bat_avg": 30.0, "bat_sr": 140.0, "bowl_eco": 8.5, "bowl_sr": 20.0},
        )

    def venue_profile(self, venue: str) -> Dict[str, Any]:
        v = self.map_stadium(venue)
        return self.venue_profiles.get(
            v,
            {
                "venue_score_prior": 170.0,
                "venue_chase_winrate_prior": 0.52,
                "toss_advantage": 0.5,
                "pitch_type": "Batting",
                "bounce_and_carry": "Medium",
            },
        )


class AutoPredictor:
    def __init__(self, base_dir: Path, api_key: str):
        self.base_dir = base_dir
        self.api = CricApiClient(api_key)
        self.model = joblib.load(base_dir / "model" / "model.joblib")
        self.builder = HistoricalFeatureBuilder(base_dir / "data" / "training_data.csv")
        self.db_path = default_db_path(base_dir)
        init_db(self.db_path)
        self.squad_players_by_team = self._load_local_squads(base_dir / "data" / "squads_2026.csv")

    def _load_local_squads(self, squads_csv: Path) -> Dict[str, set]:
        out: Dict[str, set] = {}
        if not squads_csv.exists():
            return out
        try:
            df = pd.read_csv(squads_csv)
            if not {"team", "player"}.issubset(set(df.columns)):
                return out
            for _, row in df.iterrows():
                t = self.builder.map_team(normalize_team_name(str(row["team"])))
                p = normalize_player_key(str(row["player"]))
                if not p:
                    continue
                out.setdefault(t, set()).add(p)
        except Exception:
            return {}
        return out

    def _role_scores(self, role_text: str) -> Tuple[float, float]:
        role = (role_text or "").lower()
        if "allround" in role:
            return 0.9, 0.9
        if "bowl" in role:
            return 0.35, 1.05
        if "wk" in role or "keeper" in role:
            return 0.95, 0.25
        if "bat" in role:
            return 1.0, 0.25
        return 0.7, 0.7

    def _player_multiplier_from_squad(
        self, team_name: str, squad_payload: Optional[List[Dict[str, Any]]]
    ) -> Tuple[float, float, int]:
        if not squad_payload:
            return 1.0, 1.0, 0

        target = self.builder.map_team(normalize_team_name(team_name))
        target_norm = normalize_team_name(team_name).lower()

        team_obj = None
        for entry in squad_payload:
            api_team = normalize_team_name(str(entry.get("teamName", "")))
            if api_team == target or api_team.lower() == target_norm:
                team_obj = entry
                break

        if team_obj is None:
            return 1.0, 1.0, 0

        players = team_obj.get("players") if isinstance(team_obj, dict) else []
        if not isinstance(players, list) or not players:
            return 1.0, 1.0, 0

        # API often returns full squad pre-match. We cap at first 11 to approximate announced XI.
        picked = players[:11]
        local_set = self.squad_players_by_team.get(target, set())

        bat_scores: List[float] = []
        bowl_scores: List[float] = []
        for p in picked:
            role = str(p.get("role", ""))
            name = str(p.get("name", ""))
            bat_s, bowl_s = self._role_scores(role)

            # Small confidence bump if this player exists in local squad file.
            key = normalize_player_key(name)
            if key and key in local_set:
                bat_s += 0.03
                bowl_s += 0.03

            bat_scores.append(bat_s)
            bowl_scores.append(bowl_s)

        if not bat_scores or not bowl_scores:
            return 1.0, 1.0, 0

        # Convert average role score to safe multipliers (bounded to avoid model drift).
        bat_avg = sum(bat_scores) / len(bat_scores)
        bowl_avg = sum(bowl_scores) / len(bowl_scores)

        bat_mult = min(1.12, max(0.88, bat_avg / 0.8))
        bowl_mult = min(1.12, max(0.88, bowl_avg / 0.8))

        return float(bat_mult), float(bowl_mult), len(picked)

    def select_match(
        self,
        series_search: str,
        match_id: Optional[str],
        team1_filter: Optional[str],
        team2_filter: Optional[str],
    ) -> MatchCandidate:
        if match_id:
            m = self.api.match_info(match_id)
            return self._to_candidate(m.get("data", {}))

        sr = self.api.series(series_search)
        if sr.get("status") != "success" or not sr.get("data"):
            raise RuntimeError("No series found from API for the given search text.")

        series_id = sr["data"][0]["id"]
        si = self.api.series_info(series_id)
        if si.get("status") != "success":
            raise RuntimeError("Failed to fetch series_info from API.")

        match_list = si.get("data", {}).get("matchList", [])
        candidates = [self._to_candidate(m) for m in match_list if isinstance(m, dict)]
        candidates = [c for c in candidates if c.match_id]

        if team1_filter or team2_filter:
            t1f = normalize_team_name(team1_filter) if team1_filter else None
            t2f = normalize_team_name(team2_filter) if team2_filter else None
            keep = []
            for c in candidates:
                teams = {normalize_team_name(c.team1), normalize_team_name(c.team2)}
                if t1f and t1f not in teams:
                    continue
                if t2f and t2f not in teams:
                    continue
                keep.append(c)
            candidates = keep

        upcoming = [c for c in candidates if not c.ended]
        if not upcoming:
            raise RuntimeError("No active/upcoming match found with current filters.")

        # Prefer closest future match; if already started but not ended, pick that first.
        current = now_utc()
        in_progress = [c for c in upcoming if c.started and not c.ended]
        if in_progress:
            return sorted(in_progress, key=lambda c: (c.start_dt_utc or current))[0]

        future = [c for c in upcoming if c.start_dt_utc and c.start_dt_utc >= current]
        if future:
            return sorted(future, key=lambda c: c.start_dt_utc)[0]

        return sorted(upcoming, key=lambda c: (c.start_dt_utc or current))[-1]

    def _to_candidate(self, raw: Dict[str, Any]) -> MatchCandidate:
        teams = raw.get("teams") or ["", ""]
        t1 = teams[0] if len(teams) > 0 else ""
        t2 = teams[1] if len(teams) > 1 else ""
        return MatchCandidate(
            match_id=raw.get("id", ""),
            name=raw.get("name", ""),
            team1=t1,
            team2=t2,
            venue=raw.get("venue", "Unknown"),
            start_dt_utc=parse_gmt_datetime(raw.get("dateTimeGMT", "")),
            started=bool(raw.get("matchStarted", False)),
            ended=bool(raw.get("matchEnded", False)),
        )

    def wait_and_predict(
        self,
        candidate: MatchCandidate,
        min_before: int,
        max_before: int,
        poll_seconds: int,
        timeout_minutes: int,
        require_squad: bool,
    ) -> Dict[str, Any]:
        deadline = time.time() + max(timeout_minutes, 1) * 60
        squad_checked = False
        squad_ok = False
        squad_payload: Optional[List[Dict[str, Any]]] = None
        last_match = None

        while time.time() < deadline:
            mi = self.api.match_info(candidate.match_id)
            if mi.get("status") != "success":
                print("match_info failed; retrying...")
                time.sleep(poll_seconds)
                continue

            m = mi.get("data", {})
            last_match = m

            start_dt = parse_gmt_datetime(m.get("dateTimeGMT", ""))
            mins_to_start = None
            if start_dt:
                mins_to_start = (start_dt - now_utc()).total_seconds() / 60.0

            toss_winner = normalize_team_name(m.get("tossWinner", ""))
            toss_choice = normalize_toss_choice(m.get("tossChoice", ""))
            toss_ready = bool(toss_winner)

            in_window = False
            if mins_to_start is None:
                in_window = toss_ready
            else:
                in_window = min_before <= mins_to_start <= max_before

            status_line = (
                f"Waiting | toss_ready={toss_ready} | "
                f"mins_to_start={None if mins_to_start is None else round(mins_to_start, 2)}"
            )
            print(status_line)

            if require_squad and toss_ready and not squad_checked:
                sq = self.api.match_squad(candidate.match_id)
                squad_checked = True
                squad_ok = sq.get("status") == "success" and isinstance(sq.get("data"), list)
                if squad_ok:
                    squad_payload = sq.get("data")
                print(f"match_squad status={sq.get('status')} squad_ok={squad_ok}")

            if toss_ready and in_window:
                result = self.predict_from_match_payload(
                    m,
                    squad_ok=squad_ok,
                    mins_to_start=mins_to_start,
                    squad_payload=squad_payload,
                )
                return result

            time.sleep(max(10, poll_seconds))

        if last_match and last_match.get("tossWinner"):
            print("Timeout reached; generating best-effort prediction using latest toss data.")
            return self.predict_from_match_payload(
                last_match,
                squad_ok=squad_ok,
                mins_to_start=None,
                squad_payload=squad_payload,
            )

        raise RuntimeError("Timed out waiting for toss/window. Trigger again closer to toss time.")

    def predict_from_match_payload(
        self,
        m: Dict[str, Any],
        squad_ok: bool,
        mins_to_start: Optional[float],
        squad_payload: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        teams = m.get("teams") or ["", ""]
        if len(teams) < 2:
            raise RuntimeError("API payload missing teams.")

        team1_raw, team2_raw = teams[0], teams[1]
        venue_raw = m.get("venue", "Unknown")

        team1 = self.builder.map_team(normalize_team_name(team1_raw))
        team2 = self.builder.map_team(normalize_team_name(team2_raw))
        stadium = self.builder.map_stadium(venue_raw)

        toss_winner = self.builder.map_team(normalize_team_name(m.get("tossWinner", "")))
        toss_decision = normalize_toss_choice(m.get("tossChoice", "field"))

        ref_dt = parse_gmt_datetime(m.get("dateTimeGMT", ""))
        t1_form = self.builder.recent_form(team1, ref_dt)
        t2_form = self.builder.recent_form(team2, ref_dt)

        # Blend with latest operational DB form if available.
        t1_recent = fetch_recent_team_form(self.db_path, team1, n=5)
        t2_recent = fetch_recent_team_form(self.db_path, team2, n=5)
        if t1_recent is not None:
            t1_form = 0.7 * t1_form + 0.3 * t1_recent
        if t2_recent is not None:
            t2_form = 0.7 * t2_form + 0.3 * t2_recent

        vp = self.builder.venue_profile(stadium)
        venue_recent = fetch_recent_venue_stats(self.db_path, stadium)
        if venue_recent is not None:
            recent_score, recent_toss_adv = venue_recent
            vp["venue_score_prior"] = 0.8 * vp["venue_score_prior"] + 0.2 * recent_score
            vp["toss_advantage"] = 0.8 * vp["toss_advantage"] + 0.2 * recent_toss_adv

        t1p = self.builder.team_profile(team1)
        t2p = self.builder.team_profile(team2)

        t1_bat_mult, t1_bowl_mult, t1_players_used = self._player_multiplier_from_squad(
            team1, squad_payload if squad_ok else None
        )
        t2_bat_mult, t2_bowl_mult, t2_players_used = self._player_multiplier_from_squad(
            team2, squad_payload if squad_ok else None
        )

        # Player-aware adjustment from API squad roles + local squad list.
        t1p["bat_avg"] *= t1_bat_mult
        t1p["bat_sr"] *= t1_bat_mult
        t1p["bowl_eco"] /= t1_bowl_mult
        t1p["bowl_sr"] /= t1_bowl_mult

        t2p["bat_avg"] *= t2_bat_mult
        t2p["bat_sr"] *= t2_bat_mult
        t2p["bowl_eco"] /= t2_bowl_mult
        t2p["bowl_sr"] /= t2_bowl_mult

        is_high_dew = 1 if (toss_decision == "field" and vp["venue_chase_winrate_prior"] >= 0.52) else 0

        row = {
            "team1_form_winrate_5": t1_form,
            "team2_form_winrate_5": t2_form,
            "venue_score_prior": vp["venue_score_prior"],
            "venue_chase_winrate_prior": vp["venue_chase_winrate_prior"],
            "toss_advantage": vp["toss_advantage"],
            "is_high_dew": is_high_dew,
            "t1_bat_avg": t1p["bat_avg"],
            "t1_bat_sr": t1p["bat_sr"],
            "t1_bowl_eco": t1p["bowl_eco"],
            "t1_bowl_sr": t1p["bowl_sr"],
            "t2_bat_avg": t2p["bat_avg"],
            "t2_bat_sr": t2p["bat_sr"],
            "t2_bowl_eco": t2p["bowl_eco"],
            "t2_bowl_sr": t2p["bowl_sr"],
            "pitch_type": vp["pitch_type"],
            "bounce_and_carry": vp["bounce_and_carry"],
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "team1": team1,
            "team2": team2,
            "stadium": stadium,
        }

        X = pd.DataFrame([row])
        X = X[self.model.feature_names_]

        proba = self.model.predict_proba(X)[0]
        idx_class1 = list(self.model.classes_).index(1)
        team1_win_prob = float(proba[idx_class1])

        predicted_winner = team1 if team1_win_prob >= 0.5 else team2
        confidence = team1_win_prob if team1_win_prob >= 0.5 else (1.0 - team1_win_prob)

        result = {
            "timestamp_utc": now_utc().isoformat(),
            "match_id": m.get("id"),
            "match_name": m.get("name"),
            "series_name": "Indian Premier League",
            "match_start_utc": m.get("dateTimeGMT"),
            "team1": team1,
            "team2": team2,
            "venue": stadium,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "minutes_to_start": mins_to_start,
            "squad_checked_success": squad_ok,
            "players_api_used": bool(squad_ok and squad_payload),
            "team1_players_used": t1_players_used,
            "team2_players_used": t2_players_used,
            "team1_bat_multiplier": round(t1_bat_mult, 4),
            "team1_bowl_multiplier": round(t1_bowl_mult, 4),
            "team2_bat_multiplier": round(t2_bat_mult, 4),
            "team2_bowl_multiplier": round(t2_bowl_mult, 4),
            "team1_win_probability": round(team1_win_prob, 4),
            "predicted_winner": predicted_winner,
            "confidence": round(float(confidence), 4),
            "match_started": bool(m.get("matchStarted", False)),
            "match_ended": bool(m.get("matchEnded", False)),
            "prediction_source": "auto",
        }

        return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Trigger-based auto predictor: fetches live toss/match data from CricAPI "
            "and predicts winner in a pre-match window."
        )
    )
    p.add_argument("--api-key", default=os.getenv("CRICAPI_KEY", ""), help="CricAPI key")
    p.add_argument("--match-id", default=None, help="Specific match ID (optional)")
    p.add_argument("--series-search", default="Indian Premier League", help="Series search text")
    p.add_argument("--team1", default=None, help="Optional team filter")
    p.add_argument("--team2", default=None, help="Optional team filter")
    p.add_argument("--min-before", type=int, default=5, help="Min minutes before start")
    p.add_argument("--max-before", type=int, default=10, help="Max minutes before start")
    p.add_argument("--poll-seconds", type=int, default=60, help="Polling interval seconds")
    p.add_argument("--timeout-minutes", type=int, default=120, help="Stop waiting after timeout")
    p.add_argument(
        "--no-squad-check",
        action="store_true",
        help="Skip match_squad call (saves API hits)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.api_key:
        print("ERROR: API key missing. Pass --api-key or set CRICAPI_KEY.")
        return 2

    if args.min_before > args.max_before:
        print("ERROR: --min-before cannot be greater than --max-before.")
        return 2

    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent

    predictor = AutoPredictor(base_dir, args.api_key)

    selected = predictor.select_match(
        series_search=args.series_search,
        match_id=args.match_id,
        team1_filter=args.team1,
        team2_filter=args.team2,
    )

    print("Selected match:")
    print(f"  id={selected.match_id}")
    print(f"  name={selected.name}")
    print(f"  teams={selected.team1} vs {selected.team2}")
    print(f"  venue={selected.venue}")
    if selected.start_dt_utc:
        print(f"  start_utc={selected.start_dt_utc.isoformat()}")

    result = predictor.wait_and_predict(
        selected,
        min_before=args.min_before,
        max_before=args.max_before,
        poll_seconds=args.poll_seconds,
        timeout_minutes=args.timeout_minutes,
        require_squad=not args.no_squad_check,
    )

    out_dir = base_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"prediction_{result['match_id']}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    save_prediction(predictor.db_path, result)

    print("\nPrediction:")
    print(json.dumps(result, indent=2))
    print(f"Saved: {out_file}")
    print(f"Database updated: {predictor.db_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
