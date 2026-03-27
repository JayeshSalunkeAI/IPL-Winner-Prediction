import json
import os
import urllib.request
import urllib.error
from typing import Any, Dict

API_KEY = os.getenv("CRICAPI_KEY", "")
BASE_URL = "https://api.cricapi.com/v1"


def fetch(url: str) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def safe_fetch(label: str, url: str) -> Dict[str, Any]:
    print(f"\n[{label}] {url}")
    try:
        data = fetch(url)
        print(f"status: {data.get('status')}")
        if "reason" in data:
            print(f"reason: {data['reason']}")
        info = data.get("info")
        if isinstance(info, dict):
            print(
                "usage:",
                {
                    "hitsToday": info.get("hitsToday"),
                    "hitsUsed": info.get("hitsUsed"),
                    "hitsLimit": info.get("hitsLimit"),
                },
            )
        return data
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        print(f"http_error: {e.code} {body}")
    except Exception as e:
        print(f"error: {repr(e)}")
    return {}


def main() -> None:
    if not API_KEY:
        print("ERROR: Set CRICAPI_KEY before running this script.")
        return

    print("Testing CricketData/CricAPI key for auto-prediction readiness...")

    # 1) Core list endpoint
    current = safe_fetch(
        "currentMatches",
        f"{BASE_URL}/currentMatches?apikey={API_KEY}&offset=0",
    )
    if isinstance(current.get("data"), list) and current["data"]:
        sample = current["data"][0]
        print("current sample fields:", sorted(sample.keys())[:12])

    # 2) IPL series discovery
    series = safe_fetch(
        "series search IPL",
        f"{BASE_URL}/series?apikey={API_KEY}&offset=0&search=IPL",
    )
    series_data = series.get("data") if isinstance(series.get("data"), list) else []
    if series_data:
        chosen = series_data[0]
        print("selected series:", chosen.get("name"))

        # 3) Series info for match IDs
        sid = chosen.get("id")
        if sid:
            info = safe_fetch(
                "series_info",
                f"{BASE_URL}/series_info?apikey={API_KEY}&id={sid}",
            )
            data = info.get("data") if isinstance(info.get("data"), dict) else {}
            match_list = data.get("matchList") if isinstance(data.get("matchList"), list) else []
            print("matchList size:", len(match_list))

            # 4) Match info for toss + venue
            if match_list:
                mid = match_list[0].get("id")
                if mid:
                    match_info = safe_fetch(
                        "match_info",
                        f"{BASE_URL}/match_info?apikey={API_KEY}&id={mid}",
                    )
                    md = match_info.get("data") if isinstance(match_info.get("data"), dict) else {}
                    print(
                        "match_info essentials:",
                        {
                            "teams": md.get("teams"),
                            "venue": md.get("venue"),
                            "tossWinner": md.get("tossWinner"),
                            "tossChoice": md.get("tossChoice"),
                            "matchStarted": md.get("matchStarted"),
                        },
                    )

                    # 5) Squad endpoint check (needed for XI-based feature updates)
                    squad = safe_fetch(
                        "match_squad",
                        f"{BASE_URL}/match_squad?apikey={API_KEY}&id={mid}",
                    )
                    sd = squad.get("data") if isinstance(squad.get("data"), list) else []
                    if sd:
                        first_team = sd[0]
                        players = first_team.get("players") if isinstance(first_team, dict) else []
                        print(
                            "match_squad sample:",
                            {
                                "teamName": first_team.get("teamName") if isinstance(first_team, dict) else None,
                                "players_count": len(players) if isinstance(players, list) else None,
                            },
                        )

    print("\nDone.")


if __name__ == "__main__":
    main()
