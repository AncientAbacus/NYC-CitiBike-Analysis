"""
Keep the numbers in the site's hand-written prose (and the README) in sync
with the data files the build scripts produce.

Prose stays hand-written, but every number in it is tagged with the fact it
states:

  docs/index.html   <span data-fact="bikes_to_move">5,927</span>
  README.md         <!--fact:bikes_to_move-->5,927<!--/fact-->

FACTS below computes each one from docs/data/*.json.

  python scripts/sync_facts.py           # check: exit 1 if any tagged number is stale
  python scripts/sync_facts.py --write   # rewrite every tagged number from the data
  python scripts/sync_facts.py --list    # print every fact and its current value

Runs in CI (see .github/workflows/checks.yml), so text can't silently drift
from the data. Standard library only.
"""

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "docs" / "data"
TARGETS = [REPO / "docs" / "index.html", REPO / "README.md"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def load():
    return {name: json.loads((DATA / f"{name}.json").read_text())
            for name in ("summary", "heatmap", "seasonality", "rebalancing", "forecast")}


def n(v):
    """12,345"""
    return f"{round(v):,}"


def pct(v, digits=0):
    """0.123 -> 12% (sign dropped: prose says 'cuts'/'lower')"""
    return f"{abs(v) * 100:.{digits}f}%"


def pct_range(lo, hi):
    """CI of a negative effect -> '22-32%' (ordered by magnitude)"""
    a, b = sorted((abs(lo) * 100, abs(hi) * 100))
    return f"{a:.0f}&ndash;{b:.0f}%"


def hour12(h):
    return f"{h % 12 or 12}{'am' if h < 12 else 'pm'}"


def _effect(d, name):
    return d["seasonality"]["model"]["effects"][name]


def _level(d, key):
    return next(x for x in d["forecast"]["departures"]["levels"] if x["level"] == key)


def _busiest_slot(d):
    return max(d["heatmap"]["data"], key=lambda r: r["rides"])


def _month(d, best):
    rows = d["rebalancing"]["monthly"]
    return (max if best else min)(rows, key=lambda r: r["bikes_to_move_am"])


FACTS = {
    # ---- data & cleaning ----
    "total_rides": lambda d: n(d["summary"]["total_rides"]),
    "raw_rows": lambda d: n(d["heatmap"]["cleaning"]["raw_rows"]),
    "removed_rows": lambda d: n(d["heatmap"]["cleaning"]["raw_rows"] - d["heatmap"]["cleaning"]["clean_rows"]),
    "removed_share": lambda d: pct(1 - d["heatmap"]["cleaning"]["clean_rows"] / d["heatmap"]["cleaning"]["raw_rows"], 2),
    "station_count": lambda d: n(d["summary"]["station_count"]),
    # ---- riders & timing ----
    "member_share": lambda d: pct(1 - d["heatmap"]["riders"]["casual_share"]),
    "casual_share": lambda d: pct(d["heatmap"]["riders"]["casual_share"]),
    "casual_share_weekend": lambda d: pct(d["heatmap"]["riders"]["casual_share_weekend"]),
    "electric_share": lambda d: pct(d["heatmap"]["riders"]["electric_share"]),
    "median_minutes_casual": lambda d: f'{d["heatmap"]["riders"]["median_minutes"]["casual"]:.0f}',
    "median_minutes_member": lambda d: f'{d["heatmap"]["riders"]["median_minutes"]["member"]:.0f}',
    "busiest_slot": lambda d: f'{_busiest_slot(d)["day"]} at {hour12(_busiest_slot(d)["hour"])}',
    "busiest_slot_rides": lambda d: n(_busiest_slot(d)["rides"]),
    # ---- seasons & weather ----
    "busiest_day": lambda d: d["seasonality"]["busiest_day"]["date"],
    "busiest_day_rides": lambda d: n(d["seasonality"]["busiest_day"]["rides"]),
    "quietest_day_rides": lambda d: n(d["seasonality"]["quietest_day"]["rides"]),
    "commute_rain_effect": lambda d: pct(_effect(d, "commute_rain")["effect"]),
    "commute_rain_ci": lambda d: pct_range(*_effect(d, "commute_rain")["ci95"]),
    "offpeak_rain_effect": lambda d: pct(_effect(d, "offpeak_rain")["effect"]),
    "offpeak_rain_ci": lambda d: pct_range(*_effect(d, "offpeak_rain")["ci95"]),
    "heavy_rain_effect": lambda d: pct(_effect(d, "heavy_rain")["effect"]),
    "heavy_rain_ci": lambda d: pct_range(*_effect(d, "heavy_rain")["ci95"]),
    "holiday_effect": lambda d: pct(_effect(d, "holiday")["effect"]),
    "holiday_ci": lambda d: pct_range(*_effect(d, "holiday")["ci95"]),
    "growth_ci": lambda d: pct_range(*_effect(d, "trend")["ci95"]),
    # ---- where ----
    "manhattan_station_share": lambda d: pct(d["summary"]["by_borough"]["Manhattan"]["station_share"]),
    "manhattan_ride_share": lambda d: pct(d["summary"]["by_borough"]["Manhattan"]["ride_share"]),
    "brooklyn_stations": lambda d: n(d["summary"]["by_borough"]["Brooklyn"]["stations"]),
    "manhattan_stations": lambda d: n(d["summary"]["by_borough"]["Manhattan"]["stations"]),
    "brooklyn_ride_share": lambda d: pct(d["summary"]["by_borough"]["Brooklyn"]["ride_share"]),
    "queens_station_share": lambda d: pct(d["summary"]["by_borough"]["Queens"]["station_share"]),
    "bronx_station_share": lambda d: pct(d["summary"]["by_borough"]["Bronx"]["station_share"]),
    "queens_ride_share": lambda d: pct(d["summary"]["by_borough"]["Queens"]["ride_share"]),
    "bronx_ride_share": lambda d: pct(d["summary"]["by_borough"]["Bronx"]["ride_share"]),
    # ---- rebalancing ----
    "bikes_to_move": lambda d: n(d["rebalancing"]["bikes_to_move_am"]),
    "drainer_count": lambda d: n(d["rebalancing"]["drainer_count"]),
    "filler_count": lambda d: n(d["rebalancing"]["filler_count"]),
    "am_pm_corr": lambda d: f'{d["rebalancing"]["am_pm_correlation"]:.2f}',
    "am_pm_corr_normalized": lambda d: f'{d["rebalancing"]["am_pm_correlation_normalized"]:.2f}',
    "brooklyn_am_loss": lambda d: n(abs(d["rebalancing"]["by_borough"]["Brooklyn"]["am_net"])),
    "manhattan_am_gain": lambda d: n(d["rebalancing"]["by_borough"]["Manhattan"]["am_net"]),
    "peak_move_month": lambda d: MONTHS[_month(d, True)["month"] - 1],
    "peak_move_bikes": lambda d: n(_month(d, True)["bikes_to_move_am"]),
    "low_move_month": lambda d: MONTHS[_month(d, False)["month"] - 1],
    "low_move_bikes": lambda d: n(_month(d, False)["bikes_to_move_am"]),
    "weekdays": lambda d: n(d["rebalancing"]["weekdays"]),
    # ---- forecast ----
    "forecast_stations": lambda d: n(d["forecast"]["stations"]),
    "forecast_trees": lambda d: n(d["forecast"]["departures"]["trees"]),
    "forecast_mae": lambda d: f'{d["forecast"]["departures"]["hourly"]["model"]["mae"]:.2f}',
    "forecast_7d_mae": lambda d: f'{d["forecast"]["departures"]["hourly"]["seven_day_mean"]["mae"]:.2f}',
    "forecast_vs_7d": lambda d: pct(d["forecast"]["improvement"]["vs_seven_day_mean_hourly"]),
    "forecast_vs_naive": lambda d: pct(d["forecast"]["improvement"]["vs_seasonal_naive_hourly"]),
    "arrivals_vs_7d": lambda d: pct(d["forecast"]["improvement"]["arrivals_vs_seven_day_mean_hourly"]),
    "acc_station_hour": lambda d: pct(_level(d, "station_hour")["model"]),
    "acc_station_day": lambda d: pct(_level(d, "station_day")["model"]),
    "acc_borough_hour": lambda d: pct(_level(d, "borough_hour")["model"]),
    "acc_system_hour": lambda d: pct(_level(d, "system_hour")["model"]),
    "acc_system_day": lambda d: pct(_level(d, "system_day")["model"]),
    "acc_system_day_prev": lambda d: pct(d["forecast"]["yesterdays_weather"]["levels"]["system_day"]),
    "acc_borough_hour_prev": lambda d: pct(d["forecast"]["yesterdays_weather"]["levels"]["borough_hour"]),
    "acc_station_day_prev": lambda d: pct(d["forecast"]["yesterdays_weather"]["levels"]["station_day"]),
    "acc_station_hour_prev": lambda d: pct(d["forecast"]["yesterdays_weather"]["levels"]["station_hour"]),
    "forecast_vs_7d_prev": lambda d: pct(d["forecast"]["yesterdays_weather"]["vs_seven_day_mean_hourly"]),
    "acc_system_day_7d": lambda d: pct(_level(d, "system_day")["seven_day_mean"]),
    "ceiling_station_hour": lambda d: pct(_level(d, "station_hour")["chance_ceiling"]),
    "backtest_folds": lambda d: str(len(d["forecast"]["backtest"])),
    "backtest_wins": lambda d: str(sum(f["model"] < f["seven_day_mean"] for f in d["forecast"]["backtest"])),
    "interval_coverage_busy": lambda d: pct(d["forecast"]["intervals"]["busy"]["coverage"]),
    "net_mae_model": lambda d: f'{d["forecast"]["net_flow_am"]["mae"]["model"]:.1f}',
    "net_mae_7d": lambda d: f'{d["forecast"]["net_flow_am"]["mae"]["seven_day_mean"]:.1f}',
    "net_top_k": lambda d: str(d["forecast"]["net_flow_am"]["top_k"]),
    "net_precision_model": lambda d: f'{d["forecast"]["net_flow_am"]["precision_losers"]["model"] * d["forecast"]["net_flow_am"]["top_k"]:.1f}',
    "net_precision_7d": lambda d: f'{d["forecast"]["net_flow_am"]["precision_losers"]["seven_day_mean"] * d["forecast"]["net_flow_am"]["top_k"]:.1f}',
}

PATTERNS = [
    re.compile(r'(<span data-fact="(?P<name>[a-z0-9_]+)">)(?P<value>.*?)(</span>)'),
    re.compile(r'(<!--fact:(?P<name>[a-z0-9_]+)-->)(?P<value>.*?)(<!--/fact-->)'),
]


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    data = load()
    values = {k: f(data) for k, f in FACTS.items()}
    if mode == "--list":
        for k, v in values.items():
            print(f"{k:28} {v}")
        return 0

    problems, used = [], set()
    for path in TARGETS:
        text = path.read_text()

        def fix(m):
            name = m.group("name")
            used.add(name)
            if name not in values:
                problems.append(f"{path.name}: unknown fact '{name}'")
                return m.group(0)
            if m.group("value") != values[name]:
                problems.append(f"{path.name}: {name} says {m.group('value')!r}, data says {values[name]!r}")
            return m.group(1) + values[name] + m.group(4)

        new = text
        for pat in PATTERNS:
            new = pat.sub(fix, new)
        if mode == "--write" and new != text:
            path.write_text(new)

    if mode == "--write":
        print(f"Synced {len(used)} facts across {len(TARGETS)} files")
        return 0
    if problems:
        print("Stale numbers in prose (run: python scripts/sync_facts.py --write):")
        print("\n".join("  " + p for p in problems))
        return 1
    print(f"OK: {len(used)} tagged facts match the data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
