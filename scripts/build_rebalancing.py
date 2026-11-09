"""
Build the station-rebalancing data behind the site (docs/data/rebalancing.json + .js).

Bike share has a structural problem: the morning commute moves bikes one way
(residential neighborhoods -> job centers) and the evening commute moves them
back. Stations that lose bikes faster than they gain them run empty; stations
that gain them run out of docks. Operators fix this by moving bikes on
trucks, trailers, and with rider incentives.

For each station, on non-holiday weekdays, this computes the average
departures and arrivals in the morning rush (07:00-09:59) and the evening
rush (16:00-18:59). Net flow = arrivals - departures:
  net < 0  "drainer": loses bikes, needs bikes delivered before the rush
  net > 0  "filler":  gains bikes, needs bikes removed (or docks freed)

"Bikes to move each weekday morning" is the sum of all drainers' average
morning deficits: the minimum number of bikes that have to be put back in
place, every weekday, just to undo the morning commute. It's a lower bound:
it ignores station capacity (not in the trip data), within-window timing,
and day-to-day variation.

Usage: python scripts/build_rebalancing.py [YYYY | YYYYMM]   # defaults to 2024
"""

import json
import sys

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from pipeline import DEFAULT_PERIOD, REPO, load_clean_period, period_label

AM = (7, 10)   # [start, end) hours
PM = (16, 19)
TOP_N = 10


def weekday_mask(ts: pd.Series) -> pd.Series:
    holidays = USFederalHolidayCalendar().holidays(ts.min().normalize(), ts.max().normalize())
    return (ts.dt.dayofweek < 5) & ~ts.dt.normalize().isin(holidays)


def window_counts(ts: pd.Series, station: pd.Series, window: tuple, month: bool = False) -> pd.Series:
    hour = ts.dt.hour
    keep = weekday_mask(ts) & (hour >= window[0]) & (hour < window[1])
    keys = [station[keep]]
    if month:
        keys.append(ts[keep].dt.month.rename("month"))
    out = pd.Series(1, index=keep[keep].index).groupby(keys, observed=True).sum()
    # start/end station columns carry different category sets; use plain strings
    # so departures and arrivals align on station name
    names = out.index.get_level_values(0).astype(str)
    if month:
        out.index = pd.MultiIndex.from_arrays([names, out.index.get_level_values(1)], names=["station", "month"])
    else:
        out.index = pd.Index(names, name="station")
    return out


def main() -> None:
    period = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PERIOD
    df = load_clean_period(period, columns=[
        "started_at", "ended_at", "start_station_name", "end_station_name",
        "start_lat", "start_lng",
    ])

    days = pd.Series(pd.to_datetime(df["started_at"].dt.normalize().unique()))
    n_weekdays = int(weekday_mask(days).sum())
    print(f"{n_weekdays} non-holiday weekdays", flush=True)

    flows = pd.DataFrame({
        "am_dep": window_counts(df["started_at"], df["start_station_name"], AM),
        "am_arr": window_counts(df["ended_at"], df["end_station_name"], AM),
        "pm_dep": window_counts(df["started_at"], df["start_station_name"], PM),
        "pm_arr": window_counts(df["ended_at"], df["end_station_name"], PM),
    }).fillna(0) / n_weekdays
    flows["am_net"] = flows["am_arr"] - flows["am_dep"]
    flows["pm_net"] = flows["pm_arr"] - flows["pm_dep"]

    # monthly: bikes to move each weekday morning, month by month
    month_days = days[weekday_mask(days)].dt.month.value_counts()
    m_dep = window_counts(df["started_at"], df["start_station_name"], AM, month=True)
    m_arr = window_counts(df["ended_at"], df["end_station_name"], AM, month=True)
    m_net = m_arr.sub(m_dep, fill_value=0).unstack(fill_value=0)  # station x month
    m_net = m_net / month_days.reindex(m_net.columns).values
    monthly = [
        {"month": int(m), "bikes_to_move_am": round(float((-m_net[m]).clip(lower=0).sum()))}
        for m in sorted(m_net.columns)
    ]

    # coordinates + borough from the map's station file (same pipeline, same names)
    coords = df.groupby("start_station_name", observed=True)[["start_lat", "start_lng"]].median()
    coords.index = coords.index.astype(str)
    stations_file = json.loads((REPO / "docs" / "data" / "stations.json").read_text())
    boro = {s["name"]: s["boro"] for s in stations_file["stations"]}

    flows = flows.join(coords, how="inner")
    flows["boro"] = [boro.get(n, "Other") for n in flows.index]
    # ignore near-dormant stations: under 1 rush-hour trip per weekday morning
    active = flows[(flows["am_dep"] + flows["am_arr"]) >= 1].copy()

    bikes_to_move = float((-active["am_net"]).clip(lower=0).sum())
    corr = float(active["am_net"].corr(active["pm_net"]))

    def station_record(name, r):
        return {
            "name": str(name), "boro": r.boro,
            "lat": round(float(r.start_lat), 6), "lng": round(float(r.start_lng), 6),
            "am_dep": round(float(r.am_dep), 1), "am_arr": round(float(r.am_arr), 1),
            "am_net": round(float(r.am_net), 1), "pm_net": round(float(r.pm_net), 1),
        }

    by_boro = (
        active.groupby("boro")
        .agg(stations=("am_net", "size"), am_net=("am_net", "sum"),
             drainers=("am_net", lambda s: int((s <= -2).sum())), fillers=("am_net", lambda s: int((s >= 2).sum())))
        .round(1)
    )

    out = {
        "label": period_label(period),
        "weekdays": n_weekdays,
        "windows": {"am": list(AM), "pm": list(PM)},
        "bikes_to_move_am": round(bikes_to_move),
        "am_pm_correlation": round(corr, 3),
        "active_stations": int(len(active)),
        "drainer_count": int((active["am_net"] <= -2).sum()),
        "filler_count": int((active["am_net"] >= 2).sum()),
        "top_drainers": [station_record(n, r) for n, r in active.nsmallest(TOP_N, "am_net").iterrows()],
        "top_fillers": [station_record(n, r) for n, r in active.nlargest(TOP_N, "am_net").iterrows()],
        "by_borough": {b: {k: (float(v) if isinstance(v, float) else int(v)) for k, v in row.items()} for b, row in by_boro.iterrows()},
        "monthly": monthly,
        "stations": [
            {"n": str(n), "lat": round(float(r.start_lat), 5), "lng": round(float(r.start_lng), 5),
             "am": round(float(r.am_net), 1), "pm": round(float(r.pm_net), 1), "b": r.boro}
            for n, r in active.iterrows()
        ],
    }
    data_dir = REPO / "docs" / "data"
    payload = json.dumps(out, separators=(",", ":"))
    (data_dir / "rebalancing.json").write_text(payload)
    (data_dir / "rebalancing.js").write_text("window.REBALANCING_DATA = " + payload + ";\n")
    print("Wrote docs/data/rebalancing.{json,js}", flush=True)
    print(json.dumps({k: v for k, v in out.items() if k != "stations"}, indent=1)[:4000])


if __name__ == "__main__":
    main()
