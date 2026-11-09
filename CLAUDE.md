# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository content

- `docs/index.html` — the entire site: **one scrolling page** covering all of 2024 (~44.2M cleaned trips). Sections in order: hero/overview stats, data & method (`#data`), when (`#when`: hour-by-day heatmap, then the seasons & weather sub-section `#seasons`), where (`#where`: Leaflet ridership map + 5 per-borough small-multiple maps), rebalancing (`#rebalance`), demand forecast (`#forecast`), and a written report (`#report`). The interactive pieces are separate blocks in one inline `<script>` at the bottom of the file, each wrapped in `runSection(name, function () { ... })` (catches and logs errors so one failing section, e.g. a map when the Leaflet CDN is blocked, can't stop the others from rendering) and given its own DOM ids (`overview-stat-row` / `heatmap-stat-row` / `season-stat-row` / `map-stat-row` / `rebalance-stat-row` / `forecast-stat-row`, etc.). If you add another section, follow that pattern rather than merging state across sections.
  - **The report section's numbers (and the hero's key findings) are hand-written prose**, not computed from the data files at page-load time (unlike the stat strips, tables and charts, which are). If you regenerate the data or the pipeline changes the figures, you must manually update that text in `docs/index.html` to match; nothing will flag a mismatch automatically.
  - `docs/assets/charts.js` — small dependency-free SVG chart helpers (`Charts.lineChart`, `scatterChart`, `barChart`) used by the seasonality, rebalancing and forecast sections: one y-axis, nice ticks, crosshair/per-bar tooltips (filled with `textContent`, never `innerHTML`, since station names are data), redraw on resize and on light/dark change. Series colors are `{light, dark}` pairs validated per mode with the dataviz skill's `validate_palette.js`: blue `#2a78d6`/`#3987e5`, orange `#eb6834`/`#d2611f` (the light orange fails the dark-mode lightness band, hence the darker step), red `#e34948`/`#e66767`. Rebalancing uses a diverging red (loses bikes) ↔ gray ↔ blue (gains bikes) scale.
  - **Per-borough small multiples**: the citywide map normalizes every station against the busiest station system-wide (always Manhattan), which crushes outer-borough stations to near-invisibility. The 5 borough panels each `fitBounds` to their own stations and normalize against their own max. Staten Island has zero Citi Bike stations (verified, not a bug); its panel shows bike lanes only with a note. `summary.json`'s `by_borough` is pre-seeded with all 5 boroughs so a zero-station borough still has a `{stations: 0, ...}` entry, which the page's "no stations here" logic depends on.
  - Design tokens, nav, and card/stat-tile/footer styles live in `docs/assets/site.css`. Nav links are in-page anchors.
  - Basemap tiles are Esri's keyless gray canvas (`World_Light_Gray_Base` / `World_Dark_Gray_Base`, via the shared `basemapTiles()` near the top of the inline script). Don't switch back to CARTO's `basemaps.cartocdn.com`: it serves "API KEY REQUIRED" watermarked tiles without a key.
  - Uses CDN-hosted Leaflet + leaflet.heat. If you bump the Leaflet version, recompute the `integrity` SRI hashes on its `<link>`/`<script>` tags (`curl -s <url> | openssl dgst -sha256 -binary | openssl base64`) or the browser will silently refuse to load them.
- `CitiBike_Analysis.ipynb` — an exploratory deep-dive into **January 2024 only** (EDA plus a member-vs-casual classifier). It is not the source of the site's numbers.

## Scripts (`scripts/`)

Every script takes a period: `YYYY` (full year, the default `2024`) or `YYYYMM` (one month).

- `pipeline.py` — the single source of truth: loading (`load_month`, zips cached in gitignored `.cache/`), cleaning (`clean`, logs rows removed per step), a per-month cleaned parquet cache (`clean_month_path`, `.cache/clean/YYYYMM.parquet`) and `load_clean_period(period, columns=...)`, NOAA weather (`load_weather_hourly` from LCD FM-15 reports, `load_weather_daily` from GHCN-Daily, Central Park station, cached in `.cache/`), and the notebook's classifier features (`engineer_features`, `FEATURES`). Never re-implement cleaning elsewhere. Each month is cleaned on its own and trips are assigned to the month they start in.
- `build_heatmap.py` — `docs/data/heatmap.{json,js}` and `docs/assets/heatmap.png` (the page's `og:image`).
- `build_geomap.py` — `stations.{json,js}` (per-station totals/hourly profile + `boro`), `bike_paths.{geojson,js}` (pulled live from [NYC Open Data](https://data.cityofnewyork.us/dataset/New-York-City-Bike-Routes/mzxg-pwib); the feed is not versioned and occasionally contains empty geometries, which are dropped), and `summary.{json,js}`. Requires `shapely`.
- `build_seasonality.py` — `seasonality.{json,js}`: daily rides joined to NOAA daily weather, binned dry-weekday rides by high temperature, and a log-linear model (day of week, holiday, temp + temp², rain buckets, snow, linear growth trend) whose rain/snow/holiday coefficients are the "what weather costs" numbers. The temperature terms are controls only: the quadratic can't capture the plateau-then-dip above ~30°C, so the site charts the binned averages.
- `build_rebalancing.py` — `rebalancing.{json,js}`: per-station average net flow (arrivals − departures) in the weekday AM (7–10) and PM (16–19) rush on non-holiday weekdays, top losers/gainers, by-borough totals, monthly "bikes to move" (sum of losing stations' average morning deficit, a lower bound that ignores dock capacity). Reads `stations.json` for boroughs, so run `build_geomap.py` first. Stations with < 1 rush trip per morning are excluded.
- `build_forecast.py` — `forecast.{json,js}`: station × hour departure forecast with `HistGradientBoostingRegressor` (Poisson loss). Features are all available a day ahead (hour, weekday, holiday, observed weather, lag 24h, lag 168h, same-hour 7-day mean, training-period station profile). Split by date: train Jan 8–Aug, validate Sep–Oct (early stopping in steps of 25 trees, done manually since sklearn's built-in early stopping uses a random holdout), refit on Jan–Oct, test Nov–Dec. Baselines: same hour last week, same hour yesterday, station profile. **Don't add a day-of-year feature**: with a date-based split the test dates are unseen values and validation error rose with more trees when it was included. Uses ~6 GB RAM and ~20 min on 4 cores.

Regenerate via these scripts rather than hand-editing anything under `docs/data/`. Full run order: `build_heatmap.py`, `build_geomap.py`, `build_seasonality.py`, `build_rebalancing.py`, `build_forecast.py`. A full-year rebuild downloads ~7.5 GB of zips into `.cache/` the first time.

`assign_boroughs()` in `build_geomap.py` classifies stations by real borough polygon (point-in-polygon against NYC Open Data's [Borough Boundaries](https://data.cityofnewyork.us/City-Government/Borough-Boundaries/gthc-hcne)), not a lat/lng box. Stations outside all 5 boroughs (Jersey City/Hoboken stops that received a few cross-river trips) get `boro: "Other"`.

The ridership map's `fitBounds` padding (56px) must stay larger than the heat layer's `radius + blur` (14+16=30px), or edge-of-service stations (e.g. Riverdale, Bay Ridge) get clipped at the canvas boundary.

Prior to this, the repo contained `MTA_Analysis.ipynb` (NYC subway ridership; see git history on commits `cdfd951`, `2b3238c`, `0868b1a`), removed in favor of the Citi Bike analysis.

## Data sources

Nothing is committed; everything downloads at build time into gitignored `.cache/`:

- Trips: [Citi Bike System Data](https://citibikenyc.com/system-data), monthly zips at `https://s3.amazonaws.com/tripdata/YYYYMM-citibike-tripdata.zip`.
- Weather: NOAA NCEI access API (`www.ncei.noaa.gov`), Central Park station (LCD `72505394728`, GHCND `USW00094728`). Open-Meteo was tried first but rate-limits shared IPs.
- Bike lanes and borough boundaries: NYC Open Data (`data.cityofnewyork.us`).

## Notebook structure (`CitiBike_Analysis.ipynb`, January 2024)

1. **Load / Clean** — `pipeline.load_month()` and `pipeline.clean()` (duplicate `ride_id`s, out-of-month spillover, missing station/coordinates, non-positive durations, trips over 24h, in that order; `trip_minutes` must exist before the duration filters).
2. **EDA** — trip duration distribution, member vs. casual mix, ride timing by hour (normalized per group), average duration by rider type, day × hour heatmap, top 15 start stations.
3. **Predicting Rider Type** — `pipeline.engineer_features()`, then `LogisticRegression` and `RandomForestClassifier` (both `class_weight="balanced"`) on `member_casual`. Leans on ROC-AUC and casual-class recall because "always guess member" already scores 89% accuracy.

Member vs. casual riders use a fixed color convention (`{"member": "#2a78d6", "casual": "#eb6834"}`).

Run with `pip install -r requirements.txt`, then top-to-bottom from the repo root (it imports `scripts/pipeline.py` via `sys.path`): `jupyter nbconvert --to notebook --execute --inplace CitiBike_Analysis.ipynb`.
