# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository content

- `CitiBike_Analysis.ipynb` — the analysis notebook, NYC Citi Bike trip data for January 2024 (~1.9M trips). No application code, package manifest, or test suite backs it — all logic lives in the notebook's cells.
- `docs/index.html` — the entire site: **one scrolling page**, not multiple pages. Sections in order: hero/overview stats, data & method, the hour-by-day ride-volume heatmap, the Leaflet ridership map, and a written report section. All three interactive pieces (overview stats, heatmap, map) are separate self-invoking JS blocks in one inline `<script>` at the bottom of the file — each scoped with its own `(function () { ... })()` and its own DOM ids (`overview-stat-row` / `heatmap-stat-row` / `map-stat-row`, etc.) specifically so they don't collide now that they share a page. If you add a fourth section, follow that pattern rather than merging state across sections.
  - **The report section's numbers are hand-written prose**, not computed from the data files at page-load time (unlike the stat strips, which are). If you regenerate the data with a different month or the pipeline changes the figures, you must manually update the report text in `docs/index.html` to match — nothing will flag a mismatch automatically.
  - Design tokens, nav, and card/stat-tile/footer styles live in `docs/assets/site.css`, shared by the page's single `<head>`. Nav links are in-page anchors (`#data`, `#when`, `#where`, `#report`), not separate URLs.
  - `assets/favicon.svg` — inline-SVG favicon.
  - Uses CDN-hosted Leaflet + leaflet.heat for the map section — if you bump the Leaflet version, recompute the `integrity` SRI hashes on its `<link>`/`<script>` tags (`curl -s <url> | openssl dgst -sha256 -binary | openssl base64`) or the browser will silently refuse to load them.
- `scripts/build_heatmap.py` — regenerates `docs/data/heatmap.{json,js}` and `docs/assets/heatmap.png` (used as the page's `og:image`) for a given month (`YYYYMM`), applying the same cleaning steps as the notebook.
- `scripts/build_geomap.py` — regenerates `docs/data/stations.{json,js}` (per-station ride counts/hourly profile), `docs/data/bike_paths.{geojson,js}` (pulled live from [NYC Open Data](https://data.cityofnewyork.us/dataset/New-York-City-Bike-Routes/mzxg-pwib), not versioned upstream — re-running can pick up newly added/retired bike lanes), and the page's `docs/data/summary.{json,js}`.

Regenerate via these scripts rather than hand-editing anything under `docs/data/` — they're the only place the cleaning/aggregation logic is defined. `write_summary()` in `build_geomap.py` reads from data already computed earlier in the same run (station list, bike-lane feature count) — no separate network call needed when re-running end to end.

The map's `fitBounds` padding (56px) must stay larger than the heat layer's `radius + blur` (14+16=30px) — smaller padding clips real edge-of-service stations (e.g. Riverdale, Bay Ridge) right at the canvas boundary, which previously made them disappear entirely.

Prior to this, the repo contained `MTA_Analysis.ipynb` (NYC subway ridership data with a high-ridership predictor model and feature importance analysis, see git history on commits `cdfd951`, `2b3238c`, `0868b1a`). That notebook has been removed from the working tree in favor of the Citi Bike analysis.

## Data source

The notebook downloads its dataset at runtime rather than reading a local file — there is no data committed to the repo:

```python
url = "https://s3.amazonaws.com/tripdata/202401-citibike-tripdata.zip"
```

This comes from the public [Citi Bike System Data](https://citibikenyc.com/system-data) feed (monthly zips at `https://s3.amazonaws.com/tripdata/index.html`). Any change to the analysis month means changing this URL and re-running from the top — the rest of the notebook assumes a single self-contained month.

## Notebook structure

The notebook follows a fixed pipeline, in order:

1. **Load** — download+unzip the month's CSVs directly from S3 into a single `df` via `requests`/`zipfile`/`io`.
2. **Clean** — in this order: parse `started_at`/`ended_at`, drop duplicate `ride_id`s, drop trips outside the claimed month (spillover at file edges), drop rows missing station/coordinate fields, drop non-positive durations, and drop trips over 24h (Citi Bike treats these as lost/stolen bike reports, not real rides). Each cleaning step depends on columns established by the previous one — reordering them will break things (e.g. `trip_minutes` must exist before duration-based filters run).
3. **EDA** — a sequence of independent plots/cuts over the cleaned `df`: trip duration distribution, member vs. casual rider mix, ride timing by hour (normalized per group), average trip duration by rider type, a ride-volume heatmap (day of week × hour), and top 15 busiest start stations.

Member vs. casual riders use a fixed color convention throughout (`rider_colors = {"member": "#2a78d6", "casual": "#eb6834"}`) — keep new charts consistent with this if extending the rider-mix analysis.

## Running the notebook

There's no requirements file or environment manifest in the repo. The notebook expects `pandas`, `requests`, `matplotlib`, and `seaborn` available in whatever Python environment runs it. Run top-to-bottom in Jupyter (or `jupyter nbconvert --to notebook --execute`) — cells depend on state from prior cells (e.g. the cleaned `df` is reused across every EDA cell).
