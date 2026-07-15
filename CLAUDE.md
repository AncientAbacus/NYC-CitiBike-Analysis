# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository content

- `CitiBike_Analysis.ipynb` — the analysis notebook, NYC Citi Bike trip data for January 2024 (~1.9M trips). No application code, package manifest, or test suite backs it — all logic lives in the notebook's cells.
- `docs/` — a static site (`index.html` + `data/heatmap.{json,js}` + `assets/heatmap.png`) rendering the notebook's hour-by-day ride-volume heatmap as an interactive page, meant to be served via GitHub Pages from this folder.
- `scripts/build_heatmap.py` — regenerates the `docs/` heatmap data and PNG for a given month (`YYYYMM`), applying the same cleaning steps as the notebook. Run this after changing which month the site displays, rather than hand-editing `docs/data/heatmap.json`.

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

There's no requirements file or environment manifest in the repo. The notebook expects `pandas`, `numpy`, `requests`, `matplotlib`, and `seaborn` available in whatever Python environment runs it. Run top-to-bottom in Jupyter (or `jupyter nbconvert --to notebook --execute`) — cells depend on state from prior cells (e.g. the cleaned `df` is reused across every EDA cell).
