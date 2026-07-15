# NYC Citi Bike Analysis

Exploratory analysis of NYC's Citi Bike system using January 2024 trip data
(~1.9M rides) — cleaning, EDA, and a look at how members (subscribers) and
casual riders use the system differently: when they ride, how long for, and
where.

## Contents

- [`CitiBike_Analysis.ipynb`](CitiBike_Analysis.ipynb) — the notebook: loads
  the month's trip data directly from Citi Bike's public S3 feed, cleans it,
  and walks through trip duration, rider mix, ride timing, and busiest
  stations.
- [`docs/`](docs/index.html) — a static site rendering one of the notebook's
  findings, an interactive ride-volume heatmap by hour and day of week.
  [**View it live**](https://ancientabacus.github.io/NYC-CitiBike-Analysis/)
  (once GitHub Pages is enabled for this repo — see below).

## The heatmap

![Ride volume by hour and day of week](docs/assets/heatmap.png)

Weekday ridership shows a sharp AM/PM commute pattern (8am and 5-6pm); weekend
ridership spreads across a broader, later midday window instead — members
commute, casual riders wander in on their own schedule.

## Data source

[Citi Bike System Data](https://citibikenyc.com/system-data), published
monthly as zipped CSVs at `https://s3.amazonaws.com/tripdata/index.html`. The
notebook downloads the file at runtime — no data is committed to this repo.

## Running the notebook

Requires `pandas`, `numpy`, `requests`, `matplotlib`, and `seaborn`. Run
top-to-bottom in Jupyter — later cells depend on the cleaned DataFrame built
earlier in the notebook.

## Enabling the website

The heatmap site lives in `docs/` so it can be served straight from GitHub
Pages: repo **Settings → Pages → Deploy from a branch → `main` / `docs`**.

## Regenerating the heatmap

`scripts/build_heatmap.py` downloads a month of Citi Bike data, cleans it the
same way as the notebook, and rewrites `docs/data/heatmap.{json,js}` and
`docs/assets/heatmap.png`:

```bash
python scripts/build_heatmap.py 202401   # YYYYMM, defaults to 202401
```
