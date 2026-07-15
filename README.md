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
- [`docs/index.html`](docs/index.html) — a single scrolling page built on the
  notebook's findings, telling the whole story in order: the data & method,
  an interactive ride-volume **heatmap** by hour and day of week, a
  geospatial **ridership map** of station density against NYC's bike lane
  network (scrubbable by hour of day), and a written **report** on what the
  results actually show.
  [**View it live**](https://ancientabacus.github.io/NYC-CitiBike-Analysis/)
  (once GitHub Pages is enabled for this repo — see below).

## The heatmap

![Ride volume by hour and day of week](docs/assets/heatmap.png)

Weekday ridership shows a sharp AM/PM commute pattern (8am and 5-6pm); weekend
ridership spreads across a broader, later midday window instead — members
commute, casual riders wander in on their own schedule.

## The ridership map

The map section plots the same January 2024 trips geospatially: a Leaflet
heat layer of trip starts per station (toggleable, and scrubbable through the
24 hours to watch the commute rush move across the city), overlaid on NYC
DOT's actual bike lane network colored by borough, so you can see how
ridership relates to where the infrastructure is — or isn't.

## The report

The page closes with a written summary of what the analysis found: rider mix
and trip duration (member vs. casual), the weekday commute pattern versus
weekend spread, and the geographic concentration of ridership in Manhattan
despite bike lane infrastructure being built out fairly evenly citywide.

## Data source

[Citi Bike System Data](https://citibikenyc.com/system-data), published
monthly as zipped CSVs at `https://s3.amazonaws.com/tripdata/index.html`. The
notebook downloads the file at runtime — no data is committed to this repo.

## Running the notebook

Requires `pandas`, `requests`, `matplotlib`, and `seaborn`. Run top-to-bottom
in Jupyter — later cells depend on the cleaned DataFrame built earlier in the
notebook.

## Enabling the website

The site is a single file, `docs/index.html`, so it can be served straight
from GitHub Pages: repo **Settings → Pages → Deploy from a branch → `main` /
`docs`**. Design tokens, nav, and card/stat-tile styles live in
`docs/assets/site.css`.

## Regenerating the site data

`scripts/build_heatmap.py` downloads a month of Citi Bike data, cleans it the
same way as the notebook, and rewrites `docs/data/heatmap.{json,js}` and
`docs/assets/heatmap.png`:

```bash
python scripts/build_heatmap.py 202401   # YYYYMM, defaults to 202401
```

`scripts/build_geomap.py` does the same for the ridership map: per-station
location/ride counts (`docs/data/stations.{json,js}`), a trimmed copy of NYC
DOT's current bike route network (`docs/data/bike_paths.{geojson,js}`) pulled
live from [NYC Open Data](https://data.cityofnewyork.us/dataset/New-York-City-Bike-Routes/mzxg-pwib),
and the page's summary stats (`docs/data/summary.{json,js}`):

```bash
python scripts/build_geomap.py 202401
```

If you regenerate either, double-check the figures cited in the report
section of `docs/index.html` still match — they're written prose, not pulled
live from the data files, so they don't update themselves.
