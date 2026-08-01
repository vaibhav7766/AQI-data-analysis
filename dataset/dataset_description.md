# Air Quality Dataset

**Published:** 2 March 2026
**Version:** 2
**DOI:** [10.17632/p86hnykz3d.2](https://doi.org/10.17632/p86hnykz3d.2)
**Contributor:** Grega Vrbančič

## Description

This dataset contains time-series data of PM10 (and PM2.5) concentrations with meteorological covariates, recorded on an hourly basis from 21 different air quality measurement stations. The measurements were collected as part of an air pollution monitoring effort and are intended to support research on time-series forecasting, air quality analysis, and environmental data modeling.

The dataset contains hourly observations from 21 monitoring stations (one CSV file per station) over a shared measurement window:

- **Time span (all stations):** 2024-05-02 20:00:00 to 2025-12-11 07:00:00
- **Records:** 14,100 timestamps per station, 296,100 rows total
- **Timestamp grid:** Hourly; the provided time index is complete (max observed inter-sample gap = 1 hour)
- **Primary target for forecasting experiments:** PM10, PM2.5

Missingness is primarily value-level (NaNs in variables), not missing timestamps. PM10 missingness is low overall (~1.66%), while wind direction is frequently missing (~49%).

Each CSV corresponds to one monitoring station, identified by the file stem (e.g., station `E421` is in `E421.csv`).

## Data Format

| Property | Value |
|---|---|
| Format | CSV (comma-separated values) |
| Encoding | UTF-8 (note: `clouds` values contain Slovene characters) |
| Time column | `datetime`, format `YYYY-MM-DD HH:MM:SS` |
| Missing values | Empty fields are interpreted as missing; numeric missing values should be parsed as NaN |

## Variables (Columns)

Each station file contains the same columns:

| Column | Type | Description |
|---|---|---|
| `datetime` | string/datetime | Hourly timestamp (timezone not explicitly encoded) |
| `PM10` | float | PM10 concentration (unit as provided by source; typically µg/m³) |
| `PM2.5` | float | PM2.5 concentration (unit as provided by source; typically µg/m³) |
| `temperature` | float | Air temperature (unit as provided by source; typically °C) |
| `rain` | float | Precipitation amount/intensity (unit as provided by source) |
| `pressure` | float | Surface pressure (unit as provided by source; typically hPa) |
| `precipitation` | float | Percentage-valued meteorological covariate (0–100); semantics depend on upstream provider (often relative humidity or a probability-like indicator) |
| `wind_speed` | float | Wind speed (unit as provided by source) |
| `clouds` | string | Categorical sky condition (Slovene labels) — see below |
| `wind_direction` | string | Categorical wind direction (Slovene abbreviations, often missing) — see below |

### `clouds` categories

| Slovene | English |
|---|---|
| `jasno` | Clear |
| `delno oblačno` | Partly cloudy |
| `pretežno oblačno` | Mostly cloudy |
| `oblačno` | Cloudy |

### `wind_direction` categories

| Abbreviation | Direction |
|---|---|
| `S` | North |
| `SV` | Northeast |
| `V` | East |
| `JV` | Southeast |
| `J` | South |
| `JZ` | Southwest |
| `Z` | West |
| `SZ` | Northwest |

## Files

| File | Type | Size |
|---|---|---|
| [air_quality_dataset-v2.zip](https://data.mendeley.com/public-api/zip/p86hnykz3d/download/2) | zip | 2.84 MB |

**[⬇ Download All (2.84 MB)](https://data.mendeley.com/public-api/zip/p86hnykz3d/download/2)**

## Categories

`Environmental Analysis` · `Air Quality` · `Machine Learning` · `Time Series Analysis` · `Time Series Forecasting`

## Licence

**CC BY 4.0**