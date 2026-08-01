# Air Quality Dataset Findings Report

## Overview
This report details the findings from an exploratory analysis of the Air Quality Dataset from Mendeley Data (Version 2). The data spans from May 2, 2024, to December 11, 2025, covering 21 stations in Slovenia. The primary goals are to assess data quality, uncover meteorological trends, and evaluate the dataset's fitness for productization.

The full exploratory analysis, code, and visualizations can be found in the Jupyter Notebook: `analysis/AQI_Analysis.ipynb`.

## Data Quality & Profiling
- **Completeness**: The dataset successfully loaded 21 CSVs, with 0 duplicate timestamps across the index. 
- **Missingness**: Missing values aligned exactly with the dataset description. `PM10` is missing ~1.66% and `PM2.5` is missing ~1.65% of the time. `wind_direction` has the highest missingness rate at 49.17%. Other meteorological covariates like `temperature` and `rain` are missing 1.54% of the time.
- **Slovene Terminology**:
  - `clouds`: Values represent typical weather conditions (`jasno` = clear, `delno oblačno` = partly cloudy, `pretežno oblačno` = mostly cloudy, `oblačno` = cloudy).
  - `wind_direction`: These map to standard cardinal directions (`S` = North, `J` = South, `V` = East, `Z` = West, and combinations like `SV` for North-East).

## Key Exploratory Findings
- **Diurnal and Seasonal Trends**: There are distinct diurnal peaks corresponding to morning and evening traffic/heating, as well as significantly higher PM10 levels during winter months compared to summer (as seen in the generated plots in the `images/` directory).
- **Correlations**: PM10 and PM2.5 are strongly positively correlated. Both show negative correlations with `temperature` and `wind_speed`, validating that colder, still days trap more particulates near the surface.
- **Cross-Station Variability**: Certain stations consistently exhibit higher pollution levels. Online research confirmed that `E421` corresponds to the ARSO station in Nova Gorica (Grčna), a known urban/valley location prone to temperature inversions.
- **Precipitation vs. Rain**: The `precipitation` column has low correlation with the literal `rain` amount. It likely acts as an aggregated probability or humidity index rather than an actual rainfall measurement.

## Forecasting Experiment
We built a model to predict the next hour's PM10 concentration using meteorological covariates (temperature, pressure, wind_speed, etc.) and the current PM10 reading for station E421.
- **Naive Baseline (Predict next hour = current hour)**: 
  - MAE: 3.23
  - RMSE: 5.29
- **Random Forest (with meteorological covariates)**:
  - MAE: 3.67
  - RMSE: 6.11

**Takeaway**: The naive baseline outperformed the default Random Forest model. This underscores that air quality in the short-term (1-hour ahead) is highly autocorrelated (a random walk). While covariates like wind and temperature drive macro-trends, raw 1-hour look-ahead models require more sophisticated feature engineering (e.g., lagged rolling means) or deep learning (LSTM) to beat a naive persistence model.

## Product Research & Direction

### E.1 Data-Driven Feasibility
- **Coverage**: The dataset only covers 21 Slovenian stations. It cannot power a global or Europe-wide product.
- **Recency**: The data ends in December 2025. It lacks a live feed (API).
- **Granularity**: Hourly station-level data is excellent for regional dashboards but insufficient for hyperlocal (street-by-street) routing.
- **Licensing**: CC BY 4.0 permits commercial use with attribution.

### E.2 Competitive Landscape
- **Consumer App/Dashboard**: Products like IQAir or Plume Labs dominate this space. They rely on real-time data and global coverage. A retrospective Slovenian dataset cannot compete here unless paired with a live ARSO API integration.
- **Forecasting/Prediction API**: Services like BreezoMeter (now Google Air Quality) provide global APIs. Building a paid API solely for Slovenia with historical data is not viable.
- **Alerting/Health-Warning Service**: These services trigger alerts when levels exceed WHO guidelines (e.g., PM2.5 > 15 µg/m³ 24-hr average). Since this dataset isn't live, a real-time warning service is impossible without an external feed.
- **B2B Analytics Tool**: Cities and facility managers need tools to analyze historical trends, assess the impact of low-emission zones, and prepare compliance reports.

### E.3 Market Signal
The EU recently adopted Directive (EU) 2024/2881, setting stricter, legally binding air quality limit values for 2030 (closer to WHO guidelines). This creates a compliance pressure on European municipalities to monitor, model, and report their air quality improvements over time.

### E.4 Recommendation
**Top Pick**: **B2B Historical Analytics & Compliance Dashboard for Municipalities**
Given that the dataset is historical and regionally constrained, the best product direction is a B2B analytics tool aimed at Slovenian municipalities or researchers. 

**MVP Sketch**:
- **Features**: A dashboard comparing a city's historical pollution trends against the new EU 2030 limits and WHO targets. It would feature automated "exceedance reports" (e.g., counting days PM2.5 > 15 µg/m³) and seasonal correlation insights.
- **Gaps**: To make this an ongoing product, we would need to integrate a live scraper/API connected to the ARSO network.
- **Biggest Risk**: Municipalities might already receive standard reporting directly from ARSO, limiting their willingness to pay for a third-party dashboard.

## Sources
- **Dataset Provenance**: [Mendeley Data - Air Quality Dataset](https://data.mendeley.com/datasets/p86hnykz3d/2)
- **ARSO Stations (E421)**: Identified via [ARSO Air Quality Data Portal](https://www.arso.gov.si/zrak/kakovost%20zraka/podatki/)
- **WHO Guidelines**: [WHO Global Air Quality Guidelines (2021 Update)](https://www.who.int/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health)
- **EU Air Quality Directive 2024/2881**: [European Commission - Air Quality Standards](https://environment.ec.europa.eu/topics/air/air-quality/eu-air-quality-standards_en)
