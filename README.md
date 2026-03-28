# Open-Meteo Scenario Proxy

A FastAPI abstraction layer over Open-Meteo that does two things:

1. returns raw daily or hourly temperature series for a location and date range
2. derives 10-year percentile-based scenario curves for the requested date range

## What changed in this version

Compared to the earlier thin wrapper, this version adds a second API surface for scenario modeling:

- looks back over the prior `historical_years` years, default `10`
- fetches daily `temperature_2m_mean` from Open-Meteo for the same seasonal period in each of those years
- applies a configurable smoothing window around each target day, default `±3 days`
- derives scenario curves from percentiles, default `10,50,75,90`
- always also returns `baseline_mean`

So for a requested range like `2026-07-01` to `2026-07-14`, the service can return:

- `baseline_mean`
- `cool_p10`
- `median_p50`
- `warm_p75`
- `hot_p90`

Each scenario is returned as its own date/temperature series aligned to the requested future date range.

## Endpoints

- `GET /` — small web UI
- `GET /health` — health check
- `GET /api/config` — fetch saved config
- `PUT /api/config` — save config
- `GET /api/series` — fetch a raw temperature series from query params or saved config
- `POST /api/series` — fetch a raw temperature series from JSON body or saved config
- `GET /api/scenarios` — fetch percentile-based scenario curves from query params or saved config
- `POST /api/scenarios` — fetch percentile-based scenario curves from JSON body or saved config

## Config fields

```json
{
  "location_name": "Valencia",
  "start_date": "2026-07-01",
  "end_date": "2026-07-14",
  "granularity": "daily",
  "timezone": "auto",
  "historical_years": 10,
  "climatology_window_days": 3,
  "scenario_percentiles": [10, 50, 75, 90]
}
```

Notes:
- raw `/api/series` supports `daily` or `hourly`
- `/api/scenarios` derives daily curves only
- you can set either `location_name` or `latitude` + `longitude`

## Run locally on PowerShell

```powershell
cd C:\GIT_Repos\weather
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
```

Open:

- UI: `http://127.0.0.1:8000/`
- Swagger: `http://127.0.0.1:8000/docs`

## Example calls

### Save config

```powershell
curl.exe -X PUT http://127.0.0.1:8000/api/config `
  -H "Content-Type: application/json" `
  -d "{\"location_name\":\"Valencia\",\"start_date\":\"2026-07-01\",\"end_date\":\"2026-07-14\",\"granularity\":\"daily\",\"timezone\":\"auto\",\"historical_years\":10,\"climatology_window_days\":3,\"scenario_percentiles\":[10,50,75,90]}"
```

### Fetch raw series

```powershell
curl.exe http://127.0.0.1:8000/api/series
```

### Fetch scenario curves using saved config

```powershell
curl.exe http://127.0.0.1:8000/api/scenarios
```

### Fetch scenario curves with explicit payload

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/scenarios `
  -H "Content-Type: application/json" `
  -d "{\"location_name\":\"Valencia\",\"start_date\":\"2026-07-01\",\"end_date\":\"2026-07-14\",\"granularity\":\"daily\",\"timezone\":\"auto\",\"historical_years\":10,\"climatology_window_days\":3,\"scenario_percentiles\":[10,50,75,90]}"
```

## Response shape for `/api/scenarios`

```json
{
  "source": "open-meteo",
  "granularity": "daily",
  "methodology": "Daily scenario curves derived from the prior 10 years...",
  "start_date": "2026-07-01",
  "end_date": "2026-07-14",
  "historical_years": 10,
  "historical_year_range": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
  "climatology_window_days": 3,
  "scenario_percentiles": [10, 50, 75, 90],
  "scenarios": [
    { "name": "baseline_mean", "percentile": null, "readings": [ ... ] },
    { "name": "cool_p10", "percentile": 10, "readings": [ ... ] },
    { "name": "median_p50", "percentile": 50, "readings": [ ... ] },
    { "name": "warm_p75", "percentile": 75, "readings": [ ... ] },
    { "name": "hot_p90", "percentile": 90, "readings": [ ... ] }
  ]
}
```

## Implementation notes

- The service uses Open-Meteo geocoding to resolve `location_name` when coordinates are not provided.
- Open-Meteo historical access is used for both the raw series endpoint and the scenario endpoint.
- Scenario derivation is intentionally transparent: it exposes the upstream request URLs used for each historical year.
- The service assumes Open-Meteo historical archive support starting from 1940.

## Honest limitation

This version computes percentile-based daily scenario curves from historical weather. It does **not** predict future weather in a meteorological sense. It gives you plausible temperature scenarios based on the prior 10 years of analogous seasonal conditions.
