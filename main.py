from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import fmean
from typing import Literal, Optional
from urllib.parse import urlencode

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
CONFIG_PATH = APP_DIR / "config.json"
CACHE_DIR = APP_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
MIN_ARCHIVE_YEAR = 1940


class BaseLocationConfig(BaseModel):
    start_date: date = Field(..., description="Requested start date in YYYY-MM-DD format")
    end_date: date = Field(..., description="Requested end date in YYYY-MM-DD format")
    location_name: Optional[str] = Field(
        default=None,
        description="City or place name to resolve via Open-Meteo geocoding API",
    )
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)
    timezone: str = Field(default="auto", description="Timezone forwarded to Open-Meteo")

    @model_validator(mode="after")
    def validate_dates_and_location(self) -> "BaseLocationConfig":
        if self.start_date > self.end_date:
            raise ValueError("start_date must be less than or equal to end_date")

        has_named_location = bool(self.location_name and self.location_name.strip())
        has_coordinates = self.latitude is not None and self.longitude is not None

        if not has_named_location and not has_coordinates:
            raise ValueError(
                "Provide either location_name or both latitude and longitude"
            )
        return self


class AppConfig(BaseLocationConfig):
    granularity: Literal["daily", "hourly"] = "daily"
    historical_years: int = Field(
        default=10,
        ge=3,
        le=30,
        description="How many prior years to use for scenario climatology",
    )
    climatology_window_days: int = Field(
        default=3,
        ge=0,
        le=14,
        description=(
            "Smoothing window around each target day. 3 means samples from the same day "
            "plus +/- 3 days in each historical year."
        ),
    )
    scenario_percentiles: list[int] = Field(
        default_factory=lambda: [10, 50, 75, 90],
        description="Percentiles used to derive scenario curves",
    )

    @model_validator(mode="after")
    def validate_scenario_settings(self) -> "AppConfig":
        cleaned = sorted({int(p) for p in self.scenario_percentiles})
        if not cleaned:
            raise ValueError("scenario_percentiles cannot be empty")
        if any(p < 0 or p > 100 for p in cleaned):
            raise ValueError("scenario_percentiles must be between 0 and 100")
        self.scenario_percentiles = cleaned

        earliest_year = self.start_date.year - self.historical_years
        if earliest_year < MIN_ARCHIVE_YEAR:
            raise ValueError(
                f"Requested historical window reaches {earliest_year}. Open-Meteo archive "
                f"support starts in {MIN_ARCHIVE_YEAR}. Reduce historical_years or move start_date forward."
            )
        return self


class StoredConfig(BaseModel):
    config: AppConfig


class CurrentLocationConfig(BaseModel):
    location_name: Optional[str] = Field(
        default=None,
        description="City or place name to resolve via Open-Meteo geocoding API",
    )
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)
    timezone: str = Field(default="auto", description="Timezone forwarded to Open-Meteo")

    @model_validator(mode="after")
    def validate_location(self) -> "CurrentLocationConfig":
        has_named_location = bool(self.location_name and self.location_name.strip())
        has_coordinates = self.latitude is not None and self.longitude is not None

        if not has_named_location and not has_coordinates:
            raise ValueError(
                "Provide either location_name or both latitude and longitude"
            )
        return self


class LocationResolution(BaseModel):
    name: str
    latitude: float
    longitude: float
    country: Optional[str] = None
    admin1: Optional[str] = None
    timezone: Optional[str] = None


class TemperatureReading(BaseModel):
    timestamp: str
    temperature_c: float


class TemperatureSeriesResponse(BaseModel):
    source: str = "open-meteo"
    granularity: Literal["daily", "hourly"]
    start_date: date
    end_date: date
    resolved_location: LocationResolution
    upstream_url: str
    readings: list[TemperatureReading]


class CurrentTemperatureResponse(BaseModel):
    source: str = "open-meteo"
    resolved_location: LocationResolution
    upstream_url: str
    timestamp: str
    temperature_c: float
    humidity_pct: Optional[int] = None
    elevation_m: Optional[float] = None
    barometric_pressure_inhg: Optional[float] = None
    interval_seconds: Optional[int] = None


class ScenarioSeriesPoint(BaseModel):
    date: str
    temperature_c: float


class ScenarioSeries(BaseModel):
    name: str
    percentile: Optional[int] = None
    readings: list[ScenarioSeriesPoint]


class ScenarioResponse(BaseModel):
    source: str = "open-meteo"
    granularity: Literal["daily"] = "daily"
    methodology: str
    start_date: date
    end_date: date
    resolved_location: LocationResolution
    historical_years: int
    historical_year_range: list[int]
    climatology_window_days: int
    scenario_percentiles: list[int]
    upstream_urls: list[str]
    scenarios: list[ScenarioSeries]


app = FastAPI(
    title="Open-Meteo Abstraction Layer",
    version="0.4.0",
    description=(
        "A FastAPI wrapper around Open-Meteo that exposes current temperature, raw "
        "temperature series, short-range forecasts, and 10-year percentile-based "
        "scenario curves for downstream systems."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_config() -> Optional[AppConfig]:
    if not CONFIG_PATH.exists():
        return None
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return StoredConfig.model_validate(data).config
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to read config: {exc}")


def save_config(config: AppConfig) -> None:
    payload = StoredConfig(config=config).model_dump(mode="json")
    CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_cache_key(prefix: str, params: dict[str, str | float]) -> str:
    serialized = json.dumps({"prefix": prefix, "params": params}, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def read_cache(prefix: str, params: dict[str, str | float]) -> Optional[dict]:
    cache_path = CACHE_DIR / f"{prefix}_{build_cache_key(prefix, params)}.json"
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_cache(prefix: str, params: dict[str, str | float], payload: dict) -> None:
    cache_path = CACHE_DIR / f"{prefix}_{build_cache_key(prefix, params)}.json"
    cache_path.write_text(json.dumps(payload), encoding="utf-8")


async def get_json_with_backoff(
    *,
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, str | float],
    cache_prefix: str,
    timeout_error_label: str,
) -> dict:
    cached = read_cache(cache_prefix, params)
    if cached is not None:
        return cached

    last_error: Optional[str] = None
    backoff_seconds = [1, 2, 4, 8]

    for attempt, delay in enumerate(backoff_seconds, start=1):
        try:
            response = await client.get(url, params=params)
        except httpx.TimeoutException as exc:
            last_error = str(exc)
            if attempt == len(backoff_seconds):
                raise HTTPException(
                    status_code=504,
                    detail=f"{timeout_error_label} timed out after retries: {last_error}",
                ) from exc
            continue

        if response.status_code == 200:
            payload = response.json()
            write_cache(cache_prefix, params, payload)
            return payload

        if response.status_code == 429:
            last_error = response.text
            if attempt == len(backoff_seconds):
                break
            import asyncio
            await asyncio.sleep(delay)
            continue

        raise HTTPException(
            status_code=502,
            detail=f"{timeout_error_label} failed with status {response.status_code}: {response.text}",
        )

    raise HTTPException(
        status_code=502,
        detail=f"{timeout_error_label} failed after retries due to rate limiting: {last_error}",
    )


async def resolve_location_from_values(
    *,
    location_name: Optional[str],
    latitude: Optional[float],
    longitude: Optional[float],
    timezone: str,
) -> LocationResolution:
    if latitude is not None and longitude is not None:
        return LocationResolution(
            name=location_name or "Custom coordinates",
            latitude=latitude,
            longitude=longitude,
            timezone=timezone if timezone != "auto" else None,
        )

    assert location_name is not None
    params = {
        "name": location_name,
        "count": 1,
        "language": "en",
        "format": "json",
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        payload = await get_json_with_backoff(
            client=client,
            url=OPEN_METEO_GEOCODING_URL,
            params=params,
            cache_prefix="geocode",
            timeout_error_label="Geocoding request",
        )

    results = payload.get("results") or []
    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"No location found for '{location_name}'",
        )

    item = results[0]
    return LocationResolution(
        name=item.get("name") or location_name,
        latitude=item["latitude"],
        longitude=item["longitude"],
        country=item.get("country"),
        admin1=item.get("admin1"),
        timezone=item.get("timezone"),
    )


async def resolve_location(config: BaseLocationConfig | CurrentLocationConfig) -> LocationResolution:
    return await resolve_location_from_values(
        location_name=config.location_name,
        latitude=config.latitude,
        longitude=config.longitude,
        timezone=config.timezone,
    )


async def fetch_archive_bucket(
    *,
    client: httpx.AsyncClient,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    granularity: Literal["daily", "hourly"],
    timezone: str,
) -> tuple[str, list[str], list[float]]:
    params: dict[str, str | float] = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timezone": timezone,
    }

    bucket_name: Literal["daily", "hourly"]
    temp_key: str
    if granularity == "daily":
        params["daily"] = "temperature_2m_mean"
        bucket_name = "daily"
        temp_key = "temperature_2m_mean"
    else:
        params["hourly"] = "temperature_2m"
        bucket_name = "hourly"
        temp_key = "temperature_2m"

    upstream_url = f"{OPEN_METEO_ARCHIVE_URL}?{urlencode(params)}"
    payload = await get_json_with_backoff(
        client=client,
        url=OPEN_METEO_ARCHIVE_URL,
        params=params,
        cache_prefix=f"archive_{granularity}",
        timeout_error_label="Open-Meteo archive request",
    )

    bucket = payload.get(bucket_name)
    if not bucket:
        raise HTTPException(
            status_code=502,
            detail=f"Unexpected Open-Meteo response: missing '{bucket_name}' bucket",
        )

    times = bucket.get("time") or []
    values = bucket.get(temp_key) or []
    if len(times) != len(values):
        raise HTTPException(
            status_code=502,
            detail="Unexpected Open-Meteo response: time/value length mismatch",
        )

    cleaned_times: list[str] = []
    cleaned_values: list[float] = []
    for timestamp, value in zip(times, values):
        if value is None:
            continue
        cleaned_times.append(timestamp)
        cleaned_values.append(float(value))

    return upstream_url, cleaned_times, cleaned_values


async def fetch_temperature_series(config: AppConfig) -> TemperatureSeriesResponse:
    resolved = await resolve_location(config)
    async with httpx.AsyncClient(timeout=30.0) as client:
        upstream_url, times, values = await fetch_archive_bucket(
            client=client,
            latitude=resolved.latitude,
            longitude=resolved.longitude,
            start_date=config.start_date,
            end_date=config.end_date,
            granularity=config.granularity,
            timezone=config.timezone,
        )

    readings = [
        TemperatureReading(timestamp=timestamp, temperature_c=value)
        for timestamp, value in zip(times, values)
    ]

    return TemperatureSeriesResponse(
        granularity=config.granularity,
        start_date=config.start_date,
        end_date=config.end_date,
        resolved_location=resolved,
        upstream_url=upstream_url,
        readings=readings,
    )


# ─── Humidity models ──────────────────────────────────────────────────────────

class HumidityReading(BaseModel):
    timestamp: str
    humidity_pct: int


class HumiditySeriesResponse(BaseModel):
    source: str = "open-meteo"
    granularity: Literal["daily"] = "daily"
    start_date: date
    end_date: date
    resolved_location: LocationResolution
    upstream_url: str
    readings: list[HumidityReading]


# ─── Humidity archive fetch ───────────────────────────────────────────────────
# Mirrors fetch_archive_bucket but requests relative_humidity_2m_mean.
# Intentionally separate — the temperature projection path is not touched.

async def fetch_humidity_archive_bucket(
    *,
    client: httpx.AsyncClient,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    timezone: str,
) -> tuple[str, list[str], list[int]]:
    params: dict[str, str | float] = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "relative_humidity_2m_mean",
        "timezone": timezone,
    }

    upstream_url = f"{OPEN_METEO_ARCHIVE_URL}?{urlencode(params)}"
    payload = await get_json_with_backoff(
        client=client,
        url=OPEN_METEO_ARCHIVE_URL,
        params=params,
        cache_prefix="archive_humidity_daily",
        timeout_error_label="Open-Meteo humidity archive request",
    )

    bucket = payload.get("daily")
    if not bucket:
        raise HTTPException(
            status_code=502,
            detail="Unexpected Open-Meteo response: missing 'daily' bucket in humidity archive",
        )

    times: list[str] = bucket.get("time") or []
    values: list[Optional[float]] = bucket.get("relative_humidity_2m_mean") or []

    if len(times) != len(values):
        raise HTTPException(
            status_code=502,
            detail="Unexpected Open-Meteo response: time/value length mismatch in humidity archive",
        )

    cleaned_times: list[str] = []
    cleaned_values: list[int] = []
    for timestamp, value in zip(times, values):
        if value is None:
            continue
        cleaned_times.append(timestamp)
        cleaned_values.append(int(round(float(value))))

    return upstream_url, cleaned_times, cleaned_values


async def fetch_humidity_series(config: AppConfig) -> HumiditySeriesResponse:
    resolved = await resolve_location(config)
    async with httpx.AsyncClient(timeout=30.0) as client:
        upstream_url, times, values = await fetch_humidity_archive_bucket(
            client=client,
            latitude=resolved.latitude,
            longitude=resolved.longitude,
            start_date=config.start_date,
            end_date=config.end_date,
            timezone=config.timezone,
        )

    readings = [
        HumidityReading(timestamp=timestamp, humidity_pct=value)
        for timestamp, value in zip(times, values)
    ]

    return HumiditySeriesResponse(
        start_date=config.start_date,
        end_date=config.end_date,
        resolved_location=resolved,
        upstream_url=upstream_url,
        readings=readings,
    )


async def fetch_current_temperature(config: CurrentLocationConfig) -> CurrentTemperatureResponse:
    resolved = await resolve_location(config)

    params: dict[str, str | float] = {
        "latitude": resolved.latitude,
        "longitude": resolved.longitude,
        "current": "temperature_2m,relative_humidity_2m,surface_pressure",
        "timezone": config.timezone,
    }

    upstream_url = f"{OPEN_METEO_FORECAST_URL}?{urlencode(params)}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        payload = await get_json_with_backoff(
            client=client,
            url=OPEN_METEO_FORECAST_URL,
            params=params,
            cache_prefix="forecast_current",
            timeout_error_label="Open-Meteo current weather request",
        )

    current = payload.get("current")
    if not current:
        raise HTTPException(
            status_code=502,
            detail="Unexpected Open-Meteo response: missing 'current' bucket",
        )

    timestamp = current.get("time")
    temperature = current.get("temperature_2m")
    if timestamp is None or temperature is None:
        raise HTTPException(
            status_code=502,
            detail="Unexpected Open-Meteo response: missing current time or temperature_2m",
        )

    humidity_raw = current.get("relative_humidity_2m")
    humidity_pct: Optional[int] = int(humidity_raw) if humidity_raw is not None else None

    elevation_raw = payload.get("elevation")
    elevation_m: Optional[float] = float(elevation_raw) if elevation_raw is not None else None

    pressure_raw = current.get("surface_pressure")
    barometric_pressure_inhg: Optional[float] = round(float(pressure_raw) * 0.02953, 4) if pressure_raw is not None else None

    interval_seconds = payload.get("current_units", {}).get("interval")
    try:
        interval_seconds = int(interval_seconds) if interval_seconds is not None else None
    except (TypeError, ValueError):
        interval_seconds = None

    return CurrentTemperatureResponse(
        resolved_location=resolved,
        upstream_url=upstream_url,
        timestamp=str(timestamp),
        temperature_c=float(temperature),
        humidity_pct=humidity_pct,
        elevation_m=elevation_m,
        barometric_pressure_inhg=barometric_pressure_inhg,
        interval_seconds=interval_seconds,
    )


def iter_dates(start_date: date, end_date: date) -> list[date]:
    days = (end_date - start_date).days
    return [start_date + timedelta(days=offset) for offset in range(days + 1)]


def safe_anchor_date(target_day: date, historical_year: int) -> date:
    try:
        return date(historical_year, target_day.month, target_day.day)
    except ValueError:
        # Handles February 29 for non-leap years.
        return date(historical_year, 2, 28)


def label_for_percentile(percentile: int) -> str:
    mapping = {
        10: "cool_p10",
        25: "mild_p25",
        50: "median_p50",
        75: "warm_p75",
        90: "hot_p90",
    }
    return mapping.get(percentile, f"p{percentile}")


async def fetch_scenario_curves(config: AppConfig) -> ScenarioResponse:
    resolved = await resolve_location(config)
    target_days = iter_dates(config.start_date, config.end_date)
    historical_years = list(
        range(config.start_date.year - config.historical_years, config.start_date.year)
    )

    expanded_ranges: list[tuple[int, date, date]] = []
    for historical_year in historical_years:
        translated_start = safe_anchor_date(config.start_date, historical_year)
        translated_end = safe_anchor_date(config.end_date, historical_year)
        expanded_ranges.append(
            (
                historical_year,
                translated_start - timedelta(days=config.climatology_window_days),
                translated_end + timedelta(days=config.climatology_window_days),
            )
        )

    combined_start = min(range_start for _, range_start, _ in expanded_ranges)
    combined_end = max(range_end for _, _, range_end in expanded_ranges)

    async with httpx.AsyncClient(timeout=60.0) as client:
        upstream_url, times, values = await fetch_archive_bucket(
            client=client,
            latitude=resolved.latitude,
            longitude=resolved.longitude,
            start_date=combined_start,
            end_date=combined_end,
            granularity="daily",
            timezone=config.timezone,
        )

    lookup = {timestamp: value for timestamp, value in zip(times, values)}

    baseline_points: list[ScenarioSeriesPoint] = []
    percentile_points: dict[int, list[ScenarioSeriesPoint]] = {
        p: [] for p in config.scenario_percentiles
    }

    for target_day in target_days:
        samples: list[float] = []

        for historical_year in historical_years:
            anchor = safe_anchor_date(target_day, historical_year)
            for offset in range(
                -config.climatology_window_days, config.climatology_window_days + 1
            ):
                sample_day = anchor + timedelta(days=offset)
                value = lookup.get(sample_day.isoformat())
                if value is not None:
                    samples.append(value)

        if not samples:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"No historical samples available to derive scenario for {target_day.isoformat()}"
                ),
            )

        baseline_points.append(
            ScenarioSeriesPoint(
                date=target_day.isoformat(),
                temperature_c=round(fmean(samples), 2),
            )
        )

        np_samples = np.asarray(samples, dtype=float)
        for percentile in config.scenario_percentiles:
            percentile_value = float(np.percentile(np_samples, percentile))
            percentile_points[percentile].append(
                ScenarioSeriesPoint(
                    date=target_day.isoformat(),
                    temperature_c=round(percentile_value, 2),
                )
            )

    scenarios: list[ScenarioSeries] = [
        ScenarioSeries(name="baseline_mean", percentile=None, readings=baseline_points)
    ]
    for percentile in config.scenario_percentiles:
        scenarios.append(
            ScenarioSeries(
                name=label_for_percentile(percentile),
                percentile=percentile,
                readings=percentile_points[percentile],
            )
        )

    return ScenarioResponse(
        methodology=(
            "Daily scenario curves derived from the prior 10 years of Open-Meteo daily "
            "temperature_2m_mean values, fetched in a single cached archive request and "
            "converted into percentile-based climatology with a configurable +/- day smoothing window."
        ),
        start_date=config.start_date,
        end_date=config.end_date,
        resolved_location=resolved,
        historical_years=config.historical_years,
        historical_year_range=historical_years,
        climatology_window_days=config.climatology_window_days,
        scenario_percentiles=config.scenario_percentiles,
        upstream_urls=[upstream_url],
        scenarios=scenarios,
    )


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config", response_model=AppConfig)
async def get_config() -> AppConfig:
    config = load_config()
    if config is None:
        raise HTTPException(status_code=404, detail="No config saved yet")
    return config


@app.put("/api/config", response_model=AppConfig)
async def put_config(config: AppConfig) -> AppConfig:
    save_config(config)
    return config


@app.post("/api/series", response_model=TemperatureSeriesResponse)
async def post_series(config: Optional[AppConfig] = None) -> TemperatureSeriesResponse:
    effective_config = config or load_config()
    if effective_config is None:
        raise HTTPException(
            status_code=400,
            detail="No payload provided and no saved config found",
        )
    return await fetch_temperature_series(effective_config)


@app.get("/api/series", response_model=TemperatureSeriesResponse)
async def get_series(
    start_date: Optional[date] = Query(default=None),
    end_date: Optional[date] = Query(default=None),
    location_name: Optional[str] = Query(default=None),
    latitude: Optional[float] = Query(default=None),
    longitude: Optional[float] = Query(default=None),
    granularity: Literal["daily", "hourly"] = Query(default="daily"),
    timezone: str = Query(default="auto"),
    historical_years: int = Query(default=10),
    climatology_window_days: int = Query(default=3),
    scenario_percentiles: str = Query(default="10,50,75,90"),
) -> TemperatureSeriesResponse:
    saved = load_config()

    if (
        start_date is None
        and end_date is None
        and location_name is None
        and latitude is None
        and longitude is None
        and saved is not None
    ):
        return await fetch_temperature_series(saved)

    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=400,
            detail="start_date and end_date are required unless a saved config exists",
        )

    try:
        config = AppConfig(
            start_date=start_date,
            end_date=end_date,
            location_name=location_name,
            latitude=latitude,
            longitude=longitude,
            granularity=granularity,
            timezone=timezone,
            historical_years=historical_years,
            climatology_window_days=climatology_window_days,
            scenario_percentiles=[int(x) for x in scenario_percentiles.split(",") if x],
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return await fetch_temperature_series(config)


@app.post("/api/humidity/series", response_model=HumiditySeriesResponse)
async def post_humidity_series(config: Optional[AppConfig] = None) -> HumiditySeriesResponse:
    """
    Return daily mean relative humidity (%) from the Open-Meteo archive.
    Uses the same date range and location as /api/series.
    The temperature projection path is not affected.
    """
    effective_config = config or load_config()
    if effective_config is None:
        raise HTTPException(
            status_code=400,
            detail="No payload provided and no saved config found",
        )
    return await fetch_humidity_series(effective_config)


@app.get("/api/humidity/series", response_model=HumiditySeriesResponse)
async def get_humidity_series(
    start_date: Optional[date] = Query(default=None),
    end_date: Optional[date] = Query(default=None),
    location_name: Optional[str] = Query(default=None),
    latitude: Optional[float] = Query(default=None),
    longitude: Optional[float] = Query(default=None),
    timezone: str = Query(default="auto"),
) -> HumiditySeriesResponse:
    """
    Return daily mean relative humidity (%) from the Open-Meteo archive.
    Accepts the same query parameters as /api/series.
    """
    saved = load_config()

    if (
        start_date is None
        and end_date is None
        and location_name is None
        and latitude is None
        and longitude is None
        and saved is not None
    ):
        return await fetch_humidity_series(saved)

    if saved is not None:
        start_date = start_date or saved.start_date
        end_date   = end_date   or saved.end_date
        location_name = location_name or saved.location_name
        latitude      = latitude      or saved.latitude
        longitude     = longitude     or saved.longitude

    try:
        config = AppConfig(
            start_date=start_date,
            end_date=end_date,
            location_name=location_name,
            latitude=latitude,
            longitude=longitude,
            timezone=timezone,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return await fetch_humidity_series(config)


@app.post("/api/scenarios", response_model=ScenarioResponse)
async def post_scenarios(config: Optional[AppConfig] = None) -> ScenarioResponse:
    effective_config = config or load_config()
    if effective_config is None:
        raise HTTPException(
            status_code=400,
            detail="No payload provided and no saved config found",
        )
    return await fetch_scenario_curves(effective_config)


@app.get("/api/scenarios", response_model=ScenarioResponse)
async def get_scenarios(
    start_date: Optional[date] = Query(default=None),
    end_date: Optional[date] = Query(default=None),
    location_name: Optional[str] = Query(default=None),
    latitude: Optional[float] = Query(default=None),
    longitude: Optional[float] = Query(default=None),
    timezone: str = Query(default="auto"),
    historical_years: int = Query(default=10),
    climatology_window_days: int = Query(default=3),
    scenario_percentiles: str = Query(default="10,50,75,90"),
) -> ScenarioResponse:
    saved = load_config()

    if (
        start_date is None
        and end_date is None
        and location_name is None
        and latitude is None
        and longitude is None
        and saved is not None
    ):
        return await fetch_scenario_curves(saved)

    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=400,
            detail="start_date and end_date are required unless a saved config exists",
        )

    try:
        config = AppConfig(
            start_date=start_date,
            end_date=end_date,
            location_name=location_name,
            latitude=latitude,
            longitude=longitude,
            timezone=timezone,
            granularity="daily",
            historical_years=historical_years,
            climatology_window_days=climatology_window_days,
            scenario_percentiles=[int(x) for x in scenario_percentiles.split(",") if x],
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return await fetch_scenario_curves(config)


@app.post("/api/current", response_model=CurrentTemperatureResponse)
async def post_current(
    config: Optional[CurrentLocationConfig] = None,
) -> CurrentTemperatureResponse:
    if config is not None:
        effective_config = config
    else:
        saved = load_config()
        if saved is None:
            raise HTTPException(
                status_code=400,
                detail="No payload provided and no saved config found",
            )
        effective_config = CurrentLocationConfig(
            location_name=saved.location_name,
            latitude=saved.latitude,
            longitude=saved.longitude,
            timezone=saved.timezone,
        )
    return await fetch_current_temperature(effective_config)


@app.get("/api/current", response_model=CurrentTemperatureResponse)
async def get_current(
    location_name: Optional[str] = Query(default=None),
    latitude: Optional[float] = Query(default=None),
    longitude: Optional[float] = Query(default=None),
    timezone: str = Query(default="auto"),
) -> CurrentTemperatureResponse:
    saved = load_config()

    if (
        location_name is None
        and latitude is None
        and longitude is None
        and saved is not None
    ):
        config = CurrentLocationConfig(
            location_name=saved.location_name,
            latitude=saved.latitude,
            longitude=saved.longitude,
            timezone=saved.timezone,
        )
        return await fetch_current_temperature(config)

    try:
        config = CurrentLocationConfig(
            location_name=location_name,
            latitude=latitude,
            longitude=longitude,
            timezone=timezone,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return await fetch_current_temperature(config)


# ─── Forecast endpoint ────────────────────────────────────────────────────────
#
# Returns a short-range daily temperature forecast (up to 16 days) from
# Open-Meteo's forecast API, aligned to the same shape as /api/scenarios
# readings so the compressor projector can blend them seamlessly.
#
# Blending contract (for consumers):
#   days 0–6:   high-confidence meteorological forecast
#   days 7–15:  extended forecast, degrading accuracy
#   days 16+:   not available here — use /api/scenarios climatology
#
# The response includes forecast_reliable_days so consumers know exactly
# where to switch from forecast to climatology in a blended profile.

OPEN_METEO_FORECAST_DAYS_MAX = 16


class ForecastPoint(BaseModel):
    date: str
    temperature_c: float


class ForecastResponse(BaseModel):
    source: str = "open-meteo"
    granularity: Literal["daily"] = "daily"
    resolved_location: LocationResolution
    upstream_url: str
    forecast_days: int
    forecast_reliable_days: int = 7
    readings: list[ForecastPoint]


async def fetch_forecast_series(
    *,
    location_name: Optional[str],
    latitude: Optional[float],
    longitude: Optional[float],
    timezone: str,
    days: int,
) -> ForecastResponse:
    days = max(1, min(days, OPEN_METEO_FORECAST_DAYS_MAX))

    resolved = await resolve_location_from_values(
        location_name=location_name,
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
    )

    params: dict[str, str | float | int] = {
        "latitude": resolved.latitude,
        "longitude": resolved.longitude,
        "daily": "temperature_2m_mean",
        "forecast_days": days,
        "timezone": timezone,
    }

    upstream_url = f"{OPEN_METEO_FORECAST_URL}?{urlencode(params)}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        payload = await get_json_with_backoff(
            client=client,
            url=OPEN_METEO_FORECAST_URL,
            params=params,
            cache_prefix="forecast_daily",
            timeout_error_label="Open-Meteo forecast request",
        )

    daily = payload.get("daily")
    if not daily:
        raise HTTPException(
            status_code=502,
            detail="Unexpected Open-Meteo response: missing 'daily' bucket",
        )

    times: list[str] = daily.get("time") or []
    values: list[Optional[float]] = daily.get("temperature_2m_mean") or []

    if len(times) != len(values):
        raise HTTPException(
            status_code=502,
            detail="Unexpected Open-Meteo response: time/value length mismatch in forecast",
        )

    readings: list[ForecastPoint] = []
    for timestamp, value in zip(times, values):
        if value is None:
            continue
        readings.append(ForecastPoint(date=str(timestamp), temperature_c=round(float(value), 2)))

    return ForecastResponse(
        resolved_location=resolved,
        upstream_url=upstream_url,
        forecast_days=len(readings),
        forecast_reliable_days=min(7, len(readings)),
        readings=readings,
    )


@app.post("/api/forecast", response_model=ForecastResponse)
async def post_forecast(
    days: int = Query(default=7, ge=1, le=16, description="Number of forecast days (1-16)"),
    config: Optional[CurrentLocationConfig] = None,
) -> ForecastResponse:
    """
    Return a short-range daily temperature forecast (up to 16 days).

    Uses saved config location if no body is provided.
    `forecast_reliable_days` in the response marks where meteorological confidence
    degrades (day 7 boundary). Consumers should blend with /api/scenarios
    climatology beyond that point for longer projection windows.
    """
    if config is not None:
        loc_name = config.location_name
        lat = config.latitude
        lon = config.longitude
        tz = config.timezone
    else:
        saved = load_config()
        if saved is None:
            raise HTTPException(
                status_code=400,
                detail="No payload provided and no saved config found",
            )
        loc_name = saved.location_name
        lat = saved.latitude
        lon = saved.longitude
        tz = saved.timezone

    return await fetch_forecast_series(
        location_name=loc_name,
        latitude=lat,
        longitude=lon,
        timezone=tz,
        days=days,
    )


@app.get("/api/forecast", response_model=ForecastResponse)
async def get_forecast(
    days: int = Query(default=7, ge=1, le=16, description="Number of forecast days (1-16)"),
    location_name: Optional[str] = Query(default=None),
    latitude: Optional[float] = Query(default=None),
    longitude: Optional[float] = Query(default=None),
    timezone: str = Query(default="auto"),
) -> ForecastResponse:
    """
    Return a short-range daily temperature forecast (up to 16 days).

    Falls back to saved config location if no location params are provided.
    Returns `forecast_reliable_days: 7` to indicate the confidence boundary —
    days 0-6 are high-confidence meteorological forecast, days 7-15 are
    extended forecast with degrading accuracy.
    """
    saved = load_config()

    if (
        location_name is None
        and latitude is None
        and longitude is None
        and saved is not None
    ):
        return await fetch_forecast_series(
            location_name=saved.location_name,
            latitude=saved.latitude,
            longitude=saved.longitude,
            timezone=saved.timezone,
            days=days,
        )

    if location_name is None and (latitude is None or longitude is None):
        raise HTTPException(
            status_code=400,
            detail="Provide location_name or latitude+longitude, or save a config first",
        )

    return await fetch_forecast_series(
        location_name=location_name,
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
        days=days,
    )


# ─── Humidity scenario models ────────────────────────────────────────────────

class HumidityScenarioPoint(BaseModel):
    date: str
    humidity_pct: float


class HumidityScenarioSeries(BaseModel):
    name: str
    percentile: Optional[int] = None
    readings: list[HumidityScenarioPoint]


class HumidityScenarioResponse(BaseModel):
    source: str = "open-meteo"
    granularity: Literal["daily"] = "daily"
    methodology: str
    start_date: date
    end_date: date
    resolved_location: LocationResolution
    historical_years: int
    historical_year_range: list[int]
    climatology_window_days: int
    scenario_percentiles: list[int]
    upstream_urls: list[str]
    scenarios: list[HumidityScenarioSeries]


async def fetch_humidity_scenario_curves(config: AppConfig) -> HumidityScenarioResponse:
    """Build percentile-based humidity scenario curves using the same
    climatology methodology as fetch_scenario_curves for temperature.
    Fetches historical humidity archive in a single cached request.
    """
    resolved = await resolve_location(config)
    target_days = iter_dates(config.start_date, config.end_date)
    historical_years = list(
        range(config.start_date.year - config.historical_years, config.start_date.year)
    )

    expanded_ranges: list[tuple[int, date, date]] = []
    for historical_year in historical_years:
        translated_start = safe_anchor_date(config.start_date, historical_year)
        translated_end   = safe_anchor_date(config.end_date,   historical_year)
        expanded_ranges.append((
            historical_year,
            translated_start - timedelta(days=config.climatology_window_days),
            translated_end   + timedelta(days=config.climatology_window_days),
        ))

    combined_start = min(r for _, r, _ in expanded_ranges)
    combined_end   = max(r for _, _, r in expanded_ranges)

    async with httpx.AsyncClient(timeout=60.0) as client:
        upstream_url, times, values = await fetch_humidity_archive_bucket(
            client=client,
            latitude=resolved.latitude,
            longitude=resolved.longitude,
            start_date=combined_start,
            end_date=combined_end,
            timezone=config.timezone,
        )

    lookup: dict[str, float] = {t: float(v) for t, v in zip(times, values)}

    percentile_points: dict[int, list[HumidityScenarioPoint]] = {
        p: [] for p in config.scenario_percentiles
    }
    baseline_points: list[HumidityScenarioPoint] = []

    for target_day in target_days:
        samples: list[float] = []
        for historical_year in historical_years:
            anchor = safe_anchor_date(target_day, historical_year)
            for offset in range(-config.climatology_window_days, config.climatology_window_days + 1):
                value = lookup.get((anchor + timedelta(days=offset)).isoformat())
                if value is not None:
                    samples.append(value)

        if not samples:
            raise HTTPException(
                status_code=502,
                detail=f"No humidity samples available for {target_day.isoformat()}",
            )

        baseline_points.append(HumidityScenarioPoint(
            date=target_day.isoformat(),
            humidity_pct=round(fmean(samples), 2),
        ))
        np_samples = np.asarray(samples, dtype=float)
        for p in config.scenario_percentiles:
            percentile_points[p].append(HumidityScenarioPoint(
                date=target_day.isoformat(),
                humidity_pct=round(float(np.percentile(np_samples, p)), 2),
            ))

    scenarios: list[HumidityScenarioSeries] = [
        HumidityScenarioSeries(name="baseline_mean", percentile=None, readings=baseline_points)
    ]
    for p in config.scenario_percentiles:
        scenarios.append(HumidityScenarioSeries(
            name=label_for_percentile(p),
            percentile=p,
            readings=percentile_points[p],
        ))

    return HumidityScenarioResponse(
        methodology=(
            "Daily humidity scenario curves derived from the prior 10 years of "
            "Open-Meteo relative_humidity_2m_mean values, using the same percentile "
            "climatology methodology as /api/scenarios."
        ),
        start_date=config.start_date,
        end_date=config.end_date,
        resolved_location=resolved,
        historical_years=config.historical_years,
        historical_year_range=historical_years,
        climatology_window_days=config.climatology_window_days,
        scenario_percentiles=config.scenario_percentiles,
        upstream_urls=[upstream_url],
        scenarios=scenarios,
    )


@app.get("/api/scenarios/humidity", response_model=HumidityScenarioResponse)
async def get_humidity_scenarios(
    start_date: Optional[date] = Query(default=None),
    end_date: Optional[date] = Query(default=None),
    location_name: Optional[str] = Query(default=None),
    latitude: Optional[float] = Query(default=None),
    longitude: Optional[float] = Query(default=None),
    timezone: str = Query(default="auto"),
    historical_years: int = Query(default=10),
    climatology_window_days: int = Query(default=3),
    scenario_percentiles: str = Query(default="10,50,75,90"),
) -> HumidityScenarioResponse:
    """Percentile-based daily humidity scenario curves (same methodology as /api/scenarios)."""
    saved = load_config()

    if (
        start_date is None
        and end_date is None
        and location_name is None
        and latitude is None
        and longitude is None
        and saved is not None
    ):
        return await fetch_humidity_scenario_curves(saved)

    if start_date is None or end_date is None:
        raise HTTPException(
            status_code=400,
            detail="start_date and end_date are required unless a saved config exists",
        )

    try:
        config = AppConfig(
            start_date=start_date,
            end_date=end_date,
            location_name=location_name,
            latitude=latitude,
            longitude=longitude,
            timezone=timezone,
            granularity="daily",
            historical_years=historical_years,
            climatology_window_days=climatology_window_days,
            scenario_percentiles=[int(x) for x in scenario_percentiles.split(",") if x],
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return await fetch_humidity_scenario_curves(config)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
