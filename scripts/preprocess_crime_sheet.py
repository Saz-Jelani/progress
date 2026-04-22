from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import math
import re

import pandas as pd


CURRENT_DATE = pd.Timestamp("2026-04-09")


PLACE_NORMALIZATION = {
    "new market": "newmarket",
    "shahbagh": "shahbag",
    "bongshal": "bangshal",
}

CRIME_NORMALIZATION = {
    "roberry": "robbery",
    "harrasement": "harassment",
    "harassement": "harassment",
    "shoot": "shot",
    "accdient": "accident",
}


@dataclass
class KnownPoint:
    place: str
    lat: float
    lon: float


def parse_mixed_date(value: object) -> pd.Timestamp | pd.NaT:
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        parsed = value
    elif isinstance(value, datetime):
        parsed = pd.Timestamp(value)
    else:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return pd.NaT

        iso_match = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})(?:\s+.*)?", text)
        dmy_match = re.fullmatch(r"(\d{1,2})-(\d{1,2})-(\d{4})(?:\s+.*)?", text)
        slash_match = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})(?:\s+.*)?", text)

        try:
            if iso_match:
                year, month, day = map(int, iso_match.groups())
                parsed = pd.Timestamp(year=year, month=month, day=day)
            elif dmy_match:
                day, month, year = map(int, dmy_match.groups())
                parsed = pd.Timestamp(year=year, month=month, day=day)
            elif slash_match:
                day, month, year = map(int, slash_match.groups())
                parsed = pd.Timestamp(year=year, month=month, day=day)
            else:
                parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
        except ValueError:
            return pd.NaT

    if pd.isna(parsed):
        return pd.NaT

    # Keep the same month/day while shifting obvious future dates back in time.
    while parsed > CURRENT_DATE:
        try:
            parsed = parsed.replace(year=parsed.year - 1)
        except ValueError:
            parsed = parsed - pd.DateOffset(years=1)
    return parsed.normalize()


def normalize_time(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text or text == "nan":
        return None
    text = re.sub(r"\s+", "", text).replace(".", ":")
    for fmt in ("%I:%M%p", "%H:%M"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.strftime("%H:%M")
        except ValueError:
            continue
    return None


def normalize_text(value: object, mapping: dict[str, str] | None = None) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text or text == "nan":
        return None
    if mapping:
        return mapping.get(text, text)
    return text


def parse_coordinates(value: object) -> tuple[float | None, float | None]:
    if pd.isna(value):
        return None, None
    text = str(value).strip()
    if not text or text.lower() == "nan" or "," not in text:
        return None, None
    left, right = text.split(",", 1)
    try:
        return float(left.strip()), float(right.strip())
    except ValueError:
        return None, None


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(a))


def infer_place(lat: float | None, lon: float | None, known_points: list[KnownPoint]) -> tuple[str | None, float | None]:
    if lat is None or lon is None:
        return None, None

    nearest: KnownPoint | None = None
    nearest_distance: float | None = None
    for point in known_points:
        dist = distance_km(lat, lon, point.lat, point.lon)
        if nearest_distance is None or dist < nearest_distance:
            nearest = point
            nearest_distance = dist

    if nearest is None:
        return None, None
    return nearest.place, nearest_distance


def build_known_points(frame: pd.DataFrame) -> list[KnownPoint]:
    grouped = (
        frame.dropna(subset=["Place_clean", "Latitude", "Longitude"])
        .groupby("Place_clean", as_index=False)[["Latitude", "Longitude"]]
        .median()
    )
    return [
        KnownPoint(place=row.Place_clean, lat=row.Latitude, lon=row.Longitude)
        for row in grouped.itertuples(index=False)
    ]


def main() -> None:
    source = Path("tmp_crime.xlsx")
    output_excel = Path("crime_data_cleaned.xlsx")
    output_csv = Path("crime_data_cleaned.csv")

    df = pd.read_excel(source)

    lat_lon = df["Lat, Lon"].apply(parse_coordinates)
    df["Latitude"] = [lat for lat, _ in lat_lon]
    df["Longitude"] = [lon for _, lon in lat_lon]

    df["Date_clean"] = df["Date"].apply(parse_mixed_date)
    df["Date_normalized"] = df["Date_clean"].dt.strftime("%Y-%m-%d")
    df["Time_normalized"] = df["Time"].apply(normalize_time)
    df["Shift_clean"] = df["Shift"].apply(normalize_text).str.title()
    df["Crime_type_clean"] = df["Crime type"].apply(
        lambda value: normalize_text(value, CRIME_NORMALIZATION)
    ).str.title()
    df["Gender_clean"] = df["Gender"].apply(normalize_text).str.title()
    df["Place_clean"] = df["Place"].apply(
        lambda value: normalize_text(value, PLACE_NORMALIZATION)
    )

    known_points = build_known_points(df)
    inferred_places: list[str | None] = []
    inference_distances: list[float | None] = []

    for row in df.itertuples(index=False):
        if row.Place_clean:
            inferred_places.append(row.Place_clean)
            inference_distances.append(0.0)
            continue
        place, distance = infer_place(row.Latitude, row.Longitude, known_points)
        inferred_places.append(place)
        inference_distances.append(distance)

    df["Place_filled"] = pd.Series(inferred_places).str.title()
    df["Place_inference_distance_km"] = inference_distances
    df["Place_was_missing"] = df["Place"].isna()

    cleaned = pd.DataFrame(
        {
            "Date": df["Date_normalized"],
            "Time": df["Time_normalized"],
            "Shift": df["Shift_clean"],
            "Place": df["Place_filled"],
            "Crime type": df["Crime_type_clean"],
            "Gender": df["Gender_clean"],
            "Latitude": df["Latitude"],
            "Longitude": df["Longitude"],
            "Link": df["Link"],
        }
    )

    cleaned.to_csv(output_csv, index=False)

    saved_excel_path = output_excel
    try:
        cleaned.to_excel(output_excel, index=False)
    except PermissionError:
        saved_excel_path = Path("crime_data_cleaned_updated.xlsx")
        cleaned.to_excel(saved_excel_path, index=False)

    summary = {
        "rows": len(cleaned),
        "valid_dates": int(cleaned["Date"].notna().sum()),
        "valid_times": int(cleaned["Time"].notna().sum()),
        "filled_places": int(cleaned["Place"].notna().sum()),
        "original_missing_places": int(df["Place_was_missing"].sum()),
        "coordinates_available": int(
            cleaned["Latitude"].notna().sum() and cleaned["Longitude"].notna().sum()
        ),
    }

    print("CLEANING SUMMARY")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"saved_excel: {saved_excel_path.resolve()}")
    print(f"saved_csv: {output_csv.resolve()}")


if __name__ == "__main__":
    main()
