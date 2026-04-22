from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATASET_PATH = Path("crime_data_cleaned_weather_metrics.xlsx")
GRID_SIZE = 0.005
HOTSPOT_THRESHOLD = 0.55


@dataclass
class KnownPlace:
    name: str
    lat: float
    lng: float


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius_km = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lng / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(a))


def infer_shift(hour: int | None) -> str:
    if hour is None:
        return "Night"
    if 6 <= hour <= 11:
        return "Morning"
    if 12 <= hour <= 15:
        return "Noon"
    if 16 <= hour <= 18:
        return "Evening"
    return "Night"


def score_to_level(score: float) -> str:
    if score >= 0.75:
        return "High"
    if score >= 0.5:
        return "Medium"
    return "Low"


class RouteHotspotPredictor:
    def __init__(self, dataset_path: Path = DATASET_PATH) -> None:
        self.dataset_path = dataset_path
        self.ready = False
        self.error_message = ""
        self.model: Pipeline | None = None
        self.metrics: dict[str, float] = {}
        self.global_risk = 0.0
        self.hour_risk: dict[int, float] = {}
        self.day_risk: dict[str, float] = {}
        self.place_risk: dict[str, float] = {}
        self.context_place_risk: dict[tuple[str, str, int], float] = {}
        self.month_context_place_risk: dict[tuple[str, int, str, int], float] = {}
        self.weather_features: list[str] = []
        self.weather_feature_defaults: dict[str, float] = {}
        self.weather_profiles: dict[tuple[int, int, str], dict[str, float]] = {}
        self.known_places: list[KnownPlace] = []
        self.training_frame = pd.DataFrame()
        self.features = [
            "Place",
            "Shift",
            "hour",
            "hour_bin",
            "month",
            "weekend",
            "day_number",
            "month_week_no",
            "Latitude",
            "Longitude",
            "day_name",
            "place_risk",
            "hour_risk",
            "day_risk",
            "context_place_risk",
            "month_context_place_risk",
        ]
        self._load()

    def _load(self) -> None:
        if not self.dataset_path.exists():
            self.error_message = f"Dataset not found: {self.dataset_path}"
            return

        try:
            frame = self._read_dataset()
            prepared = self._prepare_training_frame(frame)
            if len(prepared) < 50:
                self.error_message = "Not enough cleaned rows available for hotspot prediction."
                return

            self.training_frame = prepared
            self.weather_features = [
                column
                for column in ["Temperature", "Precipitation", "WeatherCode", "Humidity", "CloudCover"]
                if column in prepared.columns
            ]
            self.features.extend([column for column in self.weather_features if column not in self.features])
            self.known_places = self._build_known_places(prepared)
            self.place_risk = prepared.groupby("Place")["is_hotspot"].mean().to_dict()
            self.hour_risk = prepared.groupby("hour")["is_hotspot"].mean().to_dict()
            self.day_risk = prepared.groupby("day_name")["is_hotspot"].mean().to_dict()
            self.context_place_risk = self._build_place_context_risk(prepared)
            self.month_context_place_risk = self._build_month_place_context_risk(prepared)
            self.global_risk = float(prepared["is_hotspot"].mean())
            if self.weather_features:
                self.weather_feature_defaults = (
                    prepared[self.weather_features].median(numeric_only=True).fillna(0.0).astype(float).to_dict()
                )
                self.weather_profiles = (
                    prepared.groupby(["month", "hour", "day_name"])[self.weather_features]
                    .median(numeric_only=True)
                    .fillna(0.0)
                    .to_dict("index")
                )
            self.metrics = self._benchmark(prepared)
            self.model = self._train_full_model(prepared)
            self.ready = True
        except Exception as exc:  # pragma: no cover - defensive guard
            self.error_message = str(exc)

    def _read_dataset(self) -> pd.DataFrame:
        if self.dataset_path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(self.dataset_path)
        return pd.read_csv(self.dataset_path)

    def _prepare_training_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.copy()
        data["Date_dt"] = pd.to_datetime(data["Date"], errors="coerce")
        data["Time_dt"] = pd.to_datetime(data["Time"], format="%H:%M", errors="coerce")
        data = data.dropna(subset=["Date_dt", "Time_dt", "Latitude", "Longitude", "Place"]).copy()

        data["Place"] = data["Place"].astype(str).str.strip()
        data["hour"] = data["Time_dt"].dt.hour.astype(int)
        data["hour_bin"] = (data["hour"] // 3).astype(int)
        data["month"] = data["Date_dt"].dt.month.astype(int)
        data["weekend"] = (data["Date_dt"].dt.dayofweek < 5).astype(int)
        data["day_number"] = data["Date_dt"].dt.day.astype(int)
        data["month_week_no"] = (((data["day_number"] - 1) // 7) + 1).astype(int)
        data["day_name"] = data["Date_dt"].dt.day_name().str.lower()
        data["Shift"] = data["hour"].apply(infer_shift)
        data["lat_bin"] = (data["Latitude"] / GRID_SIZE).round().astype(int)
        data["lng_bin"] = (data["Longitude"] / GRID_SIZE).round().astype(int)

        cell_count = data.groupby(["lat_bin", "lng_bin", "hour_bin", "day_name"]).size().rename("cell_count")
        data = data.join(cell_count, on=["lat_bin", "lng_bin", "hour_bin", "day_name"])
        data["is_hotspot"] = (data["cell_count"] >= 2).astype(int)

        data["place_risk"] = data.groupby("Place")["is_hotspot"].transform("mean")
        data["hour_risk"] = data.groupby("hour")["is_hotspot"].transform("mean")
        data["day_risk"] = data.groupby("day_name")["is_hotspot"].transform("mean")
        data["context_place_risk"] = data.groupby(["Place", "day_name", "hour_bin"])["is_hotspot"].transform("mean")
        data["month_context_place_risk"] = (
            data.groupby(["Place", "month", "day_name", "hour_bin"])["is_hotspot"].transform("mean")
        )
        return data.sort_values("Date_dt").reset_index(drop=True)

    def _build_model(self) -> Pipeline:
        categorical = ["Place", "Shift", "day_name"]
        numeric = [
            "hour",
            "hour_bin",
            "month",
            "weekend",
            "day_number",
            "month_week_no",
            "Latitude",
            "Longitude",
            "place_risk",
            "hour_risk",
            "day_risk",
            "context_place_risk",
            "month_context_place_risk",
            *self.weather_features,
        ]
        return Pipeline(
            steps=[
                (
                    "prep",
                    ColumnTransformer(
                        transformers=[
                            (
                                "cat",
                                Pipeline(
                                    steps=[
                                        ("imputer", SimpleImputer(strategy="most_frequent")),
                                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                                    ]
                                ),
                                categorical,
                            ),
                            (
                                "num",
                                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                                numeric,
                            ),
                        ]
                    ),
                ),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

    def _benchmark(self, data: pd.DataFrame) -> dict[str, float]:
        split_index = int(len(data) * 0.8)
        train = data.iloc[:split_index].copy()
        test = data.iloc[split_index:].copy()
        if train.empty or test.empty:
            return {}

        train_place_risk = train.groupby("Place")["is_hotspot"].mean().to_dict()
        train_hour_risk = train.groupby("hour")["is_hotspot"].mean().to_dict()
        train_day_risk = train.groupby("day_name")["is_hotspot"].mean().to_dict()
        train_context_place_risk = self._build_place_context_risk(train)
        train_month_context_place_risk = self._build_month_place_context_risk(train)
        train_global_risk = float(train["is_hotspot"].mean())

        for subset in (train, test):
            subset["place_risk"] = subset["Place"].map(train_place_risk).fillna(train_global_risk)
            subset["hour_risk"] = subset["hour"].map(train_hour_risk).fillna(train_global_risk)
            subset["day_risk"] = subset["day_name"].map(train_day_risk).fillna(train_global_risk)
            subset["context_place_risk"] = subset.apply(
                lambda row: train_context_place_risk.get(
                    (str(row["Place"]), str(row["day_name"]), int(row["hour_bin"])),
                    train_place_risk.get(str(row["Place"]), train_global_risk),
                ),
                axis=1,
            )
            subset["month_context_place_risk"] = subset.apply(
                lambda row: train_month_context_place_risk.get(
                    (str(row["Place"]), int(row["month"]), str(row["day_name"]), int(row["hour_bin"])),
                    train_context_place_risk.get(
                        (str(row["Place"]), str(row["day_name"]), int(row["hour_bin"])),
                        train_place_risk.get(str(row["Place"]), train_global_risk),
                    ),
                ),
                axis=1,
            )

        model = self._build_model()
        model.fit(train[self.features], train["is_hotspot"])
        prediction = model.predict(test[self.features])
        return {
            "accuracy": round(float(accuracy_score(test["is_hotspot"], prediction)), 4),
            "precision": round(float(precision_score(test["is_hotspot"], prediction, zero_division=0)), 4),
            "recall": round(float(recall_score(test["is_hotspot"], prediction, zero_division=0)), 4),
            "f1": round(float(f1_score(test["is_hotspot"], prediction, zero_division=0)), 4),
            "positive_rate": round(float(data["is_hotspot"].mean()), 4),
            "rows_used": int(len(data)),
        }

    def _train_full_model(self, data: pd.DataFrame) -> Pipeline:
        model = self._build_model()
        model.fit(data[self.features], data["is_hotspot"])
        return model

    def _build_known_places(self, data: pd.DataFrame) -> list[KnownPlace]:
        grouped = data.groupby("Place", as_index=False)[["Latitude", "Longitude"]].median()
        return [
            KnownPlace(name=row.Place, lat=float(row.Latitude), lng=float(row.Longitude))
            for row in grouped.itertuples(index=False)
        ]

    def _candidate_points_for_trip(self, trip_dt: pd.Timestamp) -> list[dict[str, float]]:
        if self.training_frame.empty:
            return [{"lat": place.lat, "lng": place.lng} for place in self.known_places]

        day_name = trip_dt.day_name().lower()
        month = int(trip_dt.month)
        hour_bin = int(trip_dt.hour) // 3

        candidate_frame = self.training_frame[
            (self.training_frame["day_name"] == day_name)
            & (self.training_frame["month"] == month)
            & (self.training_frame["hour_bin"].sub(hour_bin).abs() <= 1)
        ].copy()

        if len(candidate_frame) < 5:
            candidate_frame = self.training_frame[
                (self.training_frame["day_name"] == day_name)
                & (self.training_frame["hour_bin"].sub(hour_bin).abs() <= 1)
            ].copy()

        if len(candidate_frame) < 5:
            candidate_frame = self.training_frame[
                self.training_frame["hour_bin"].sub(hour_bin).abs() <= 1
            ].copy()

        if candidate_frame.empty:
            return [{"lat": place.lat, "lng": place.lng} for place in self.known_places]

        grouped = (
            candidate_frame.groupby("Place", as_index=False)[["Latitude", "Longitude"]]
            .median()
            .sort_values("Place")
        )
        return [
            {"lat": float(row.Latitude), "lng": float(row.Longitude)}
            for row in grouped.itertuples(index=False)
        ]

    def _nearest_place(self, lat: float, lng: float) -> tuple[str, float]:
        best_name = "Unknown"
        best_distance = float("inf")
        for place in self.known_places:
            distance = haversine_km(lat, lng, place.lat, place.lng)
            if distance < best_distance:
                best_distance = distance
                best_name = place.name
        return best_name, best_distance

    def _build_place_context_risk(self, data: pd.DataFrame) -> dict[tuple[str, str, int], float]:
        grouped = data.groupby(["Place", "day_name", "hour_bin"])["is_hotspot"].mean()
        return {
            (str(place), str(day_name), int(hour_bin)): float(value)
            for (place, day_name, hour_bin), value in grouped.items()
        }

    def _build_month_place_context_risk(self, data: pd.DataFrame) -> dict[tuple[str, int, str, int], float]:
        grouped = data.groupby(["Place", "month", "day_name", "hour_bin"])["is_hotspot"].mean()
        return {
            (str(place), int(month), str(day_name), int(hour_bin)): float(value)
            for (place, month, day_name, hour_bin), value in grouped.items()
        }

    def _build_feature_frame(
        self,
        candidate_points: list[dict[str, float]],
        trip_dt: pd.Timestamp,
    ) -> pd.DataFrame:
        hour = int(trip_dt.hour)
        hour_bin = hour // 3
        month = int(trip_dt.month)
        weekend = 1 if trip_dt.dayofweek < 5 else 0
        day_number = int(trip_dt.day)
        month_week_no = ((day_number - 1) // 7) + 1
        day_name = trip_dt.day_name().lower()
        shift = infer_shift(hour)
        weather_profile = self.weather_profiles.get((month, hour, day_name), self.weather_feature_defaults)
        rows: list[dict[str, Any]] = []

        for point in candidate_points:
            lat = float(point["lat"])
            lng = float(point["lng"])
            place_name, nearest_distance_km = self._nearest_place(lat, lng)
            rows.append(
                {
                    "Place": place_name,
                    "Shift": shift,
                    "hour": hour,
                    "hour_bin": hour_bin,
                    "month": month,
                    "weekend": weekend,
                    "day_number": day_number,
                    "month_week_no": month_week_no,
                    "Latitude": lat,
                    "Longitude": lng,
                    "day_name": day_name,
                    "place_risk": self.place_risk.get(place_name, self.global_risk),
                    "hour_risk": self.hour_risk.get(hour, self.global_risk),
                    "day_risk": self.day_risk.get(day_name, self.global_risk),
                    "context_place_risk": self.context_place_risk.get(
                        (place_name, day_name, hour_bin),
                        self.place_risk.get(place_name, self.global_risk),
                    ),
                    "month_context_place_risk": self.month_context_place_risk.get(
                        (place_name, month, day_name, hour_bin),
                        self.context_place_risk.get(
                            (place_name, day_name, hour_bin),
                            self.place_risk.get(place_name, self.global_risk),
                        ),
                    ),
                    "nearest_place_distance_km": round(nearest_distance_km, 3),
                    **{
                        feature: float(weather_profile.get(feature, self.weather_feature_defaults.get(feature, 0.0)))
                        for feature in self.weather_features
                    },
                }
            )
        return pd.DataFrame(rows)

    def _normalize_trip_datetime(self, date_text: str | None, time_text: str | None) -> pd.Timestamp:
        trip_date = pd.to_datetime(date_text, errors="coerce")
        if pd.isna(trip_date):
            trip_date = pd.Timestamp.now().normalize()

        trip_time = pd.to_datetime(time_text, format="%H:%M", errors="coerce")
        if pd.isna(trip_time):
            current = pd.Timestamp.now()
            return trip_date.normalize() + pd.Timedelta(hours=current.hour, minutes=current.minute)
        return trip_date.normalize() + pd.Timedelta(hours=trip_time.hour, minutes=trip_time.minute)

    def predict_hotspots(
        self,
        *,
        date_text: str | None,
        time_text: str | None,
        limit: int = 10,
    ) -> dict[str, Any]:
        if not self.ready or self.model is None:
            return {"ok": False, "error": self.error_message or "Prediction model is not ready."}

        trip_dt = self._normalize_trip_datetime(date_text, time_text)
        candidate_points = self._candidate_points_for_trip(trip_dt)
        feature_frame = self._build_feature_frame(candidate_points, trip_dt)
        probabilities = self.model.predict_proba(feature_frame[self.features])[:, 1]
        feature_frame["risk_score"] = probabilities
        feature_frame["risk_percent"] = (feature_frame["risk_score"] * 100).round(1)
        feature_frame["risk_level"] = feature_frame["risk_score"].apply(score_to_level)

        hotspot_rows = feature_frame.sort_values("risk_score", ascending=False).head(limit)
        hotspot_candidates = [
            {
                "place": row["Place"],
                "lat": round(float(row["Latitude"]), 6),
                "lng": round(float(row["Longitude"]), 6),
                "risk_score": round(float(row["risk_score"]), 4),
                "risk_percent": round(float(row["risk_percent"]), 1),
                "risk_level": row["risk_level"],
                "nearest_place_distance_km": round(float(row["nearest_place_distance_km"]), 3),
            }
            for _, row in hotspot_rows.iterrows()
        ]

        flagged_count = int((feature_frame["risk_score"] >= HOTSPOT_THRESHOLD).sum())
        top_score = float(feature_frame["risk_score"].max())
        return {
            "ok": True,
            "prediction": {
                "overall_risk_score": round(top_score, 4),
                "overall_risk_percent": round(top_score * 100, 1),
                "overall_risk_level": score_to_level(top_score),
                "flagged_hotspot_count": flagged_count,
                "candidate_place_count": int(len(feature_frame)),
                "day_of_week": trip_dt.day_name(),
                "date": trip_dt.strftime("%Y-%m-%d"),
                "time": trip_dt.strftime("%H:%M"),
                "hotspot_candidates": hotspot_candidates,
                "model": {
                    "name": "RandomForestClassifier",
                    "label_strategy": "Grid hotspot label on 500m cells + 3 hour bins + day_of_week",
                    "features": self.features,
                    "benchmark": self.metrics,
                },
            },
        }
