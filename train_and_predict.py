"""Freight rate prediction: validate, train, and write the submission files.

Run from the repo root:
    python train_and_predict.py

Outputs:
    validation_predictions.csv          (load_id, predicted_rate)
    data/december_chart_inputs.csv      (predicted_rate column filled in)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

DATA = Path("data")
RPM_MIN, RPM_MAX = 0.8, 6.0      # rate-per-mile range kept for TRAINING rows
SPLIT_DATE = "2025-09-01"        # time-based holdout: train Jan-Aug, test Sep-Oct
RANDOM_STATE = 42


# ----------------------------------------------------------------------------
# Data and features
# ----------------------------------------------------------------------------
def load_data():
    train = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
    val = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])
    return train, val


def drop_bad_rates(d: pd.DataFrame) -> pd.DataFrame:
    """Remove implausible labels (training rows only)."""
    d = d.copy()
    d["rpm"] = d["posted_rate"] / d["distance"]
    return d[d["rpm"].between(RPM_MIN, RPM_MAX)]


def make_features(d: pd.DataFrame, equipment_levels: list[str]) -> pd.DataFrame:
    """Features without city names (they generalise badly to unseen cities;
    coordinates carry the location information instead)."""
    X = pd.DataFrame(index=d.index)
    X["distance"] = d["distance"]
    X["weight"] = d["weight"].abs()            # negative weights = sign errors
    X["market_index"] = d["market_index"]      # NaN is handled by the model
    X["quote_signal"] = d["quote_signal"]
    for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        X[c] = d[c]
    X["dow"] = d["date"].dt.dayofweek          # month deliberately NOT used
    X["equipment"] = pd.Categorical(d["equipment"], categories=equipment_levels)
    return X


def new_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        categorical_features="from_dtype",
        learning_rate=0.05,
        max_iter=400,
        random_state=RANDOM_STATE,
    )


def fit(train: pd.DataFrame, equipment_levels: list[str]):
    clean = drop_bad_rates(train)
    model = new_model()
    # Target: log(rate per mile). Prediction = exp(output) * distance.
    model.fit(make_features(clean, equipment_levels), np.log(clean["rpm"]))
    return model


def predict(model, d: pd.DataFrame, equipment_levels: list[str]) -> np.ndarray:
    return np.exp(model.predict(make_features(d, equipment_levels))) * d["distance"].to_numpy()


def report(y_true, y_pred, name: str) -> None:
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs(y_true - y_pred) / y_true) * 100
    print(f"{name:40s} MAE: ${mae:,.0f}   MAPE: {mape:.1f}%")


# ----------------------------------------------------------------------------
# Step 1: time-based validation (train on the past, test on the future)
# ----------------------------------------------------------------------------
def run_validation(train_all: pd.DataFrame, equipment_levels: list[str]) -> None:
    past = train_all[train_all["date"] < SPLIT_DATE]
    future = train_all[train_all["date"] >= SPLIT_DATE]
    print(f"Time split: train {len(past):,} rows (to {past['date'].max().date()}), "
          f"test {len(future):,} rows (from {future['date'].min().date()})")

    # Baseline: distance x median rate-per-mile of each equipment type
    p = drop_bad_rates(past)
    rpm_by_equip = p.groupby("equipment")["rpm"].median()
    base = future["distance"] * future["equipment"].map(rpm_by_equip)
    report(future["posted_rate"], base, "Baseline (distance x equipment rpm)")

    model = fit(past, equipment_levels)
    pred = predict(model, future, equipment_levels)
    report(future["posted_rate"], pred, "Gradient boosting")


# ----------------------------------------------------------------------------
# Step 2: final model on all labelled data -> submission files
# ----------------------------------------------------------------------------
def write_validation_predictions(model, val, equipment_levels) -> pd.DataFrame:
    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    val = val.copy()
    val["predicted_rate"] = predict(model, val, equipment_levels)
    out = template[["load_id"]].merge(val[["load_id", "predicted_rate"]], on="load_id", how="left")
    assert len(out) == 12_000
    assert out["predicted_rate"].notna().all() and (out["predicted_rate"] > 0).all()
    out.to_csv("validation_predictions.csv", index=False)
    print(f"Wrote validation_predictions.csv ({len(out):,} rows)")
    return val


def write_december(model, train, val, equipment_levels) -> None:
    dec = pd.read_csv(DATA / "december_chart_inputs.csv")
    dates = pd.to_datetime(dec["date"])

    # The chart file has no market signals, so use each day's average
    # from validation.csv (which covers every December day).
    daily = val.groupby("date")[["market_index", "quote_signal"]].mean()

    lex = train.loc[train["pickup"] == "Lexington", ["pickup_lat", "pickup_lon"]].median()
    fw = train.loc[train["delivery"] == "Fort Wayne", ["delivery_lat", "delivery_lon"]].median()

    X = pd.DataFrame({
        "date": dates,
        "distance": dec["distance"].astype(float),
        "weight": dec["weight"].astype(float),
        "equipment": dec["equipment"],
        "pickup_lat": lex["pickup_lat"], "pickup_lon": lex["pickup_lon"],
        "delivery_lat": fw["delivery_lat"], "delivery_lon": fw["delivery_lon"],
        "market_index": dates.map(daily["market_index"]).to_numpy(),
        "quote_signal": dates.map(daily["quote_signal"]).to_numpy(),
    })
    dec["predicted_rate"] = predict(model, X, equipment_levels)
    dec.to_csv(DATA / "december_chart_inputs.csv", index=False)
    print("Wrote data/december_chart_inputs.csv "
          f"(min ${dec['predicted_rate'].min():,.0f}, max ${dec['predicted_rate'].max():,.0f})")


def main() -> None:
    train, val = load_data()
    equipment_levels = sorted(train["equipment"].unique())

    run_validation(train, equipment_levels)

    model = fit(train, equipment_levels)           # all of Jan-Oct
    val = write_validation_predictions(model, val, equipment_levels)
    write_december(model, train, val, equipment_levels)


if __name__ == "__main__":
    main()
