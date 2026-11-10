"""
Очистка apartments.csv и сравнение моделей: baseline, LinearRegression, Ridge, RandomForest.
Запуск: python train.py
Результат: таблица метрик в консоли + графики в папке plots/
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM = ["area_total", "area_living", "area_kitchen", "rooms", "floor", "floors_total",
       "year_built", "metro_time", "lat", "lon", "dist_center_km"]
CAT = ["house_type", "repair_state", "balcony_type", "metro_station"]
TARGET = "price_usd"
Path("plots").mkdir(exist_ok=True)


def load_and_clean(path="apartments.csv"):
    df = pd.read_csv(path)
    print(f"Загружено строк: {len(df)}")
    df = df.drop_duplicates(subset=["id"]) if df["id"].notna().any() else df.drop_duplicates()
    df = df.dropna(subset=[TARGET, "area_total", "lat", "lon"])
    # расстояние до центра Минска (пл. Независимости), км, по формуле гаверсинусов
    la, lo = np.radians(df["lat"]), np.radians(df["lon"])
    la0, lo0 = np.radians(53.8963), np.radians(27.5469)
    a = np.sin((la - la0) / 2) ** 2 + np.cos(la) * np.cos(la0) * np.sin((lo - lo0) / 2) ** 2
    df["dist_center_km"] = 6371 * 2 * np.arcsin(np.sqrt(a))
    # грубая чистка явных ошибок ввода
    df = df[(df[TARGET] > 5_000) & (df[TARGET] < 1_000_000)]
    df = df[(df["area_total"] > 10) & (df["area_total"] < 300)]
    df = df[df["rooms"].fillna(1).between(1, 8)]
    df = df[df["year_built"].isna() | df["year_built"].between(1850, 2030)]
    # этаж не может быть выше этажности
    bad = df["floor"].notna() & df["floors_total"].notna() & (df["floor"] > df["floors_total"])
    df = df[~bad]
    # редкие категории -> "other"
    for c in CAT:
        df[c] = df[c].fillna("unknown").astype(str)
        top = df[c].value_counts()
        df[c] = df[c].where(df[c].map(top) >= 10, "other")
    print(f"После очистки: {len(df)} строк")
    return df


def make_pre():
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])


def main():
    df = load_and_clean()
    X, y = df[NUM + CAT], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Baseline (среднее)": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "Ridge": RidgeCV(alphas=np.logspace(-2, 3, 20)),
        "RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                              random_state=42, n_jobs=-1),
    }

    results, preds, fitted = [], {}, {}
    for name, model in models.items():
        pipe = Pipeline([("pre", make_pre()), ("m", model)]).fit(Xtr, ytr)
        p = pipe.predict(Xte)
        preds[name], fitted[name] = p, pipe
        results.append({
            "model": name,
            "MAE, $": mean_absolute_error(yte, p),
            "RMSE, $": mean_squared_error(yte, p) ** 0.5,
            "R2": r2_score(yte, p),
        })
    res = pd.DataFrame(results).round(2)
    print("\n", res.to_string(index=False))
    res.to_csv("plots/metrics.csv", index=False)

    # график 1: сравнение моделей по MAE
    plt.figure(figsize=(7, 4))
    plt.bar(res["model"], res["MAE, $"], color="#2a6fdb")
    plt.ylabel("MAE, $")
    plt.title("Средняя ошибка по моделям (меньше = лучше)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("plots/mae_by_model.png", dpi=150)
    plt.close()

    # график 2: предсказание vs факт у лучшей модели
    best = res.sort_values("MAE, $").iloc[0]["model"]
    p = preds[best]
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(yte, p, s=12, alpha=0.5)
    lim = [min(yte.min(), p.min()), max(yte.max(), p.max())]
    plt.plot(lim, lim, "r--")
    plt.xlabel("Цена в объявлении, $")
    plt.ylabel("Предсказание, $")
    plt.title(f"{best}: предсказание vs факт")
    plt.tight_layout()
    plt.savefig("plots/pred_vs_actual.png", dpi=150)
    plt.close()

    # график 3: важность признаков у случайного леса
    rf = fitted["RandomForest"]
    names = rf.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(rf.named_steps["m"].feature_importances_, index=names).nlargest(12)
    plt.figure(figsize=(7, 4.5))
    imp[::-1].plot.barh(color="#2a6fdb")
    plt.title("Важность признаков (RandomForest)")
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png", dpi=150)
    plt.close()
    print(f"\nЛучшая модель по MAE: {best}. Графики в папке plots/")


if __name__ == "__main__":
    main()
