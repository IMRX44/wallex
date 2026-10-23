"""Calibrated LightGBM classifier with per-decision explanations.

* Time-ordered split with an embargo gap (no look-ahead, no leakage across the split).
* Isotonic calibration so ``p=0.6`` really means ~60% hit rate; position size uses it.
* ``explain`` returns each feature's contribution (LightGBM's native SHAP values).
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FEATURE_FA = {
    "edge": "عمق تخفیف نسبت به پریمیوم عادی",
    "std": "نوسان پریمیوم",
    "std_ratio": "تغییر رژیم نوسان (۶ساعت/بلندمدت)",
    "baseline": "پریمیوم عادی بازار",
    "dev_now": "فاصله‌ی قیمت فعلی از پریمیوم عادی",
    "ref_ret_5": "حرکت ۵ دقیقه‌ای بایننس",
    "ref_ret_15": "حرکت ۱۵ دقیقه‌ای بایننس",
    "ref_ret_60": "حرکت ۱ ساعته‌ی بایننس",
    "mkt_ret_15": "حرکت ۱۵ دقیقه‌ای والکس",
    "mkt_ret_60": "حرکت ۱ ساعته‌ی والکس",
    "usdt_ret_15": "حرکت ۱۵ دقیقه‌ای تتر",
    "usdt_ret_60": "حرکت ۱ ساعته‌ی تتر",
    "log_vol_24h": "حجم ۲۴ ساعته",
    "revert_rate": "نرخ برگشت افت‌های قبلی این بازار",
    "n_dips": "تعداد افت‌های ۳ روز اخیر",
    "hour_sin": "ساعت روز",
    "hour_cos": "ساعت روز",
}


@dataclass
class ModelReport:
    n_train: int = 0
    n_test: int = 0
    base_rate: float = 0.0
    auc: float = float("nan")
    brier: float = float("nan")
    precision_at: dict = field(default_factory=dict)
    feature_importance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class ProbModel:
    def __init__(self, features: list[str], name: str = "model", seed: int = 7):
        self.features = list(features)
        self.name = name
        self.seed = seed
        self.booster = None
        self.iso = None
        self.report = ModelReport()

    # ------------------------------------------------------------------
    def _params(self) -> dict:
        return {
            "objective": "binary", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 40,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
            "verbose": -1, "seed": self.seed, "deterministic": True, "num_threads": 2,
        }

    def _train(self, X: pd.DataFrame, y: np.ndarray, Xv: pd.DataFrame, yv: np.ndarray):
        import lightgbm as lgb
        dtr = lgb.Dataset(X[self.features], label=y)
        dva = lgb.Dataset(Xv[self.features], label=yv, reference=dtr)
        return lgb.train(self._params(), dtr, num_boost_round=600, valid_sets=[dva],
                         callbacks=[lgb.early_stopping(50, verbose=False)])

    @staticmethod
    def _split(n: int, a: float, b: float, embargo: int) -> tuple[slice, slice, slice]:
        i1, i2 = int(n * a), int(n * b)
        return slice(0, max(0, i1 - embargo)), slice(i1, max(i1, i2 - embargo)), slice(i2, n)

    def fit(self, df: pd.DataFrame, label: str = "y", time_col: str = "ts", embargo: int = 30,
            min_rows: int = 300, min_pos: int = 30) -> ModelReport:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import brier_score_loss, roc_auc_score

        df = df.dropna(subset=self.features + [label]).sort_values(time_col).reset_index(drop=True)
        y = df[label].astype(int).to_numpy()
        if len(df) < min_rows or y.sum() < min_pos or (len(y) - y.sum()) < min_pos:
            log.warning("%s: not enough data to train (%d rows, %d positives)", self.name, len(df), int(y.sum()))
            self.report = ModelReport(n_train=len(df), base_rate=float(y.mean()) if len(y) else 0.0)
            return self.report

        # 1) evaluation run: train 70% | calibrate 15% | test 15% (embargo between blocks)
        tr, ca, te = self._split(len(df), 0.70, 0.85, embargo)
        booster = self._train(df.iloc[tr], y[tr], df.iloc[ca], y[ca])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(booster.predict(df.iloc[ca][self.features]), y[ca])
        p_te = iso.predict(booster.predict(df.iloc[te][self.features]))
        rep = ModelReport(n_train=tr.stop, n_test=len(y[te]), base_rate=float(y.mean()))
        if len(set(y[te])) > 1:
            rep.auc = float(roc_auc_score(y[te], p_te))
        rep.brier = float(brier_score_loss(y[te], p_te))
        for thr in (0.5, 0.55, 0.6, 0.65):
            m = p_te >= thr
            rep.precision_at[str(thr)] = {"n": int(m.sum()), "precision": float(y[te][m].mean()) if m.any() else None}

        # 2) deployment model: train on first 85%, calibrate on the most recent 15%
        tr2, ca2, _ = self._split(len(df), 0.85, 1.0, embargo)
        self.booster = self._train(df.iloc[tr2], y[tr2], df.iloc[ca2], y[ca2])
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.iso.fit(self.booster.predict(df.iloc[ca2][self.features]), y[ca2])
        imp = self.booster.feature_importance("gain")
        rep.feature_importance = {f: float(v) for f, v in sorted(zip(self.features, imp), key=lambda x: -x[1])}
        self.report = rep
        log.info("%s trained: AUC=%.3f brier=%.4f base=%.3f", self.name, rep.auc, rep.brier, rep.base_rate)
        return rep

    @property
    def ready(self) -> bool:
        return self.booster is not None

    def predict(self, rows: pd.DataFrame | dict) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("model not trained")
        X = pd.DataFrame([rows]) if isinstance(rows, dict) else rows
        raw = self.booster.predict(X[self.features].astype(float))
        return self.iso.predict(raw) if self.iso is not None else raw

    def predict_one(self, row: dict) -> float:
        return float(self.predict(row)[0])

    def explain(self, row: dict, top: int = 4) -> list[tuple[str, float]]:
        if not self.ready:
            return []
        X = pd.DataFrame([row])[self.features].astype(float)
        contrib = self.booster.predict(X, pred_contrib=True)[0][:-1]
        pairs = sorted(zip(self.features, contrib), key=lambda x: -abs(x[1]))[:top]
        return [(f, float(c)) for f, c in pairs]

    def explain_fa(self, row: dict, top: int = 3) -> str:
        parts = []
        for f, c in self.explain(row, top):
            parts.append(f"{FEATURE_FA.get(f, f)} {'↑' if c > 0 else '↓'}")
        return "، ".join(parts)

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"features": self.features, "name": self.name,
                         "booster": self.booster.model_to_string() if self.booster else None,
                         "iso": self.iso, "report": self.report.to_dict()}, f)
        path.with_suffix(".json").write_text(json.dumps(self.report.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "ProbModel | None":
        path = Path(path)
        if not path.exists():
            return None
        import lightgbm as lgb
        with open(path, "rb") as f:
            d = pickle.load(f)
        m = cls(d["features"], d.get("name", "model"))
        if d.get("booster"):
            m.booster = lgb.Booster(model_str=d["booster"])
        m.iso = d.get("iso")
        m.report = ModelReport(**d.get("report", {}))
        return m
