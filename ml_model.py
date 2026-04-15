"""
ml_model.py
===========
Elite ML pipeline for Hydraulic Transient Analysis.

Architecture
------------
    DataGenerator  →  creates a rich synthetic dataset by sweeping penstock params
                       and measuring safe closure times via MOC simulation.
    FeatureEncoder →  normalises + engineers features for the model.
    MLModel        →  wraps a RandomForestRegressor with full training, cross-val,
                       evaluation, and prediction API.
    SafeRangeFinder→  post-hoc analyser that identifies the safe Tc band for a
                       given parameter set.

All classes are self-contained; no external state is shared.
"""

import numpy as np
import pickle
import os
import json
from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional, Dict, List

# ── lazy sklearn imports so the module loads even without sklearn installed ──
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# ── import simulation engine (same package) ──────────────────────────────────
from moc_module import PenstockParams, SimulationEngine


# ═══════════════════════════════════════════════════════════════════════════════
#  1. DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SampleRecord:
    """One training sample: 7 penstock features → target safe_tc (seconds)."""
    length:              float
    diameter:            float
    wave_speed:          float
    initial_velocity:    float
    initial_pressure_head: float
    max_pressure_head:   float
    min_pressure_head:   float
    # ── engineered features (computed, not user inputs) ──────────────────
    joukowsky_rise:      float   # c*V0/g  — theoretical max ΔH
    wave_period:         float   # 2L/c    — characteristic time
    head_margin:         float   # (Hmax - H0) / H0
    velocity_ratio:      float   # V0 / (wave_speed / 9.81)
    pipe_aspect:         float   # L / D
    # ── target ──────────────────────────────────────────────────────────
    safe_tc:             float   # minimum safe closure time [s]
    is_safe_at_tc:       bool


class DataGenerator:
    """
    Sweeps parameter space and finds the minimum safe Tc for each sample
    using a binary-search + MOC simulation strategy.

    Parameters
    ----------
    n_samples : int
        Number of random penstock configurations to generate.
    tc_search_points : int
        How many candidate Tc values to test in the binary search per sample.
    seed : int
        Random seed for reproducibility.
    """

    # Parameter space boundaries (matching Dashboard ranges)
    PARAM_BOUNDS = {
        "length":               (200,  4000),
        "diameter":             (0.6,   8.0),
        "wave_speed":           (600, 1800),
        "initial_velocity":     (0.5,   8.0),
        "initial_pressure_head":(60,   400),
        "max_head_margin":      (0.15, 0.55),  # (Hmax-H0)/H0
        "min_head_fraction":    (0.05, 0.20),  # H_min = H0 * frac
    }

    def __init__(self, n_samples: int = 2000, tc_search_points: int = 20, seed: int = 42):
        self.n_samples         = n_samples
        self.tc_search_points  = tc_search_points
        self.rng               = np.random.default_rng(seed)
        self._engine           = SimulationEngine()

    # ── public ──────────────────────────────────────────────────────────────

    def generate(self, verbose: bool = True) -> List[SampleRecord]:
        records: List[SampleRecord] = []
        failed = 0

        for i in range(self.n_samples):
            if verbose and i % 200 == 0:
                print(f"  Generating sample {i}/{self.n_samples}  (failures so far: {failed})")

            try:
                rec = self._generate_one()
                if rec is not None:
                    records.append(rec)
                else:
                    failed += 1
            except Exception:
                failed += 1

        if verbose:
            print(f"  Done. {len(records)} valid samples, {failed} failures.")
        return records

    # ── private ─────────────────────────────────────────────────────────────

    def _generate_one(self) -> Optional[SampleRecord]:
        b = self.PARAM_BOUNDS
        L  = self.rng.uniform(*b["length"])
        D  = self.rng.uniform(*b["diameter"])
        c  = self.rng.uniform(*b["wave_speed"])
        V0 = self.rng.uniform(*b["initial_velocity"])
        H0 = self.rng.uniform(*b["initial_pressure_head"])
        hm_frac = self.rng.uniform(*b["max_head_margin"])
        Hmax = H0 * (1.0 + hm_frac)
        Hmin = H0 * self.rng.uniform(*b["min_head_fraction"])

        params = PenstockParams(
            length=L, diameter=D, wave_speed=c,
            initial_velocity=V0, initial_pressure_head=H0,
            max_pressure_head=Hmax, min_pressure_head=Hmin,
            friction_factor=0.015, n_segments=50   # coarser grid for speed
        )
        if not params.validate():
            return None

        safe_tc = self._find_min_safe_tc(params)
        if safe_tc is None:
            return None

        # engineered features
        jrise       = c * V0 / 9.81
        wave_period = 2.0 * L / c
        head_margin = (Hmax - H0) / H0
        vel_ratio   = V0 / (c / 9.81)
        pipe_aspect = L / D

        return SampleRecord(
            length=L, diameter=D, wave_speed=c,
            initial_velocity=V0, initial_pressure_head=H0,
            max_pressure_head=Hmax, min_pressure_head=Hmin,
            joukowsky_rise=jrise, wave_period=wave_period,
            head_margin=head_margin, velocity_ratio=vel_ratio,
            pipe_aspect=pipe_aspect,
            safe_tc=safe_tc, is_safe_at_tc=True
        )

    def _find_min_safe_tc(self, params: PenstockParams) -> Optional[float]:
        """
        Binary-search for the smallest Tc that keeps the system safe.
        Tc search range: [wave_period/4,  wave_period * 10]
        """
        wave_period = 2.0 * params.length / params.wave_speed
        tc_min_search = max(1.0, wave_period / 4.0)
        tc_max_search = min(60.0, wave_period * 10.0)

        if tc_min_search >= tc_max_search:
            return None

        # Check that a long closure is indeed safe (sanity check)
        result = self._engine.run_simulation(params, tc_max_search)
        if not result.is_safe:
            return None  # even slow closure is unsafe for this config — skip

        # Binary search
        lo, hi = tc_min_search, tc_max_search
        for _ in range(self.tc_search_points):
            mid = (lo + hi) / 2.0
            res = self._engine.run_simulation(params, mid)
            if res.is_safe:
                hi = mid  # can try even shorter
            else:
                lo = mid  # too short, need longer

        return round(hi, 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  2. FEATURE ENCODER
# ═══════════════════════════════════════════════════════════════════════════════

RAW_FEATURES = [
    "length", "diameter", "wave_speed", "initial_velocity",
    "initial_pressure_head", "max_pressure_head", "min_pressure_head",
]
ENGINEERED_FEATURES = [
    "joukowsky_rise", "wave_period", "head_margin",
    "velocity_ratio", "pipe_aspect",
]
ALL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES


def _engineer(raw: Dict) -> np.ndarray:
    """Compute engineered features from a raw parameter dict."""
    L   = raw["length"]
    c   = raw["wave_speed"]
    V0  = raw["initial_velocity"]
    H0  = raw["initial_pressure_head"]
    Hmax= raw["max_pressure_head"]
    D   = raw["diameter"]
    g   = 9.81

    jrise       = c * V0 / g
    wave_period = 2.0 * L / c
    head_margin = (Hmax - H0) / max(H0, 1e-6)
    vel_ratio   = V0 / (c / g)
    pipe_aspect = L / max(D, 1e-6)

    return np.array([jrise, wave_period, head_margin, vel_ratio, pipe_aspect])


def records_to_arrays(records: List[SampleRecord]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a list of SampleRecord objects to X (n,12) and y (n,) arrays."""
    X_rows, y_rows = [], []
    for r in records:
        raw_vals = np.array([getattr(r, f) for f in RAW_FEATURES])
        eng_vals = np.array([getattr(r, f) for f in ENGINEERED_FEATURES])
        X_rows.append(np.concatenate([raw_vals, eng_vals]))
        y_rows.append(r.safe_tc)
    return np.array(X_rows), np.array(y_rows)


def raw_dict_to_feature_vector(raw: Dict) -> np.ndarray:
    """Convert a user-supplied parameter dict to a 12-feature vector."""
    base = np.array([raw[f] for f in RAW_FEATURES])
    eng  = _engineer(raw)
    return np.concatenate([base, eng]).reshape(1, -1)


# ═══════════════════════════════════════════════════════════════════════════════
#  3. ML MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingReport:
    """Human-readable summary of a training run."""
    n_samples:     int
    n_features:    int
    feature_names: List[str]
    cv_mae_mean:   float
    cv_mae_std:    float
    cv_r2_mean:    float
    cv_r2_std:     float
    test_mae:      float
    test_rmse:     float
    test_r2:       float
    feature_importances: Dict[str, float]

    def print(self):
        print("\n" + "═"*60)
        print("  TRAINING REPORT — Hydraulic Transient ML Model")
        print("═"*60)
        print(f"  Samples        : {self.n_samples}")
        print(f"  Features       : {self.n_features}")
        print(f"  CV  MAE        : {self.cv_mae_mean:.3f} ± {self.cv_mae_std:.3f} s")
        print(f"  CV  R²         : {self.cv_r2_mean:.4f} ± {self.cv_r2_std:.4f}")
        print(f"  Test MAE       : {self.test_mae:.3f} s")
        print(f"  Test RMSE      : {self.test_rmse:.3f} s")
        print(f"  Test R²        : {self.test_r2:.4f}")
        print("\n  Feature Importances:")
        for name, imp in sorted(self.feature_importances.items(),
                                key=lambda x: -x[1]):
            bar = "█" * int(imp * 40)
            print(f"    {name:<30s} {imp:.4f}  {bar}")
        print("═"*60 + "\n")

    def to_dict(self) -> Dict:
        return asdict(self)


class MLModel:
    """
    Predicts the minimum safe guide-vane closure time (Tc) for a given
    penstock configuration using a trained RandomForestRegressor.

    Usage
    -----
        model = MLModel()
        model.train(records)          # train from SampleRecord list
        tc = model.predict(params)    # PenstockParams → float
        model.save("model.pkl")
        model2 = MLModel.load("model.pkl")
    """

    MODEL_VERSION = "1.0"

    def __init__(self):
        self._pipeline: Optional[Pipeline] = None
        self._report:   Optional[TrainingReport] = None
        self._loaded    = False

    # ── training ────────────────────────────────────────────────────────────

    def train(self, records: List[SampleRecord],
              test_split: float = 0.15,
              cv_folds:   int   = 5,
              verbose:    bool  = True) -> TrainingReport:

        if not SKLEARN_OK:
            raise ImportError("scikit-learn is required for training. pip install scikit-learn")

        X, y = records_to_arrays(records)
        n = len(X)

        # Train / test split (chronological — last 15% as held-out)
        split = int(n * (1 - test_split))
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        if verbose:
            print(f"\n  Training on {len(X_tr)} samples, testing on {len(X_te)}...")

        # Model: RandomForest with tuned hyperparams
        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=3,
            max_features=0.6,
            n_jobs=-1,
            random_state=42
        )
        scaler   = StandardScaler()
        pipeline = Pipeline([("scaler", scaler), ("rf", rf)])

        # Cross-validation on training set
        kf     = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_mae = -cross_val_score(pipeline, X_tr, y_tr, cv=kf,
                                   scoring="neg_mean_absolute_error")
        cv_r2  =  cross_val_score(pipeline, X_tr, y_tr, cv=kf, scoring="r2")

        # Final fit on all training data
        pipeline.fit(X_tr, y_tr)

        # Test metrics
        y_pred  = pipeline.predict(X_te)
        t_mae   = mean_absolute_error(y_te, y_pred)
        t_rmse  = np.sqrt(mean_squared_error(y_te, y_pred))
        t_r2    = r2_score(y_te, y_pred)

        # Feature importances (from RF, before scaling — RF is scale-invariant)
        importances = pipeline.named_steps["rf"].feature_importances_
        imp_dict    = {name: float(imp)
                       for name, imp in zip(ALL_FEATURES, importances)}

        report = TrainingReport(
            n_samples=n,
            n_features=len(ALL_FEATURES),
            feature_names=ALL_FEATURES,
            cv_mae_mean=float(cv_mae.mean()),
            cv_mae_std=float(cv_mae.std()),
            cv_r2_mean=float(cv_r2.mean()),
            cv_r2_std=float(cv_r2.std()),
            test_mae=t_mae,
            test_rmse=t_rmse,
            test_r2=t_r2,
            feature_importances=imp_dict
        )

        if verbose:
            report.print()

        self._pipeline = pipeline
        self._report   = report
        self._loaded   = True
        return report

    # ── prediction ──────────────────────────────────────────────────────────

    def predict(self, params: PenstockParams) -> float:
        """
        Returns predicted minimum safe closure time in seconds.
        Raises RuntimeError if the model has not been trained / loaded.
        """
        self._require_loaded()
        raw = {
            "length":               params.length,
            "diameter":             params.diameter,
            "wave_speed":           params.wave_speed,
            "initial_velocity":     params.initial_velocity,
            "initial_pressure_head":params.initial_pressure_head,
            "max_pressure_head":    params.max_pressure_head,
            "min_pressure_head":    params.min_pressure_head,
        }
        X = raw_dict_to_feature_vector(raw)
        tc_pred = float(self._pipeline.predict(X)[0])
        return max(1.0, round(tc_pred, 2))   # physical lower bound: 1 s

    def predict_with_confidence(self, params: PenstockParams) -> Dict:
        """
        Returns prediction plus per-tree spread as a proxy for uncertainty.
        """
        self._require_loaded()
        raw = {
            "length":               params.length,
            "diameter":             params.diameter,
            "wave_speed":           params.wave_speed,
            "initial_velocity":     params.initial_velocity,
            "initial_pressure_head":params.initial_pressure_head,
            "max_pressure_head":    params.max_pressure_head,
            "min_pressure_head":    params.min_pressure_head,
        }
        X  = raw_dict_to_feature_vector(raw)
        rf = self._pipeline.named_steps["rf"]
        X_scaled = self._pipeline.named_steps["scaler"].transform(X)
        tree_preds = np.array([t.predict(X_scaled)[0] for t in rf.estimators_])
        tc_mean  = float(np.mean(tree_preds))
        tc_std   = float(np.std(tree_preds))
        tc_low   = float(np.percentile(tree_preds, 10))
        tc_high  = float(np.percentile(tree_preds, 90))

        return {
            "predicted_tc": max(1.0, round(tc_mean, 2)),
            "std":          round(tc_std, 2),
            "p10":          round(tc_low, 2),
            "p90":          round(tc_high, 2),
            "confidence":   "high" if tc_std < 1.5 else ("medium" if tc_std < 4 else "low"),
        }

    # ── persistence ─────────────────────────────────────────────────────────

    def save(self, path: str = "model.pkl"):
        self._require_loaded()
        payload = {
            "version":  self.MODEL_VERSION,
            "pipeline": self._pipeline,
            "report":   self._report.to_dict() if self._report else None,
            "features": ALL_FEATURES,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"  Model saved → {path}")

    @classmethod
    def load(cls, path: str = "model.pkl") -> "MLModel":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = cls()
        obj._pipeline = payload["pipeline"]
        obj._loaded   = True
        if payload.get("report"):
            d = payload["report"]
            obj._report = TrainingReport(**d)
        print(f"  Model loaded ← {path}  (v{payload.get('version','?')})")
        return obj

    # ── misc ────────────────────────────────────────────────────────────────

    def validate_model(self) -> bool:
        return self._loaded and self._pipeline is not None

    @property
    def report(self) -> Optional[TrainingReport]:
        return self._report

    def _require_loaded(self):
        if not self._loaded:
            raise RuntimeError("Model not trained or loaded. Call .train() or MLModel.load().")


# ═══════════════════════════════════════════════════════════════════════════════
#  4. SAFE RANGE FINDER
# ═══════════════════════════════════════════════════════════════════════════════

class SafeRangeFinder:
    """
    For a given PenstockParams, sweeps Tc across a grid and classifies each
    closure time as safe or unsafe, returning the contiguous safe band.
    """

    def __init__(self, n_points: int = 40):
        self.n_points = n_points
        self._engine  = SimulationEngine()

    def find(self, params: PenstockParams) -> Dict:
        """
        Returns a dict with:
            tc_values   : array of tested Tc values
            is_safe     : boolean array
            peak_heads  : float array
            safe_min_tc : float  (smallest safe Tc)
            safe_max_tc : float  (largest tested, should always be safe)
            recommended : float  (safe_min_tc + 20% margin)
        """
        wave_period   = 2.0 * params.length / params.wave_speed
        tc_lo         = max(1.0, wave_period * 0.25)
        tc_hi         = min(60.0, wave_period * 8.0)
        tc_values     = np.linspace(tc_lo, tc_hi, self.n_points)

        is_safe    = []
        peak_heads = []
        min_heads  = []

        for tc in tc_values:
            res = self._engine.run_simulation(params, tc)
            is_safe.append(res.is_safe)
            peak_heads.append(res.head)
            min_heads.append(res.min_head)

        is_safe    = np.array(is_safe)
        peak_heads = np.array(peak_heads)

        safe_tc_vals   = tc_values[is_safe]
        safe_min_tc    = float(safe_tc_vals.min()) if len(safe_tc_vals) else float("nan")
        safe_max_tc    = float(safe_tc_vals.max()) if len(safe_tc_vals) else float("nan")
        recommended    = round(safe_min_tc * 1.20, 2) if not np.isnan(safe_min_tc) else float("nan")

        return {
            "tc_values":  tc_values.tolist(),
            "is_safe":    is_safe.tolist(),
            "peak_heads": peak_heads.tolist(),
            "min_heads":  min_heads,
            "safe_min_tc":  safe_min_tc,
            "safe_max_tc":  safe_max_tc,
            "recommended_tc": recommended,
            "wave_period":  wave_period,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  5. CONVENIENCE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_save(n_samples: int = 2000, output_path: str = "model.pkl") -> MLModel:
    """
    Full pipeline: generate data → train → evaluate → save.
    Returns the trained MLModel instance.
    """
    if not SKLEARN_OK:
        raise ImportError("scikit-learn required. pip install scikit-learn")

    print(f"\n{'═'*60}")
    print("  Hydraulic Transient ML — Training Pipeline")
    print(f"{'═'*60}")

    print(f"\n[1/3] Generating {n_samples} synthetic samples via MOC simulation...")
    gen     = DataGenerator(n_samples=n_samples, seed=42)
    records = gen.generate(verbose=True)

    print(f"\n[2/3] Training RandomForest model on {len(records)} samples...")
    model   = MLModel()
    model.train(records, verbose=True)

    print(f"\n[3/3] Saving model to '{output_path}'...")
    model.save(output_path)

    print("\n  Pipeline complete. ✓")
    return model


if __name__ == "__main__":
    train_and_save(n_samples=2000)
