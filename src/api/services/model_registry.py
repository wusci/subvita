from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List
import json

import joblib
import pandas as pd


@dataclass(frozen=True)
class ModelSpec:
    disease: str                  # e.g., "t2d"
    cycle: str                    # e.g., "2017-2018"
    model_path: Path              # joblib
    feature_list_path: Path       # json
    perm_importance_path: Optional[Path] = None


class ModelBundle:
    """
    Holds loaded artifacts for a disease model.
    """
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.model = joblib.load(spec.model_path)
        self.feature_list: List[str] = json.loads(spec.feature_list_path.read_text(encoding="utf-8"))

        self.global_top_features: List[str] = []
        if spec.perm_importance_path and spec.perm_importance_path.exists():
            df_imp = pd.read_csv(spec.perm_importance_path)
            self.global_top_features = (
                df_imp.sort_values("importance_mean", ascending=False)["feature"].head(10).tolist()
            )

    def build_X(self, payload: dict) -> pd.DataFrame:
        # One-row feature frame with exact model columns
        row = {col: payload.get(col, None) for col in self.feature_list}
        return pd.DataFrame([row], columns=self.feature_list)

    def predict_proba(self, payload: dict):
        X = self.build_X(payload)
        return self.model.predict_proba(X)[0]


class ModelRegistry:
    def __init__(self, specs: List[ModelSpec]):
        self.specs = {s.disease: s for s in specs}
        self.bundles: Dict[str, ModelBundle] = {}

    def load_all(self):
        for disease, spec in self.specs.items():
            self.bundles[disease] = ModelBundle(spec)

    def list_models(self) -> List[dict]:
        out = []
        for disease, spec in self.specs.items():
            out.append({
                "disease": disease,
                "cycle": spec.cycle,
                "model_path": str(spec.model_path),
                "num_features": self.bundles[disease].feature_list.__len__() if disease in self.bundles else None,
            })
        return out

    def get(self, disease: str) -> ModelBundle:
        if disease not in self.bundles:
            raise KeyError(f"Unknown disease '{disease}'. Available: {list(self.bundles.keys())}")
        return self.bundles[disease]
