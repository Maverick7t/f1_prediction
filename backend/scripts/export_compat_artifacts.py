"""Export compatibility-safe model artifacts.

Goal
- Avoid pickled sklearn/xgboost objects that break across versions.
- Export:
  - models_spencer/metadata_compat.json (feature_cols + encoder class lists)
  - models_spencer/xgb_winner.json and xgb_podium.json (native XGBoost format)

Run (from repo root):
  conda activate medibot
  python backend/scripts/export_compat_artifacts.py

This is an "Option 2" fix: re-export artifacts to match the deployment runtime.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_meta(meta_path: Path) -> dict:
    meta = joblib.load(meta_path)
    if not isinstance(meta, dict):
        raise TypeError(f"metadata is not a dict: {type(meta)}")
    if "feature_cols" not in meta or "encoders" not in meta:
        raise KeyError("metadata missing required keys: 'feature_cols' and/or 'encoders'")
    return meta


def _export_metadata_compat(meta: dict, out_path: Path) -> None:
    encoders = meta.get("encoders") or {}
    encoder_classes: dict[str, list[str]] = {}

    for name, enc in encoders.items():
        # Support sklearn LabelEncoder (classes_) and pre-exported lists.
        classes = None
        if hasattr(enc, "classes_"):
            classes = list(enc.classes_)
        elif isinstance(enc, (list, tuple)):
            classes = list(enc)

        if classes is None:
            raise TypeError(f"Unsupported encoder type for '{name}': {type(enc)}")

        encoder_classes[str(name)] = [str(c) for c in classes]

    payload = {
        "feature_cols": list(meta.get("feature_cols") or []),
        "encoders": encoder_classes,
        "format_version": 1,
    }

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _export_xgb_json(joblib_path: Path, out_path: Path) -> None:
    obj = joblib.load(joblib_path)
    model = obj.get("model") if isinstance(obj, dict) else obj

    # xgboost.sklearn.XGBClassifier has save_model(); Booster also has save_model().
    if not hasattr(model, "save_model"):
        raise TypeError(f"Loaded object does not support save_model(): {type(model)}")

    model.save_model(str(out_path))


def main() -> int:
    repo = _repo_root()
    backend_dir = repo / "backend"
    sys.path.insert(0, str(backend_dir))

    from utils.config import config  # pylint: disable=import-error

    model_dir = Path(config.MODEL_DIR)
    meta_path = Path(config.META_FILE)

    winner_joblib = Path(config.MODEL_WIN_FILE)
    podium_joblib = Path(config.MODEL_POD_FILE)

    out_meta = model_dir / "metadata_compat.json"
    out_winner = model_dir / "xgb_winner.json"
    out_podium = model_dir / "xgb_podium.json"

    print("Exporting compat artifacts")
    print("- model_dir:", model_dir)
    print("- meta:", meta_path)
    print("- winner joblib:", winner_joblib)
    print("- podium joblib:", podium_joblib)

    meta = _load_meta(meta_path)
    _export_metadata_compat(meta, out_meta)
    print("✓ wrote", out_meta)

    _export_xgb_json(winner_joblib, out_winner)
    print("✓ wrote", out_winner)

    _export_xgb_json(podium_joblib, out_podium)
    print("✓ wrote", out_podium)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
