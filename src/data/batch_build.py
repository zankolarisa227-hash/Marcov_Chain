"""
raw parquet → processed panel/features (батчами по файлам)
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import yaml

from src.data.loader import iter_raw_parquet, load_raw_file
from src.data.preprocess_panel import aggregate_transactions_to_panel
from src.features.build_features import build_feature_matrix, add_next_state
from src.features.leakage_checks import run_leakage_checks
from src.features.feature_contracts import FEATURE_COLS



def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def process_one_file(raw_file: Path, part_id: int, cfg: dict) -> dict:
    cpls = cfg["data"].get("raw_columns")
    raw_df = load_raw_file(raw_file)

    if part_id == 0:
        print("[batch_build] наименования колонок датасета:", list(raw_df.columns))
    print(f"[batch_build] загруженный файл {part_id + 1}: {raw_file.name}")

    panel = aggregate_transactions_to_panel(
        raw_df,
        max_months=cfg["data"]["max_months"],
        default_state=cfg["model"]["default_state"],
    )
    if part_id == 0:
        print("[batch_build] panel columns:", list(panel.columns))

    del raw_df
    gc.collect()

    features = build_feature_matrix(panel, min_history=cfg["data"]["min_history"])
    features = add_next_state(features)

    if part_id == 0:
        print("[batch_build] feature columns:", list(features.columns))
    run_leakage_checks(features, FEATURE_COLS)
    
    panel_dir = Path(cfg["paths"]["panel_dir"])
    feat_dir = Path(cfg["paths"]["features_dir"])
    panel_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    panel_path = panel_dir / f"part-{part_id:03d}.parquet"
    feat_path = feat_dir / f"part-{part_id:03d}.parquet"

    panel.to_parquet(panel_path, index=False)
    features.to_parquet(feat_path, index=False)

    n_panel, n_feat = len(panel), len(features)
    del panel, features
    gc.collect()

    print(
        f"[batch_build] file {part_id + 1} memory cleared "
        f"panel={n_panel} features={n_feat}"
)

    return {
        "part_id": part_id,
        "source": raw_file.name,
        "panel_rows": int(n_panel),      # ← используем сохранённые длины
        "feature_rows": int(n_feat),     # ← используем сохранённые длины
        "panel_path": str(panel_path),
        "features_path": str(feat_path),
    }

def build_processed_vitrine(config_path: str = "configs/default.yaml") -> list[dict]:
    cfg = load_config(config_path)
    files = list(iter_raw_parquet(cfg["paths"]["raw_dir"]))

    max_files = cfg["data"].get("max_files")
    if max_files is not None:
        files = files[: int(max_files)]

    manifests = []
    for i, f in enumerate(files):
        print(f"[batch_build] {f.name}")
        manifests.append(process_one_file(f, i, cfg))
       

    print(f"[batch_build] всего загружено файлов: {len(manifests)}")
    
    meta_dir = Path(cfg["paths"]["meta_dir"])
    meta_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = meta_dir / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifests


if __name__ == "__main__":
    build_processed_vitrine()