"""
run_pipeline.py — оркестратор MVP-пайплайна.

Порядок:
    1) raw → panel (+ features, если batch_build это делает)
    2) train (M1/M2) → models/ + meta
    3) research OOT (ladder / tail / stress) → reports/research/
    4) batch_score → reports/scores/

Запуск из корня проекта:
    python run_pipeline.py
    python run_pipeline.py --skip-build
    python run_pipeline.py --only train,research

NOTE:
    Импорты подстройте под реальные имена модулей.
    API сюда не входит — поднимается отдельно через uvicorn.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Корень проекта в PYTHONPATH (если запускают как script)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Шаги (обёртки — чтобы pipeline не падал из‑за чужих print)
# ---------------------------------------------------------------------------

def step_build() -> None:
    """raw → data/processed/panel (+ features, если так устроено)."""
    try:
        from src.data.batch_build import build_processed_vitrine
    except ImportError:
        from src.data.batch_build import build_processed_vitrine  # type: ignore

    logger.info("batch_build: start")
    build_processed_vitrine()
    logger.info("batch_build: done")


def step_features() -> None:
    """
    panel → features (PIT + next_state).
    Пропустите, если build_processed_vitrine уже пишет features.
    """
    try:
        from src.features.build_features import build_features
    except ImportError:
        logger.warning("build_features not found — skip (assume batch_build wrote features)")
        return

    logger.info("build_features: start")
    build_features()
    logger.info("build_features: done")


def step_train() -> None:
    """features → models/*.joblib + meta.joblib (+ базовый OOT внутри train)."""
    try:
        from src.training.train import train
    except ImportError:
        try:
            from src.training.train import train
        except ImportError as e:
            raise ImportError(
                "Cannot import train(). Expected src.train.train or src.training.train"
            ) from e

    logger.info("train: start")
    train()
    logger.info("train: done")


def step_research() -> None:
    """
    OOT research: model ladder, tail ECL, stress, (bootstrap) → reports/research/.
    Для MVP можно временно no-op, если модуля ещё нет.
    """
    try:
        from src.research.run_research import run_research
    except ImportError:
        logger.warning("src.research.run_research not found — skip research step")
        return

    logger.info("research: start")
    run_research()
    logger.info("research: done")


def step_batch_score() -> None:
    """Витрина → PD/ECL scores → reports/scores/."""
    try:
        from src.scoring.batch_score import batch_score
    except ImportError as e:
        raise ImportError("Cannot import batch_score from src.scoring.batch_score") from e

    logger.info("batch_score: start")
    batch_score()
    logger.info("batch_score: done")


STEPS = {
    "build": step_build,
    "features": step_features,
    "train": step_train,
    "research": step_research,
    "score": step_batch_score,
}

# Полный MVP-порядок
DEFAULT_ORDER = ["build", "features", "train", "research", "score"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Credit migration PD — MVP pipeline")
    p.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated steps, e.g. train,research,score",
    )
    p.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip raw→panel/features (use existing processed data)",
    )
    p.add_argument(
        "--skip-research",
        action="store_true",
        help="Skip OOT research reports",
    )
    p.add_argument(
        "--skip-score",
        action="store_true",
        help="Skip batch scoring",
    )
    return p.parse_args()


def resolve_steps(args: argparse.Namespace) -> list[str]:
    if args.only.strip():
        steps = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = [s for s in steps if s not in STEPS]
        if unknown:
            raise SystemExit(f"Unknown steps: {unknown}. Choose from {list(STEPS)}")
        return steps

    steps = list(DEFAULT_ORDER)
    if args.skip_build:
        steps = [s for s in steps if s not in ("build", "features")]
    if args.skip_research:
        steps = [s for s in steps if s != "research"]
    if args.skip_score:
        steps = [s for s in steps if s != "score"]
    return steps


def main() -> None:
    args = parse_args()
    steps = resolve_steps(args)

    logger.info("Pipeline steps: %s", " → ".join(steps))
    t0 = time.perf_counter()

    for i, name in enumerate(steps, 1):
        t_step = time.perf_counter()
        logger.info("=== %s/%s %s ===", i, len(steps), name)
        try:
            STEPS[name]()
        except Exception:
            logger.exception("Step '%s' failed", name)
            raise SystemExit(1) from None
        logger.info(
            "=== %s done in %.1fs ===",
            name,
            time.perf_counter() - t_step,
        )

    logger.info(
        "Pipeline complete in %.1fs | steps=%s",
        time.perf_counter() - t0,
        steps,
    )


if __name__ == "__main__":
    main()