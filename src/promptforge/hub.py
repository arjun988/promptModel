from __future__ import annotations

from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from promptforge.local_paths import (
    default_optimizer_dir,
    default_quality_dir,
    ensure_dirs,
    save_user_config,
    load_user_config,
)


DEFAULT_QUALITY_REPO = "promptforge/PromptForge-Quality"
DEFAULT_OPTIMIZER_REPO = "promptforge/PromptForge-Optimizer"


def download_model(
    repo_id: str,
    local_dir: str | Path,
    token: str | None = None,
    revision: str | None = None,
) -> Path:
    """Download a Hugging Face model/adapter snapshot into local_dir."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        token=token,
        revision=revision,
    )
    return Path(path)


def download_quality_model(
    repo_id: str = DEFAULT_QUALITY_REPO,
    local_dir: str | Path | None = None,
    token: str | None = None,
) -> Path:
    ensure_dirs()
    target = Path(local_dir) if local_dir else default_quality_dir()
    return download_model(repo_id, target, token=token)


def download_optimizer_model(
    repo_id: str = DEFAULT_OPTIMIZER_REPO,
    local_dir: str | Path | None = None,
    token: str | None = None,
) -> Path:
    ensure_dirs()
    target = Path(local_dir) if local_dir else default_optimizer_dir()
    return download_model(repo_id, target, token=token)


def download_defaults(
    quality_repo: str = DEFAULT_QUALITY_REPO,
    optimizer_repo: str = DEFAULT_OPTIMIZER_REPO,
    token: str | None = None,
    update_config: bool = True,
) -> dict[str, Path]:
    """Download both Phase-1 and Phase-2 models into ~/.promptforge/models."""
    quality = download_quality_model(repo_id=quality_repo, token=token)
    optimizer = download_optimizer_model(repo_id=optimizer_repo, token=token)

    if update_config:
        cfg = load_user_config()
        cfg["quality_model_path"] = str(quality)
        cfg["optimizer_model_path"] = str(optimizer)
        cfg.setdefault("prefer_gpu", True)
        save_user_config(cfg)

    return {"quality": quality, "optimizer": optimizer}


def doctor_report() -> dict[str, Any]:
    """Local environment diagnostics for Phase 4."""
    import torch

    from promptforge import __version__
    from promptforge.local_paths import (
        default_config_path,
        models_dir,
        prefer_gpu_from_env,
        promptforge_home,
        resolve_model_path,
    )

    quality = resolve_model_path(None, kind="quality", allow_missing=True)
    optimizer = resolve_model_path(None, kind="optimizer", allow_missing=True)

    def _ready(path: str | None) -> bool:
        if not path:
            return False
        p = Path(path)
        if p.exists():
            return True
        # Hub repo id heuristic (org/name) — not downloaded yet, but configured
        return "/" in path and not path.startswith(".") and not p.is_absolute()

    return {
        "promptforge_version": __version__,
        "home": str(promptforge_home()),
        "config_path": str(default_config_path()),
        "config_exists": default_config_path().exists(),
        "models_dir": str(models_dir()),
        "quality_model": quality,
        "quality_model_ready": _ready(quality),
        "optimizer_model": optimizer,
        "optimizer_model_ready": _ready(optimizer),
        "prefer_gpu": prefer_gpu_from_env(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
