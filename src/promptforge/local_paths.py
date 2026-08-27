from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Env overrides
ENV_QUALITY = "PROMPTFORGE_QUALITY_MODEL"
ENV_OPTIMIZER = "PROMPTFORGE_OPTIMIZER_MODEL"
ENV_CONFIG = "PROMPTFORGE_CONFIG"
ENV_HOME = "PROMPTFORGE_HOME"
ENV_PREFER_GPU = "PROMPTFORGE_PREFER_GPU"


def repo_root() -> Path:
    """Best-effort repo root when developing from source."""
    here = Path(__file__).resolve()
    # src/promptforge/local_paths.py → repo root
    candidate = here.parents[2]
    if (candidate / "pyproject.toml").exists():
        return candidate
    return Path.cwd()


def promptforge_home() -> Path:
    """User-level PromptForge directory (models, config)."""
    if os.environ.get(ENV_HOME):
        return Path(os.environ[ENV_HOME]).expanduser()
    return Path.home() / ".promptforge"


def default_config_path() -> Path:
    if os.environ.get(ENV_CONFIG):
        return Path(os.environ[ENV_CONFIG]).expanduser()
    return promptforge_home() / "config.yaml"


def models_dir() -> Path:
    return promptforge_home() / "models"


def default_quality_dir() -> Path:
    return models_dir() / "PromptForge-Quality"


def default_optimizer_dir() -> Path:
    return models_dir() / "PromptForge-Optimizer"


def load_user_config(path: Path | None = None) -> dict[str, Any]:
    path = path or default_config_path()
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def save_user_config(data: dict[str, Any], path: Path | None = None) -> Path:
    path = path or default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return path


def _normalize_ref(value: str | Path) -> str:
    """Expand ~ for filesystem paths; leave Hub repo ids untouched."""
    text = str(value).strip()
    if text.startswith("~") or text.startswith("/") or (len(text) > 2 and text[1] == ":"):
        return str(Path(text).expanduser())
    # Windows absolute without drive already handled; Hub ids like org/name stay as-is
    if "\\" in text or text.startswith("."):
        return str(Path(text).expanduser())
    return text


def resolve_model_path(
    explicit: str | Path | None,
    *,
    kind: str,
    allow_missing: bool = False,
) -> str | None:
    """
    Resolve a model path/id in order:
      1) explicit CLI/API argument
      2) environment variable
      3) ~/.promptforge/config.yaml
      4) ~/.promptforge/models/PromptForge-*
      5) ./outputs/promptforge-*-model (dev convenience)
    """
    if explicit is not None and str(explicit).strip():
        return _normalize_ref(explicit)

    env_key = ENV_QUALITY if kind == "quality" else ENV_OPTIMIZER
    if os.environ.get(env_key):
        return _normalize_ref(os.environ[env_key])

    cfg = load_user_config()
    cfg_key = "quality_model_path" if kind == "quality" else "optimizer_model_path"
    if cfg.get(cfg_key):
        return _normalize_ref(cfg[cfg_key])

    local_default = default_quality_dir() if kind == "quality" else default_optimizer_dir()
    if local_default.exists():
        return str(local_default)

    dev = repo_root() / "outputs" / (
        "promptforge-quality-model" if kind == "quality" else "promptforge-optimizer-model"
    )
    if dev.exists():
        return str(dev)

    if allow_missing:
        return None
    return None


def prefer_gpu_from_env(default: bool = True) -> bool:
    raw = os.environ.get(ENV_PREFER_GPU)
    if raw is None:
        cfg = load_user_config()
        if "prefer_gpu" in cfg:
            return bool(cfg["prefer_gpu"])
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def ensure_dirs() -> dict[str, Path]:
    home = promptforge_home()
    models = models_dir()
    home.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    return {"home": home, "models": models, "config": default_config_path()}
