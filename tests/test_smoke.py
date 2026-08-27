from pathlib import Path

from promptforge.comparison import infer_changes
from promptforge.evaluation import (
    information_preservation,
    instruction_preservation,
)
from promptforge.local_paths import ensure_dirs, resolve_model_path
from promptforge.scorer import PromptQualityScorer


def test_infer_issues_on_low_scores():
    dims = {
        "clarity": 30,
        "specificity": 15,
        "context": 10,
        "goal_definition": 25,
        "constraints": 5,
        "completeness": 15,
        "actionability": 20,
    }
    issues = PromptQualityScorer._infer_issues(dims)
    assert "too_vague" in issues
    assert "missing_context" in issues


def test_generate_example_schema():
    from promptforge.data.generate import generate_example

    ex = generate_example()
    assert "prompt" in ex
    assert 0 <= ex["quality_score"] <= 100
    assert ex["task_type"] in {"coding", "writing", "research", "data", "creative"}


def test_infer_changes():
    changes = infer_changes(
        "Make an app.",
        "Build a production-ready REST API for developers.\nRequirements:\n- Use Python\nReturn JSON.",
    )
    assert any("Added" in c or "Expanded" in c for c in changes)


def test_preservation_metrics():
    original = "Build a Python API"
    optimized = "Build a production-ready Python API with FastAPI"
    assert instruction_preservation(original, optimized) > 0
    assert information_preservation(original, optimized) > 0.3


def test_ensure_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPTFORGE_HOME", str(tmp_path / "pf"))
    paths = ensure_dirs()
    assert paths["home"].exists()
    assert paths["models"].exists()


def test_resolve_model_path_env(monkeypatch):
    monkeypatch.setenv("PROMPTFORGE_QUALITY_MODEL", "my/quality-model")
    assert resolve_model_path(None, kind="quality") == "my/quality-model"
    assert resolve_model_path("explicit", kind="quality") == "explicit"
