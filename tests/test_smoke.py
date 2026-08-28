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

    # Short rewrites should not score 0 when topic is preserved
    assert instruction_preservation("Make an app.", "Build a mobile application") >= 0.25


def test_ensure_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPTFORGE_HOME", str(tmp_path / "pf"))
    paths = ensure_dirs()
    assert paths["home"].exists()
    assert paths["models"].exists()

def test_generate_optimizer_example_schema():
    from promptforge.data.optimizer_generate import generate_optimizer_example

    ex = generate_optimizer_example()
    assert "prompt" in ex
    assert "optimized_prompt" in ex
    assert "messages_json" in ex
    assert "canonical_task" in ex
    assert ex["task_type"] in {"coding", "writing", "research", "data", "general", "creative"}


def test_optimizer_validation_rejects_repetition():
    from promptforge.optimizer_validation import detect_repetition, validate_optimization

    bad = "Return a step-by-step explanation.\n" * 5
    assert detect_repetition(bad)
    result = validate_optimization("Make an app.", bad)
    assert not result["valid"]
    assert "repetitive" in result["issues"]


def test_tokenize_sft_masks_prompt():
    from transformers import AutoTokenizer

    from promptforge.data.optimizer_generate import generate_optimizer_example
    from promptforge.optimizer_chat import tokenize_sft_messages

    ex = generate_optimizer_example()
    messages = __import__("json").loads(ex["messages_json"])
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    encoded = tokenize_sft_messages(tokenizer, messages, max_length=512)
    assert any(label == -100 for label in encoded["labels"])
    assert any(label != -100 for label in encoded["labels"])


def test_resolve_model_path_env(monkeypatch):
    monkeypatch.setenv("PROMPTFORGE_QUALITY_MODEL", "my/quality-model")
    assert resolve_model_path(None, kind="quality") == "my/quality-model"
    assert resolve_model_path("explicit", kind="quality") == "explicit"
