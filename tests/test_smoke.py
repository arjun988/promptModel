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
