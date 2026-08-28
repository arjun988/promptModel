from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = (
    "You are PromptForge Optimizer. Rewrite the user's prompt into a clear, "
    "specific, complete, and actionable LLM prompt. Preserve the original intent "
    "and topic. Add missing audience, context, constraints, and output format when "
    "needed. Return ONLY the optimized prompt text — no commentary."
)


def format_optimizer_input(
    prompt: str,
    analysis: dict[str, Any],
    task_type: str = "general",
) -> str:
    dims = analysis.get("dimensions", analysis)
    lines = [
        f"Task type: {task_type}",
        "Original prompt:",
        prompt.strip(),
        "",
        "Quality analysis:",
        f"- quality_score: {analysis.get('quality_score', dims.get('quality_score', 'n/a'))}",
    ]
    for key in (
        "clarity",
        "specificity",
        "context",
        "goal_definition",
        "constraints",
        "completeness",
        "actionability",
    ):
        if key in dims:
            lines.append(f"- {key}: {dims[key]}")

    issues = analysis.get("issues") or []
    missing = analysis.get("missing_information") or analysis.get("missing") or []
    if issues:
        lines.append(f"- issues: {', '.join(issues)}")
    if missing:
        lines.append(f"- missing_information: {', '.join(missing)}")

    lines.extend(
        [
            "",
            "Rewrite this into a high-quality optimized prompt.",
            "Keep the same goal and topic as the original.",
            "Return only the optimized prompt.",
        ]
    )
    return "\n".join(lines)


def build_optimizer_messages(
    prompt: str,
    analysis: dict[str, Any],
    optimized_prompt: str | None = None,
    task_type: str = "general",
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": format_optimizer_input(prompt, analysis, task_type=task_type),
        },
    ]
    if optimized_prompt is not None:
        messages.append({"role": "assistant", "content": optimized_prompt.strip()})
    return messages


def apply_chat_prompt(
    tokenizer,
    prompt: str,
    analysis: dict[str, Any],
    task_type: str = "general",
) -> str:
    """Prompt text for generation (system + user + assistant header)."""
    messages = build_optimizer_messages(prompt, analysis, task_type=task_type)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def tokenize_sft_messages(
    tokenizer,
    messages: list[dict[str, str]],
    max_length: int,
) -> dict[str, list[int]]:
    """
    Tokenize a chat example and mask loss on system + user tokens.
    Only assistant completion tokens are trained.
    """
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("SFT example must end with an assistant message")

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )

    full = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    prompt_len = min(len(prompt_ids), len(full["input_ids"]))

    labels = list(full["input_ids"])
    for i in range(prompt_len):
        labels[i] = -100

    full["labels"] = labels
    return full


def generation_stop_ids(tokenizer) -> list[int]:
    """Extra token ids that should stop generation (Qwen chat markers)."""
    stops: list[int] = []
    for token in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is not None and tid != tokenizer.unk_token_id:
            stops.append(tid)
    if tokenizer.eos_token_id is not None:
        stops.append(tokenizer.eos_token_id)
    return list(dict.fromkeys(stops))
