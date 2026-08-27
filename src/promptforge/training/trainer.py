from __future__ import annotations

from typing import Any

from transformers import Trainer


class PromptForgeTrainer(Trainer):
    """Trainer that feeds multi-label regression targets into the custom model."""

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[Any, Any, Any]:
        # Keep default behavior; logits key is picked up by Trainer.
        return super().prediction_step(
            model,
            inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
        )
