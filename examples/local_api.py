"""Minimal local API usage after Phase-4 install."""

from __future__ import annotations

import json

from promptforge import PromptForge


def main() -> None:
    # Resolves models from env / ~/.promptforge / ./outputs
    pf = PromptForge.from_config()

    analysis = pf.analyze("Build me a website")
    print("ANALYZE")
    print(json.dumps(analysis, indent=2))

    if pf.optimizer is not None and pf.scorer is not None:
        result = pf.run("Build me a website")
        print("\nRUN")
        print(json.dumps(result, indent=2))
    elif pf.optimizer is not None:
        print("\nOPTIMIZE")
        print(json.dumps(pf.optimize("Build me a website"), indent=2))


if __name__ == "__main__":
    main()
