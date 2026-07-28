"""Minimal notebook builder.

Notebooks are generated artifacts here. Editing eight .ipynb JSON files by hand
means eight chances to break the bootstrap cell; generating them from one place
means a fix lands once. Run tools/build_notebooks.py after any change.

    md("## Heading")   -> markdown cell
    code("x = 1")      -> code cell
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


def _lines(text: str) -> list[str]:
    """Split into the line list nbformat expects (newline kept on all but last)."""
    text = dedent(text).strip("\n")
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]] if parts else []


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def notebook(cells: list[dict], title: str = "") -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True},
            "title": title,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(path: Path, cells: list[dict], title: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(notebook(cells, title), f, indent=1, ensure_ascii=False)
        f.write("\n")
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"  {path.name:<34} {len(cells):>3} cells ({n_code} code)")
