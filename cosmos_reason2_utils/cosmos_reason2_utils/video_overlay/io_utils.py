# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""I/O helpers for the video overlay pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel
import yaml

T = TypeVar("T", bound=BaseModel)


def ensure_parent_dir(path: str | Path) -> None:
    """Create the parent directory for a file path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(model: BaseModel, path: str | Path) -> None:
    """Write a Pydantic model as formatted JSON."""
    ensure_parent_dir(path)
    Path(path).write_text(model.model_dump_json(indent=2), encoding="utf-8")


def read_json(path: str | Path, model_cls: type[T]) -> T:
    """Read a Pydantic model from JSON."""
    return model_cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_prompt(prompt: str | None, prompt_file: str | Path | None) -> str:
    """Load prompt text from either inline content or a file."""
    if prompt_file:
        path = Path(prompt_file)
        if path.suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(data, dict):
                user_prompt = data.get("user_prompt")
                if isinstance(user_prompt, str) and user_prompt.strip():
                    return user_prompt.strip()
            raise ValueError(f"Prompt YAML must contain a non-empty user_prompt: {path}")
        return path.read_text(encoding="utf-8").strip()
    if prompt:
        return prompt.strip()
    raise ValueError("A prompt or prompt file is required.")
