# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline video overlay pipeline for Cosmos-Reason2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from cosmos_reason2_utils.video_overlay.config import VideoOverlayConfig
from cosmos_reason2_utils.video_overlay.pipeline import run_infer, run_overlay, run_sample


def build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=defaults.get("config"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("sample", help="Sample video frames.")
    _add_config_arg(sample_parser, defaults)
    sample_parser.add_argument("--input", default=defaults.get("input"))
    sample_parser.add_argument(
        "--infer-fps", type=float, default=defaults.get("infer_fps", 2.0)
    )
    sample_parser.add_argument(
        "--sampled-dir", default=defaults.get("sampled_dir", "cosmos_reason2_utils/cosmos_reason2_utils/video_overlay/sampled_frames")
    )
    sample_parser.add_argument(
        "--metadata-out",
        default=defaults.get(
            "metadata_out", "cosmos_reason2_utils/cosmos_reason2_utils/video_overlay/results/sampled_metadata.json"
        ),
    )

    infer_parser = subparsers.add_parser("infer", help="Run inference on sampled frames.")
    _add_config_arg(infer_parser, defaults)
    infer_parser.add_argument(
        "--metadata",
        default=defaults.get("metadata", defaults.get("metadata_out")),
    )
    infer_parser.add_argument(
        "--results-out",
        default=defaults.get(
            "results_out", "cosmos_reason2_utils/cosmos_reason2_utils/video_overlay/results/inference_results.json"
        ),
    )
    infer_parser.add_argument(
        "--debug-dir", default=defaults.get("debug_dir", "cosmos_reason2_utils/cosmos_reason2_utils/video_overlay/debug_frames")
    )
    _add_prompt_args(infer_parser, defaults)
    _add_backend_args(infer_parser, defaults)
    _add_style_args(infer_parser, defaults)

    overlay_parser = subparsers.add_parser("overlay", help="Render the overlay video.")
    _add_config_arg(overlay_parser, defaults)
    overlay_parser.add_argument("--input", default=defaults.get("input"))
    overlay_parser.add_argument(
        "--results",
        default=defaults.get("results", defaults.get("results_out")),
    )
    overlay_parser.add_argument(
        "--segments-out",
        default=defaults.get(
            "segments_out", "cosmos_reason2_utils/cosmos_reason2_utils/video_overlay/results/overlay_segments.json"
        ),
    )
    overlay_parser.add_argument(
        "--output", default=defaults.get("output", "cosmos_reason2_utils/cosmos_reason2_utils/video_overlay/output/overlay.mp4")
    )
    _add_style_args(overlay_parser, defaults)

    all_parser = subparsers.add_parser("all", help="Run sample, infer, and overlay.")
    _add_config_arg(all_parser, defaults)
    all_parser.add_argument("--input", default=defaults.get("input"))
    all_parser.add_argument(
        "--infer-fps", type=float, default=defaults.get("infer_fps", 2.0)
    )
    all_parser.add_argument(
        "--sampled-dir", default=defaults.get("sampled_dir", "cosmos_reason2_utils/cosmos_reason2_utils/video_overlay/sampled_frames")
    )
    all_parser.add_argument(
        "--metadata-out",
        default=defaults.get(
            "metadata_out", "cosmos_reason2_utils/cosmos_reason2_utils/video_overlay/results/sampled_metadata.json"
        ),
    )
    all_parser.add_argument(
        "--results-out",
        default=defaults.get(
            "results_out", "cosmos_reason2_utils/cosmos_reason2_utils/video_overlay/results/inference_results.json"
        ),
    )
    all_parser.add_argument(
        "--segments-out",
        default=defaults.get(
            "segments_out", "cosmos_reason2_utils/cosmos_reason2_utils/video_overlay/results/overlay_segments.json"
        ),
    )
    all_parser.add_argument(
        "--output", default=defaults.get("output", "cosmos_reason2_utils/cosmos_reason2_utils/video_overlay/output/overlay.mp4")
    )
    all_parser.add_argument(
        "--debug-dir", default=defaults.get("debug_dir", "cosmos_reason2_utils/cosmos_reason2_utils/video_overlay/debug_frames")
    )
    _add_prompt_args(all_parser, defaults)
    _add_backend_args(all_parser, defaults)
    _add_style_args(all_parser, defaults)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the video overlay CLI."""
    argv = list(sys.argv[1:] if argv is None else argv)
    defaults = _load_defaults(argv)
    parser = build_parser(defaults)
    args = parser.parse_args(argv)
    merged = defaults | vars(args)
    config = VideoOverlayConfig.model_validate(
        {key: value for key, value in merged.items() if key in VideoOverlayConfig.model_fields}
    )
    overlay_style = config.overlay_style()

    if args.command == "sample":
        _require(args.input, "--input is required for sample.")
        manifest = run_sample(
            input_path=args.input,
            infer_fps=args.infer_fps,
            sampled_dir=args.sampled_dir,
            metadata_out=args.metadata_out,
        )
        print(
            f"Sampled {len(manifest.sampled_frames)} frames into {args.sampled_dir} "
            f"and wrote {args.metadata_out}"
        )
        return

    if args.command == "infer":
        _require(args.metadata, "--metadata is required for infer.")
        manifest = run_infer(
            metadata_path=args.metadata,
            config=config,
            results_out=args.results_out,
            debug_dir=args.debug_dir,
            overlay_style=overlay_style,
        )
        print(
            f"Wrote {len(manifest.results)} inference results to {args.results_out} "
            f"and debug frames to {args.debug_dir}"
        )
        return

    if args.command == "overlay":
        _require(args.input, "--input is required for overlay.")
        _require(args.results, "--results is required for overlay.")
        manifest = run_overlay(
            input_path=args.input,
            results_path=args.results,
            output_path=args.output,
            overlay_style=overlay_style,
            segments_out=args.segments_out,
        )
        print(
            f"Rendered {len(manifest.segments)} overlay segments to {args.output} "
            f"and wrote {args.segments_out}"
        )
        return

    if args.command == "all":
        _require(args.input, "--input is required for all.")
        run_sample(
            input_path=args.input,
            infer_fps=args.infer_fps,
            sampled_dir=args.sampled_dir,
            metadata_out=args.metadata_out,
        )
        run_infer(
            metadata_path=args.metadata_out,
            config=config,
            results_out=args.results_out,
            debug_dir=args.debug_dir,
            overlay_style=overlay_style,
        )
        run_overlay(
            input_path=args.input,
            results_path=args.results_out,
            output_path=args.output,
            overlay_style=overlay_style,
            segments_out=args.segments_out,
        )
        print(
            f"Wrote metadata {args.metadata_out}, results {args.results_out}, "
            f"segments {args.segments_out}, and overlay video {args.output}"
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


def _add_config_arg(parser: argparse.ArgumentParser, defaults: dict[str, Any]) -> None:
    parser.add_argument("--config", default=defaults.get("config"))


def _add_prompt_args(parser: argparse.ArgumentParser, defaults: dict[str, Any]) -> None:
    parser.add_argument("--prompt", default=defaults.get("prompt"))
    parser.add_argument("--prompt-file", default=defaults.get("prompt_file"))
    parser.add_argument(
        "--system-prompt",
        default=defaults.get("system_prompt", "You are a helpful assistant."),
    )


def _add_backend_args(parser: argparse.ArgumentParser, defaults: dict[str, Any]) -> None:
    parser.add_argument(
        "--backend",
        choices=["mock", "online", "offline"],
        default=defaults.get("backend", "mock"),
    )
    parser.add_argument(
        "--model", default=defaults.get("model", "nvidia/Cosmos-Reason2-2B")
    )
    parser.add_argument("--revision", default=defaults.get("revision"))
    parser.add_argument("--host", default=defaults.get("host", "localhost"))
    parser.add_argument("--port", type=int, default=defaults.get("port", 8000))
    parser.add_argument(
        "--max-model-len", type=int, default=defaults.get("max_model_len", 16384)
    )
    parser.add_argument("--min-pixels", type=int, default=defaults.get("min_pixels"))
    parser.add_argument("--total-pixels", type=int, default=defaults.get("total_pixels"))
    parser.add_argument("--max-tokens", type=int, default=defaults.get("max_tokens", 1024))
    parser.add_argument("--temperature", type=float, default=defaults.get("temperature"))
    parser.add_argument("--top-p", type=float, default=defaults.get("top_p"))
    parser.add_argument("--top-k", type=int, default=defaults.get("top_k"))
    parser.add_argument(
        "--presence-penalty", type=float, default=defaults.get("presence_penalty")
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=defaults.get("repetition_penalty"),
    )
    parser.add_argument("--seed", type=int, default=defaults.get("seed"))


def _add_style_args(parser: argparse.ArgumentParser, defaults: dict[str, Any]) -> None:
    parser.add_argument("--font-scale", type=float, default=defaults.get("font_scale", 0.7))
    parser.add_argument("--line-spacing", type=int, default=defaults.get("line_spacing", 28))
    parser.add_argument("--margin-x", type=int, default=defaults.get("margin_x", 20))
    parser.add_argument("--margin-y", type=int, default=defaults.get("margin_y", 30))
    parser.add_argument(
        "--max-width-ratio", type=float, default=defaults.get("max_width_ratio", 0.45)
    )
    parser.add_argument("--box-alpha", type=float, default=defaults.get("box_alpha", 0.65))
    parser.add_argument("--thickness", type=int, default=defaults.get("thickness", 1))


def _extract_config_path(argv: list[str]) -> str | None:
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return None


def _load_defaults(argv: list[str]) -> dict[str, Any]:
    config_path = _extract_config_path(argv)
    if config_path is None:
        return {}
    config = VideoOverlayConfig.from_yaml(config_path).model_dump()
    config["config"] = config_path
    return config


def _require(value: Any, message: str) -> None:
    if value in (None, ""):
        raise SystemExit(message)


if __name__ == "__main__":
    main()
