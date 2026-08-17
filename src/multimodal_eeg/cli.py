"""Command-line surface for preparing, running, and publishing the NICE study."""

from __future__ import annotations

import argparse
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="multimodal-eeg")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download-data", help="Download the BraVL feature archive")
    download.add_argument("--destination", type=Path, default=Path("data"))
    templates = commands.add_parser("prepare-nice-templates", help="Build non-EEG templates")
    templates.add_argument("--images-root", type=Path, required=True)
    templates.add_argument("--eeg-metadata", type=Path, required=True)
    templates.add_argument("--eeg-manifest", type=Path, required=True)
    templates.add_argument("--eeg-images-root", type=Path, required=True)
    templates.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run-nice-averaged-comparison", help="Run the NICE comparison")
    run.add_argument("--dataset", type=Path, default=Path("data/ThingsEEG-Text"))
    run.add_argument("--alignment-manifest", type=Path, required=True)
    run.add_argument("--dinov2-probe", type=Path, required=True)
    run.add_argument("--template-features", type=Path, required=True)
    run.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/nice-averaged-study.json",
    )
    run.add_argument("--output", type=Path, default=Path("results/nice-averaged-comparison"))
    run.add_argument("--bundles", type=Path, default=Path("artifacts/nice-bundles"))
    publish = commands.add_parser("publish-nice-study", help="Regenerate public outputs")
    publication_inputs = (
        "comparison",
        "template-manifest",
        "template-report",
        "template-audit",
        "alignment-manifest",
    )
    for name in publication_inputs:
        publish.add_argument(f"--{name}", type=Path, required=True)
    publish.add_argument("--output", type=Path, default=Path("docs/results/nice"))
    publish.add_argument("--figures", type=Path, default=Path("figures/nice"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "download-data":
        from .download import download_feature_archive
        download_feature_archive(args.destination)
    elif args.command == "prepare-nice-templates":
        from .nice_templates import build_template_manifest
        build_template_manifest(
            args.images_root,
            args.eeg_metadata,
            args.eeg_manifest,
            args.eeg_images_root,
            args.output,
        )
    elif args.command == "run-nice-averaged-comparison":
        from .aligned_comparison import run_aligned_comparison
        run_aligned_comparison(
            args.dataset,
            args.alignment_manifest,
            args.dinov2_probe,
            args.config,
            args.output,
            args.bundles,
            args.template_features,
        )
    else:
        from .nice_publication import publish_nice_study
        publish_nice_study(
            args.comparison,
            args.template_manifest,
            args.template_report,
            args.template_audit,
            args.alignment_manifest,
            args.output,
            args.figures,
        )


if __name__ == "__main__":
    main()
