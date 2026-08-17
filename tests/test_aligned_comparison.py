import csv
from pathlib import Path

from multimodal_eeg.aligned_comparison import LOCKED_SEEDS, _nice_literature_ledger
from multimodal_eeg.cli import REPOSITORY_ROOT, _parser


def test_nice_is_direct_and_atm_fails_closed_to_context() -> None:
    ledger = {row["study"]: row for row in _nice_literature_ledger()}
    assert ledger["Current NICE-style five-run study"]["eligibility"] == "direct"
    assert ledger["NICE-GA headline (Song et al., ICLR 2024)"]["eligibility"] == "direct"
    assert ledger["ATM (Li et al., NeurIPS 2024)"]["eligibility"] == "context-only"


def test_locked_seeds_are_predeclared() -> None:
    assert LOCKED_SEEDS == (17, 29, 43, 71, 101)


def test_csv_module_is_available_for_ledger_export() -> None:
    assert csv.DictWriter is not None


def test_nice_run_requires_explicit_alignment_and_templates() -> None:
    values = _parser().parse_args(
        [
            "run-nice-averaged-comparison",
            "--alignment-manifest",
            "artifacts/stimulus-manifest.csv",
            "--dinov2-probe",
            "artifacts/dinov2-probe",
            "--template-features",
            "artifacts/nice-concept-templates/features",
        ]
    )
    assert values.config == REPOSITORY_ROOT / "configs/nice-averaged-study.json"
    assert values.template_features == Path("artifacts/nice-concept-templates/features")


def test_removed_commands_are_not_public() -> None:
    choices = _parser()._subparsers._group_actions[0].choices
    assert "run" + "-study" not in choices
    assert "publish" + "-study" not in choices
    assert "audit" + "-study" not in choices
    assert "inspect-dinov2-probe" not in choices
