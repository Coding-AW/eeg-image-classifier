"""Fail closed when the tracked release contains private, oversized, or stale content."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_BYTES = 50 * 1024 * 1024
FORBIDDEN_SUFFIXES = {".joblib", ".mat", ".npy", ".npz", ".pt", ".pth", ".safetensors"}
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret)\s*[:=]\s*['\"][^'\"]{8,}"),
    re.compile(r"\b(?:sk|ghp|github_pat)_[A-Za-z0-9_-]{20,}\b"),
)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)


def _tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        text=False,
    ).split(b"\0")
    return [ROOT / item.decode("utf-8") for item in output if item]


def check_release() -> list[str]:
    failures = []
    password_file = ROOT / "data/external/things-full/password_images.txt"
    secret = None
    if password_file.is_file():
        line = password_file.read_text(encoding="utf-8").strip().splitlines()[-1]
        if ":" in line:
            secret = line.rsplit(":", 1)[1].strip()
    actual_paths = (
        str(ROOT).casefold(),
        str(Path.home()).casefold(),
    )
    for path in _tracked_files():
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if path.stat().st_size > MAX_TRACKED_BYTES:
            failures.append(f"oversized tracked file: {relative}")
        if path.suffix.casefold() in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden tracked binary: {relative}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        folded = text.casefold()
        if any(value in folded for value in actual_paths):
            failures.append(f"local absolute path in tracked file: {relative}")
        if secret and len(secret) >= 4 and secret in text:
            failures.append(f"dataset password in tracked file: {relative}")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                failures.append(f"possible credential in tracked file: {relative}")
        if EMAIL.search(text) and relative not in {"CITATION.cff", "docs/references.bib"}:
            failures.append(f"unexpected email address in tracked file: {relative}")
        if path.suffix.casefold() == ".md":
            for match in MARKDOWN_LINK.finditer(text):
                target = match.group(1).split("#", 1)[0]
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                resolved = (path.parent / target).resolve()
                if not resolved.exists():
                    failures.append(f"broken Markdown link in {relative}: {target}")
    return failures


if __name__ == "__main__":
    problems = check_release()
    if problems:
        raise SystemExit("\n".join(problems))
    print("Tracked release safety checks passed.")
