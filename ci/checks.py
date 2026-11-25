"""Reviewer-style static checks run in CI (offline, no LLM). Exit 1 on any violation.

    python -m ci.checks [--branch track/ci/foo --base origin/main]

Covers: gate_config.json == CONTRACT.md section 5, secrets, tracked .env, forbidden Gemini
imports outside src/llm.py, and (when --branch is given) track ownership of changed files.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OWNERS: dict[str, tuple[str, ...]] = {
    "ingestion": ("src/ingestion/", "data/", "tests/ingestion/"),
    "agent": ("src/agent/", "tests/agent/"),
    "eval": ("eval/", "tests/eval/"),
    "ci": ("ci/", ".github/", "tests/ci/"),
}
SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    re.compile(r"GEMINI_API_KEY\s*=\s*['\"]?[A-Za-z0-9_\-]{10,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"api[_-]?key\s*[:=]\s*['\"][^'\"\s]{8,}['\"]", re.I),
]
FORBIDDEN_IMPORT = re.compile(
    r"^\s*(?:from\s+(?:langchain_google_genai|google\.generativeai|google\.genai)\b"
    r"|from\s+google\s+import\s+(?:generativeai|genai)\b"
    r"|import\s+(?:langchain_google_genai|google\.generativeai|google\.genai)\b)",
    re.M,
)
GEMINI_ENDPOINT = "generativelanguage.googleapis.com"
SCAN_EXEMPT = ("ci/checks.py", "tests/ci/")  # they contain the patterns as data
TEXT_SUFFIXES = {".py", ".json", ".jsonl", ".md", ".toml", ".yml", ".yaml", ".txt", ".cfg", ".env"}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def tracked_files() -> list[str]:
    return [p for p in git("ls-files").splitlines() if p]


def contract_thresholds(contract_text: str) -> dict[str, float]:
    """Parse the section 5 rule table into the gate_config.json key space."""
    sec = contract_text.split("## 5.", 1)[-1].split("\n## ", 1)[0]
    out: dict[str, float] = {}
    for line in sec.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 2:
            continue
        rule, value = cells
        m = re.search(r"<=\s*\+?([0-9.]+)(%?)", value)
        if not m:
            continue
        num = float(m.group(1)) / (100 if m.group(2) else 1)
        if rule.startswith("hallucination_rate"):
            out["hallucination_rate_max"] = num
        elif rule.startswith("false_refusals"):
            out["false_refusal_rate_max"] = num
        elif "p95" in rule:
            out["p95_latency_regression_max"] = num
        elif "p50" in rule:
            out["p50_latency_regression_max"] = num
    return out


def check_gate_config(contract_text: str, config: dict) -> list[str]:
    want = contract_thresholds(contract_text)
    if len(want) != 4:
        return [f"could not parse 4 thresholds from CONTRACT.md section 5 (got {sorted(want)})"]
    errs = []
    for key, val in want.items():
        got = config.get(key)
        if got is None or abs(got - val) > 1e-9:
            errs.append(f"ci/gate_config.json {key}={got!r} but CONTRACT.md section 5 says {val}")
    extra = set(config) - set(want)
    if extra:
        errs.append(f"ci/gate_config.json has keys not in the contract: {sorted(extra)}")
    return errs


def scan_secrets(files: dict[str, str]) -> list[str]:
    errs = []
    for path, text in files.items():
        for pat in SECRET_PATTERNS:
            for m in pat.finditer(text):
                errs.append(f"{path}: possible secret {m.group(0)[:4]}**** ({pat.pattern[:20]}...)")
    return errs


def scan_imports(files: dict[str, str]) -> list[str]:
    errs = []
    for p, t in files.items():
        if not p.endswith(".py") or p == "src/llm.py" or p.startswith(SCAN_EXEMPT):
            continue
        if FORBIDDEN_IMPORT.search(t):
            errs.append(f"{p}: imports a Gemini SDK; only src/llm.py may (CONTRACT.md section 6)")
        if GEMINI_ENDPOINT in t:
            errs.append(f"{p}: calls a Gemini endpoint directly; only src/llm.py may")
    return errs


def env_tracked(paths: list[str]) -> list[str]:
    return [
        f"{p}: env file is tracked by git"
        for p in paths
        if re.search(r"(^|/)\.env", p) and not p.endswith(".env.example")
    ]


def ownership_violations(branch: str, changed: list[str]) -> list[str]:
    m = re.fullmatch(r"track/([a-z]+)/[^/]+(?:/.*)?", branch)
    if not m:
        return [f"branch {branch!r} does not match track/<name>/<topic>"]
    track = m.group(1)
    if track == "shared":
        return []  # foundation / contract-change branches are governed by the reviewer agent
    if track not in OWNERS:
        return [f"unknown track {track!r} in branch name"]
    return [
        f"{p}: outside the {track} track's folders {OWNERS[track]}"
        for p in changed
        if not p.startswith(OWNERS[track])
    ]


def read_text_files(paths: list[str]) -> dict[str, str]:
    out = {}
    for p in paths:
        f = ROOT / p
        if f.suffix in TEXT_SUFFIXES or f.name.startswith(".env"):
            try:
                out[p] = f.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--branch", help="PR head branch; enables the ownership check")
    ap.add_argument("--base", default="origin/main", help="ref to diff the branch against")
    args = ap.parse_args(argv)

    errs: list[str] = []
    contract = (ROOT / "CONTRACT.md").read_text(encoding="utf-8")
    config = json.loads((ROOT / "ci" / "gate_config.json").read_text(encoding="utf-8"))
    errs += check_gate_config(contract, config)

    paths = tracked_files()
    files = read_text_files(paths)
    errs += scan_secrets(files)
    errs += scan_imports(files)
    errs += env_tracked(paths)
    if ".env" not in (ROOT / ".gitignore").read_text(encoding="utf-8").split():
        errs.append(".gitignore does not ignore .env")
    if args.branch:
        changed = git("diff", "--name-only", f"{args.base}...HEAD").splitlines()
        errs += ownership_violations(args.branch, changed)

    for e in errs:
        print(f"VIOLATION: {e}")
    print("checks: FAIL" if errs else "checks: ok")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
