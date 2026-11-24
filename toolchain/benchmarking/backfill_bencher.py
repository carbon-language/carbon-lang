"""Temporary backfill helpers for the `action-test` workflow."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(_REPO_ROOT))
from toolchain.benchmarking.bencher_suites import (  # noqa: E402
    TRUNK_THRESHOLDS,
    Tier,
)
from toolchain.benchmarking.run_bencher import (  # noqa: E402
    BmfValidationError,
    convert_suite_bmf_to_v1,
    format_threshold_args,
    merge_bmf_payloads,
    sync_measure_units,
    validate_bmf,
)

REPO = "carbon-language/carbon-lang"
PROJECT = "carbon-lang"
GIT_REF = "refs/remotes/origin/trunk"
SINCE_REF = "v0.0.0-0.nightly.2026.11.01"
NIGHTLY_TAG_PREFIX = "v0.0.0-0.nightly."

RUNNERS = ("ubuntu-22.04", "ubuntu-22.04-arm")
TRUSTED_BRANCHES = frozenset({"trunk", "action-test"})
_ARTIFACT_NAME_RE = re.compile(
    r"^bencher-bmf-(continuous|nightly)-([A-Za-z0-9._-]+)-([0-9a-f]{40})$"
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _gh_api_json_lines(
    endpoint: str, jq_expr: str, paginate: bool = True
) -> list[dict[str, Any]]:
    cmd = ["gh", "api", *(["--paginate"] if paginate else []), endpoint]
    out = subprocess.run(
        [*cmd, "--jq", jq_expr], capture_output=True, text=True, check=True
    ).stdout
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def _load_commits(git_ref: str) -> dict[str, tuple[int, int]]:
    since_sha = _git("rev-parse", "--verify", f"{SINCE_REF}^{{commit}}").strip()
    out = _git(
        "log",
        "--first-parent",
        "--reverse",
        "--format=%H:%ct",
        f"{SINCE_REF}^..{git_ref}",
    )
    commits = {
        sha: (idx, int(epoch))
        for idx, line in enumerate(out.splitlines())
        for sha, epoch in [line.strip().split(":", 1)]
    }
    if since_sha not in commits:
        raise BmfValidationError(
            f"{SINCE_REF} ({since_sha[:12]}) not in {git_ref} first-parent."
        )
    return commits


def _load_code_shas(git_ref: str) -> list[str]:
    out = _git(
        "log",
        "--first-parent",
        "--reverse",
        "--format=%H",
        f"{SINCE_REF}^..{git_ref}",
        "--",
        ".",
        ":(exclude)*.md",
        ":(top,exclude)LICENSE",
        ":(top,exclude)CODEOWNERS",
        ":(top,exclude).gitignore",
        ":(top,exclude).gitattributes",
        ":(top,exclude).gitmodules",
        ":(top,exclude).git-blame-ignore-revs",
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def _load_nightly_tags(
    commits: dict[str, tuple[int, int]],
) -> list[tuple[str, str]]:
    out = _git(
        "for-each-ref",
        "--sort=refname",
        "--format=%(refname:short) %(objectname) %(*objectname)",
        f"refs/tags/{NIGHTLY_TAG_PREFIX}*",
    )
    tags: list[tuple[str, str]] = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            sha = parts[2] if len(parts) >= 3 else parts[1]
            if sha in commits:
                tags.append((parts[0], sha))
    return tags


def _list_bmf_artifacts() -> dict[tuple[Tier, str, str], int]:
    items = _gh_api_json_lines(
        f"repos/{REPO}/actions/artifacts?per_page=100",
        ".artifacts[] | {id, name, expired, "
        "head_branch: .workflow_run.head_branch, "
        "repo_id: .workflow_run.repository_id, "
        "head_repo_id: .workflow_run.head_repository_id}",
    )
    selected: dict[tuple[Tier, str, str], int] = {}
    for art in items:
        repo_id = art.get("repo_id")
        if (
            art.get("expired")
            or repo_id is None
            or repo_id != art.get("head_repo_id")
            or art.get("head_branch") not in TRUSTED_BRANCHES
        ):
            continue
        m = _ARTIFACT_NAME_RE.fullmatch(str(art.get("name", "")))
        if m is None:
            continue
        tier, runner, sha = Tier(m.group(1)), m.group(2), m.group(3)
        if runner not in RUNNERS:
            continue
        key = (tier, sha, runner)
        selected[key] = max(selected.get(key, -1), int(art["id"]))
    return selected


def _normalize_to_bmf_v1(data: Any) -> dict[str, list[dict[str, Any]]]:
    """Converts a legacy BMF v0 payload to BMF v1, or validates BMF v1."""
    if not isinstance(data, dict) or not data:
        raise BmfValidationError("BMF payload must be a non-empty JSON object.")

    if all(isinstance(v, list) for v in data.values()):
        validate_bmf(data)
        return data

    if not all(isinstance(v, dict) for v in data.values()):
        raise BmfValidationError("Mixed or invalid BMF payload structure.")

    by_suite: dict[str, dict[str, Any]] = {}
    for full_name, measures in data.items():
        if not isinstance(full_name, str) or "::" not in full_name:
            raise BmfValidationError(
                f"Expected `<suite>::<benchmark>` in BMF v0 key: {full_name!r}"
            )
        target_name, raw_name = full_name.split("::", 1)
        by_suite.setdefault(target_name, {})[raw_name] = measures

    payloads = [
        convert_suite_bmf_to_v1(target_name, suite_bmf)
        for target_name, suite_bmf in by_suite.items()
    ]
    merged, _ = merge_bmf_payloads(payloads)
    return merged


def _download_bmf_artifact(artifact_id: int) -> bytes:
    zip_bytes = subprocess.run(
        ["gh", "api", f"repos/{REPO}/actions/artifacts/{artifact_id}/zip"],
        capture_output=True,
        check=True,
    ).stdout
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            infos = zf.infolist()
            if len(infos) != 1:
                raise BmfValidationError("Expected 1 BMF file in artifact zip.")
            bmf_bytes = zf.read(infos[0].filename)
        normalized = _normalize_to_bmf_v1(json.loads(bmf_bytes.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError, zipfile.BadZipFile) as e:
        raise BmfValidationError(f"Invalid BMF artifact zip: {e}") from e
    return (json.dumps(normalized, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _check_no_active_benchmark_runs() -> None:
    runs = _gh_api_json_lines(
        f"repos/{REPO}/actions/workflows/benchmarks.yaml/runs?per_page=20",
        ".workflow_runs[] | {id, status}",
        paginate=False,
    )
    active = [r["id"] for r in runs if r.get("status") != "completed"]
    if active:
        raise RuntimeError(f"Live `benchmarks.yaml` run(s) active: {active}")


def _run_bencher_with_retry(cmd: list[str]) -> None:
    for attempt in range(1, 4):
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.stdout:
            sys.stdout.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        if proc.returncode == 0:
            return
        output = f"{proc.stdout}\n{proc.stderr}".lower()
        if ("429" in output or "too many requests" in output) and attempt < 3:
            print(
                f"Bencher HTTP 429 (attempt {attempt}/3); sleeping 30s...",
                file=sys.stderr,
            )
            time.sleep(30)
            continue
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, proc.stdout, proc.stderr
        )


def plan_backfill(git_ref: str, github_output: Optional[Path]) -> None:
    commits = _load_commits(git_ref)
    nightly_shas = {sha for _, sha in _load_nightly_tags(commits)}
    code_shas = set(_load_code_shas(git_ref))
    target_shas = [
        sha for sha in commits if sha in nightly_shas or sha in code_shas
    ]

    existing = _list_bmf_artifacts()
    matrix: list[dict[str, Any]] = []
    skipped = 0
    for sha in target_shas:
        for runner in RUNNERS:
            need_nightly = (
                sha in nightly_shas
                and (Tier.NIGHTLY, sha, runner) not in existing
            )
            need_continuous = (
                sha in code_shas
                and (Tier.CONTINUOUS, sha, runner) not in existing
            )
            if need_nightly or need_continuous:
                matrix.append(
                    {
                        "sha": sha,
                        "short_sha": sha[:12],
                        "runner": runner,
                        "need_nightly": need_nightly,
                        "need_continuous": need_continuous,
                    }
                )
            else:
                skipped += 1

    print(
        f"plan: {len(target_shas)} commits ({len(nightly_shas)} nightly, "
        f"{len(code_shas)} continuous), {skipped} pairs skipped, "
        f"{len(matrix)} jobs",
        file=sys.stderr,
    )
    if github_output is not None:
        matrix_json = json.dumps({"include": matrix}, separators=(",", ":"))
        with github_output.open("a", encoding="utf-8") as f:
            f.write(f"has_work={'true' if matrix else 'false'}\n")
            f.write(f"matrix={matrix_json}\n")


def attach_nightly_releases(
    git_ref: str, dry_run: bool, replace: bool = False
) -> None:
    if not dry_run:
        _check_no_active_benchmark_runs()
    commits = _load_commits(git_ref)
    releases = _gh_api_json_lines(
        f"repos/{REPO}/releases?per_page=100",
        ".[] | select(.draft | not) | {tag_name, assets: [.assets[].name]}",
        paginate=True,
    )
    assets_by_tag = {str(r["tag_name"]): list(r["assets"]) for r in releases}
    existing = _list_bmf_artifacts()

    plan: list[tuple[str, str, list[str]]] = []
    missing: list[str] = []
    for tag, sha in _load_nightly_tags(commits):
        if tag not in assets_by_tag:
            print(f"SKIP {tag}: no published Release.", file=sys.stderr)
            continue
        if not replace and any(
            a.startswith("benchmarks-bmf-") and a.endswith(".tar.gz")
            for a in assets_by_tag[tag]
        ):
            print(f"SKIP {tag}: already has BMF archive.", file=sys.stderr)
            continue
        missing.extend(
            f"{tag} ({sha[:12]}, {r})"
            for r in RUNNERS
            if (Tier.NIGHTLY, sha, r) not in existing
        )
        plan.append((tag, sha, list(RUNNERS)))

    print(
        f"attach-releases (dry_run={dry_run}, replace={replace}): "
        f"{len(plan)} release(s)",
        file=sys.stderr,
    )
    if missing:
        raise BmfValidationError(
            f"Missing {len(missing)} nightly artifact(s): {', '.join(missing)}"
        )
    if dry_run or not plan:
        return

    # Prefetch, normalize to BMF v1, and validate every unique `(sha, runner)`
    # artifact before modifying any published release asset.
    payloads: dict[tuple[str, str], bytes] = {}
    for _, sha, runners in plan:
        for runner in runners:
            key = (sha, runner)
            if key not in payloads:
                payloads[key] = _download_bmf_artifact(
                    existing[(Tier.NIGHTLY, sha, runner)]
                )
    print(
        f"Prefetched and validated {len(payloads)} BMF v1 nightly artifacts.",
        file=sys.stderr,
    )

    _check_no_active_benchmark_runs()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        for tag, sha, runners in plan:
            archive_path = (
                tmp_root / f"benchmarks-bmf-{tag.removeprefix('v')}.tar.gz"
            )
            with tarfile.open(archive_path, "w:gz") as tf:
                for runner in runners:
                    payload = payloads[(sha, runner)]
                    info = tarfile.TarInfo(name=f"bmf-{runner}.json")
                    info.size = len(payload)
                    tf.addfile(info, io.BytesIO(payload))
            cmd = [
                "gh",
                "release",
                "upload",
                tag,
                str(archive_path),
                "--repo",
                REPO,
            ]
            if replace:
                cmd.append("--clobber")
            subprocess.run(cmd, check=True)


def replay_artifacts(
    tier: Tier, branch: str, git_ref: str, dry_run: bool
) -> None:
    commits = _load_commits(git_ref)
    if not dry_run and branch in {"trunk", "nightly"}:
        _check_no_active_benchmark_runs()

    existing = _list_bmf_artifacts()
    if tier == Tier.NIGHTLY:
        expected_shas = list(
            dict.fromkeys(sha for _, sha in _load_nightly_tags(commits))
        )
    else:
        expected_shas = _load_code_shas(git_ref)

    missing = [
        f"{sha[:12]}:{r}"
        for sha in expected_shas
        for r in RUNNERS
        if (tier, sha, r) not in existing
    ]

    expected_set = set(expected_shas)
    plan = [
        (sha, runners)
        for sha in commits
        if sha in expected_set
        and (runners := [r for r in RUNNERS if (tier, sha, r) in existing])
    ]
    print(
        f"replay tier={tier.value} branch={branch} dry_run={dry_run}: "
        f"{len(plan)}/{len(expected_shas)} commits present",
        file=sys.stderr,
    )
    for sha, runners in plan:
        print(f"  {sha[:12]} {runners}", file=sys.stderr)
    if missing:
        raise BmfValidationError(
            f"Missing {len(missing)} required {tier.value} artifact(s): "
            + ", ".join(missing)
        )
    if not plan:
        raise BmfValidationError(f"No {tier.value} artifacts found to replay.")
    if dry_run:
        return

    reset_key = str(uuid.uuid4())
    # Pass `TRUNK_THRESHOLDS` even on `action-test` so smoke tests verify the
    # BMF v1 threshold flags end-to-end against Bencher before live replay.
    threshold_args = format_threshold_args(TRUNK_THRESHOLDS)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        # Prefetch, normalize to BMF v1, and validate every payload before
        # resetting the branch start point on Bencher.
        prepared: list[tuple[str, str, str, Path]] = []
        for sha, runners in plan:
            epoch = str(commits[sha][1])
            for runner in runners:
                bmf_path = tmp_root / f"{sha}-{runner}.json"
                bmf_path.write_bytes(
                    _download_bmf_artifact(existing[(tier, sha, runner)])
                )
                prepared.append((sha, epoch, runner, bmf_path))
        print(
            f"Prefetched and validated {len(prepared)} BMF v1 payloads.",
            file=sys.stderr,
        )

        if branch in {"trunk", "nightly"}:
            _check_no_active_benchmark_runs()

        for call_idx, (sha, epoch, runner, bmf_path) in enumerate(prepared):
            cmd = [
                "bencher",
                "run",
                "--project",
                PROJECT,
                "--branch",
                branch,
                "--hash",
                sha,
                "--testbed",
                runner,
                "--backdate",
                epoch,
                "--adapter",
                "json",
                "--file",
                str(bmf_path),
                "--thresholds-reset",
                *threshold_args,
            ]
            if call_idx == 0:
                cmd.extend(
                    [
                        "--start-point-reset",
                        "--start-point-idempotency-key",
                        reset_key,
                    ]
                )
            if call_idx > 0:
                time.sleep(1.1)
            _run_bencher_with_retry(cmd)

    sync_measure_units(project=PROJECT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-ref", default=GIT_REF)
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--github-output", type=Path)

    p_attach = sub.add_parser("attach-releases")
    p_attach.add_argument("--dry-run", action="store_true")
    p_attach.add_argument("--replace", action="store_true")

    p_replay = sub.add_parser("replay")
    p_replay.add_argument(
        "--tier", choices=("nightly", "continuous"), required=True
    )
    p_replay.add_argument("--branch", required=True)
    p_replay.add_argument("--dry-run", action="store_true")

    ns = parser.parse_args()
    if ns.subcommand == "plan":
        plan_backfill(ns.git_ref, ns.github_output)
    elif ns.subcommand == "attach-releases":
        attach_nightly_releases(ns.git_ref, ns.dry_run, replace=ns.replace)
    elif ns.subcommand == "replay":
        replay_artifacts(Tier(ns.tier), ns.branch, ns.git_ref, ns.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
