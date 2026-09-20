#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.12"
# ///


"""Checks diagnostic use.

Validates that each diagnostic declared with CARBON_DIAGNOSTIC_KIND is
referenced by one (and only one) CARBON_DIAGNOSTIC.

Labels, contexts, and location info have no kind to register, so this also
stands in for the registry. `label` below stands for any of the three:

- Each label is attached somewhere. The compiler warns about most of these,
  but not about one declared in a header.

- Each label is exercised by a file_test, which is what `coverage_test` does
  for kinds. It can't do it for labels, because there is no list of them to
  enumerate from C++; this reads the declarations and the testdata directly.

- Each name a testdata file matches on is a kind or a label that exists, so a
  renamed one doesn't leave a test matching nothing.

Being attached does not imply being covered, which is why both are checked: a
label attached only on a branch no test takes is referenced by the code and
drawn by nothing.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import collections
import itertools
import os
import re
import sys
from concurrent import futures
from pathlib import Path
from typing import Dict, List, NamedTuple, Set, override

# Test diagnostics, ignored because they're expected to not pass.
IGNORED = set(["TestDiagnostic"])

# The declarations that name something other than a diagnostic kind. A label
# marks a range of source; a context names the operation a problem happened
# inside; location info is a step in the path a location was reached by. None
# of them is registered, and each carries a name that output and testdata can
# match on, so all three are checked here.
LABEL_DECL_RE = (
    r"CARBON_DIAGNOSTIC_(?:LABEL|SOFT_CONTEXT|CONTEXT|LOCATION_INFO)"
    r"\(\s*(\w+),"
)

# Labels that no file_test exercises. This is `coverage_test`'s UntestedKinds
# for labels, and entries are checked for being genuinely uncovered so that one
# that gains a test has to be removed from here.
UNCOVERED_LABELS = {
    # These exist only for unit tests, which write no testdata.
    "TestContext",
    "TestContext2",
    "TestInfo",
    "TestLabel",
    "TestSoftContext",
    "TestSoftContext2",
    "TestSortContext",
    # TODO: This is currently hard to test because it requires building and
    # importing a module, which attempts to create additional files with
    # unpredictable names in the module cache, which bazel doesn't permit.
    "InCppModule",
}


def strip_noise(content: str) -> str:
    """Returns `content` with comments and macro definitions blanked out.

    The macros are documented by example, and an example is not a
    declaration; they also name their own parameters, and
    `CARBON_DIAGNOSTIC_LABEL(LabelName, ...)` is not a label called
    `LabelName`. Blanking rather than deleting keeps offsets intact so that
    line numbers stay right.
    """
    # String literals are matched first and kept, so a `//` inside one -- a
    # URL in a diagnostic's message -- is not taken for a comment that hides
    # the rest of its line.
    content = re.sub(
        r'"(?:[^"\\\n]|\\.)*"|//[^\n]*',
        lambda m: m.group(0) if m.group(0).startswith('"') else "",
        content,
    )
    # A macro definition runs to the first line not continued with a backslash.
    # Its newlines are kept so that what follows keeps its line number.
    return re.sub(
        r"^[ \t]*#[ \t]*define(?:[^\n\\]|\\\n?)*",
        lambda m: "\n" * m.group(0).count("\n"),
        content,
        flags=re.M,
    )


class Loc(NamedTuple):
    """A location for a diagnostic."""

    @override
    def __str__(self) -> str:
        return f"{str(self.path)}:{self.line}"

    path: Path
    line: int


def load_diagnostic_kind() -> Set[str]:
    """Returns the set of declared diagnostic kinds.

    This isn't validated for uniqueness because the compiler does that.
    """
    path = Path("toolchain/diagnostics/kind.def")
    content = strip_noise(path.read_text())
    decls = set(re.findall(r"^CARBON_DIAGNOSTIC_KIND\((\w+)\)", content, re.M))
    return decls.difference(IGNORED)


def load_diagnostic_uses_in(
    path: Path,
) -> Dict[str, List[Loc]]:
    """Returns the path's CARBON_DIAGNOSTIC uses."""
    content = strip_noise(path.read_text())

    # Keep a line cursor so that we don't keep re-scanning the file.
    line = 1
    line_offset = 0

    found: Dict[str, List[Loc]] = collections.defaultdict(lambda: [])
    # `CARBON_DIAGNOSTIC_ON_SCOPE` declares a diagnostic too; the label,
    # context, and location-info macros do not, and have no kind to register.
    for m in re.finditer(
        r"CARBON_DIAGNOSTIC(?:_ON_SCOPE)?\(\s*(\w+),", content
    ):
        diag = m.group(1)
        if diag in IGNORED:
            continue
        line += content.count("\n", line_offset, m.start())
        line_offset = m.start()
        found[diag].append(Loc(path, line))
    return found


def load_labels_in(path: Path) -> Dict[str, List[Loc]]:
    """Returns the path's label, context, and location-info declarations.

    All three carry a name the same way a kind does, so all three are checked
    the same way, and `label` stands for any of them throughout this script.
    """
    content = strip_noise(path.read_text())

    found: Dict[str, List[Loc]] = collections.defaultdict(list)
    for m in re.finditer(LABEL_DECL_RE, content):
        label = m.group(1)
        if label not in IGNORED:
            found[label].append(
                Loc(path, 1 + content.count("\n", 0, m.start()))
            )
    return found


def load_unattached_labels_in(path: Path) -> List[Loc]:
    """Returns the path's labels that nothing attaches.

    A label is a file-local constant, so it must be named again in the file
    that declares it, by the `Attach` call that attaches it.

    Two declarations of one name in a file -- test scaffolding, and the reason
    same-file duplicates are allowed at all -- share their attach counts, so
    one of them going unattached hides behind the other's attaches.
    """
    content = strip_noise(path.read_text())
    return [
        loc
        for label, locs in load_labels_in(path).items()
        for loc in locs
        if len(re.findall(rf"\b{label}\b", content)) <= len(locs)
    ]


def sources() -> List[Path]:
    """Returns the toolchain sources to scan."""
    return list(
        itertools.chain(
            *[Path("toolchain").glob(f"**/*.{ext}") for ext in ("h", "cpp")]
        )
    )


def load_diagnostic_uses() -> Dict[str, List[Loc]]:
    """Returns all CARBON_DIAGNOSTIC uses."""
    with futures.ThreadPoolExecutor() as exec:
        results = exec.map(load_diagnostic_uses_in, sources())

    found: Dict[str, List[Loc]] = collections.defaultdict(lambda: [])
    for result in results:
        for diag, locations in result.items():
            found[diag].extend(locations)
    return found


def load_unattached_labels() -> List[Loc]:
    """Returns all label declarations that go unused."""
    with futures.ThreadPoolExecutor() as exec:
        results = exec.map(load_unattached_labels_in, sources())
    return [loc for result in results for loc in result]


def load_labels() -> Dict[str, List[Loc]]:
    """Returns all label declarations.

    A name is what a test matches on and what `--include-diagnostic-kind`
    prints, so two labels sharing one are indistinguishable to both. Every
    declaration is kept rather than the last so that `check_label_uniqueness`
    can say so.
    """
    with futures.ThreadPoolExecutor() as exec:
        results = exec.map(load_labels_in, sources())
    found: Dict[str, List[Loc]] = collections.defaultdict(list)
    for result in results:
        for label, locs in result.items():
            found[label].extend(locs)
    return found


def load_covered_names() -> Set[str]:
    """Returns the kinds and labels that a file_test matches on.

    This is the same line that `coverage_test` looks for, so the two agree on
    what a match looks like. They scan different sets: this walks every
    testdata file on disk, while `coverage_test` reads the files its bazel
    manifest names, so a file no test runs still counts here.
    """
    covered_re = re.compile(r"^ *// CHECK:STDERR: .* \[(\w+)\]$")

    def scan(path: Path) -> Set[str]:
        found = set()
        for line in path.read_text(errors="replace").splitlines():
            m = covered_re.match(line)
            if m:
                found.add(m.group(1))
        return found

    testdata = [
        p for p in Path("toolchain").glob("**/testdata/**/*") if p.is_file()
    ]
    with futures.ThreadPoolExecutor() as exec:
        return set().union(*exec.map(scan, testdata))


def check_uniqueness(uses: Dict[str, List[Loc]]) -> bool:
    """If any diagnostic is non-unique, prints an error and returns true."""
    has_errors = False
    for diag in sorted(uses.keys()):
        if len(uses[diag]) > 1:
            print(f"Non-unique diagnostic {diag}:", file=sys.stderr)
            for loc in uses[diag]:
                print(f"  - {loc}", file=sys.stderr)
            has_errors = True
    return has_errors


def check_label_uniqueness(labels: Dict[str, List[Loc]]) -> bool:
    """If a label name is declared in two files, prints and returns true.

    A name is what a test matches on and what `--include-diagnostic-kind`
    prints, so two labels sharing one are indistinguishable to both: a test
    naming it covers whichever the reader guesses.

    Declaring one name twice in a file is left alone. A label is scoped to
    where it is attached, so several tests declaring the same scaffolding are
    separate objects the author can see together.
    """
    has_errors = False
    for label, locs in sorted(labels.items()):
        files = sorted({loc.path for loc in locs})
        if len(files) > 1:
            print(
                f"Label {label} is declared in more than one file:",
                file=sys.stderr,
            )
            for loc in locs:
                print(f"  - {loc}", file=sys.stderr)
            has_errors = True
    return has_errors


def check_unused(decls: Set[str], uses: Dict[str, List[Loc]]) -> bool:
    """If any diagnostic is unused, prints an error and returns true."""
    unused = decls.difference(uses.keys())
    if not unused:
        return False
    for diag in sorted(unused):
        print(f"Unused diagnostic: {diag}", file=sys.stderr)
    return True


def check_unattached(unattached: List[Loc]) -> bool:
    """If any label is never attached, prints an error and returns true."""
    for loc in sorted(unattached):
        print(
            f"Unattached label: {loc} (a label is attached in the file that "
            "declares it)",
            file=sys.stderr,
        )
    return bool(unattached)


def check_label_kind_collision(
    decls: Set[str], labels: Dict[str, List[Loc]]
) -> bool:
    """If a label reuses a kind's name, prints an error and returns true.

    A label sharing a kind's name would satisfy both coverage checks with the
    other one's tests, so neither name would be checked at all.
    """
    has_errors = False
    for label in sorted(decls.intersection(labels)):
        print(
            f"Label {label} reuses a diagnostic kind's name: "
            f"{labels[label][0]}",
            file=sys.stderr,
        )
        has_errors = True
    return has_errors


def check_label_coverage(
    labels: Dict[str, List[Loc]], covered: Set[str]
) -> bool:
    """If label coverage doesn't match expectations, prints and returns true."""
    has_errors = False
    for label, locs in sorted(labels.items()):
        if label in UNCOVERED_LABELS:
            if label in covered:
                print(
                    f"Label {label} has coverage even though none was "
                    f"expected; remove it from UNCOVERED_LABELS: {locs[0]}",
                    file=sys.stderr,
                )
                has_errors = True
        elif label not in covered:
            print(f"Label has no tests: {label}: {locs[0]}", file=sys.stderr)
            has_errors = True
    return has_errors


def check_unknown_names(
    decls: Set[str], labels: Dict[str, List[Loc]], covered: Set[str]
) -> bool:
    """If a test names something unknown, prints an error and returns true."""
    unknown = sorted(covered - decls - set(labels) - IGNORED)
    for name in unknown:
        print(
            f"Tests match a name that is neither a kind nor a label: {name}",
            file=sys.stderr,
        )
    return bool(unknown)


def main() -> None:
    # Run from the repo root.
    os.chdir(Path(__file__).parents[2])
    decls = load_diagnostic_kind()
    uses = load_diagnostic_uses()
    labels = load_labels()
    unattached = load_unattached_labels()
    covered = load_covered_names()

    if any(
        [
            check_uniqueness(uses),
            check_label_uniqueness(labels),
            check_label_kind_collision(decls, labels),
            check_unused(decls, uses),
            check_unattached(unattached),
            check_label_coverage(labels, covered),
            check_unknown_names(decls, labels, covered),
        ]
    ):
        exit(1)


if __name__ == "__main__":
    main()
