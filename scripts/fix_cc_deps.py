#!/usr/bin/env python3

"""Automatically fixes bazel C++ dependencies.

Bazel has some support for detecting when an include refers to a missing
dependency. However, the ideal state is that a given build target depends
directly on all #include'd headers, and Bazel doesn't enforce that. This
automates the addition for technical correctness.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import hashlib
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
from typing import Callable, Dict, List, NamedTuple, Set, Tuple
import urllib.request
from xml.etree import ElementTree


# Maps external repository names to a method translating bazel labels to file
# paths for that repository.
EXTERNAL_REPOS: Dict[str, Callable[[str], str]] = {
    "@llvm-project": lambda x: re.sub("^(.*:(lib|include))/", "", x)
}

URL = "https://github.com/bazelbuild/buildtools/releases/download/4.2.5/"

# Checksums gathered with:
#   for v in darwin-amd64 darwin-arm64 linux-amd64 linux-arm64 windows-amd64.exe
#   do
#     echo \"$v\": \"$(wget -q -O - https://github.com/bazelbuild/buildtools/releases/download/4.2.5/buildozer-$v | sha256sum | cut -d ' ' -f1)\", \# noqa: E501
#   done
VERSIONS = {
    "darwin-amd64": "3fe671620e6cb7d2386f9da09c1de8de88b02b9dd9275cdecd8b9e417f74df1b",  # noqa: E501
    "darwin-arm64": "ff4d297023fe3e0fd14113c78f04cef55289ca5bfe5e45a916be738b948dc743",  # noqa: E501
    "linux-amd64": "e8e39b71c52318a9030dd9fcb9bbfd968d0e03e59268c60b489e6e6fc1595d7b",  # noqa: E501
    "linux-arm64": "96227142969540def1d23a9e8225524173390d23f3d7fd56ce9c4436953f02fc",  # noqa: E501
    "windows-amd64.exe": "2a9a7176cbd3b2f0ef989502128efbafd3b156ddabae93b9c979cd4017ffa300",  # noqa: E501
}


class Rule(NamedTuple):
    # For cc_* rules:
    # The hdrs + textual_hdrs attributes, as relative paths to the file.
    hdrs: Set[str]
    # The srcs attribute, as relative paths to the file.
    srcs: Set[str]
    # The deps attribute, as full bazel labels.
    deps: Set[str]

    # For genrules:
    # The outs attribute, as relative paths to the file.
    outs: Set[str]


def get_hash(file: Path) -> str:
    """Returns the sha256 of a file."""
    digest = hashlib.sha256()
    with file.open("rb") as f:
        while True:
            chunk = f.read(1024 * 64)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def install_buildozer() -> str:
    """Install buildozer to a cache."""
    cache_dir = Path.home().joinpath(".cache", "carbon-lang-pre-commit")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Translate platform information into Bazel's release form.
    machine = platform.machine()
    if machine == "x86_64":
        machine = "amd64"
    version = f"{platform.system().lower()}-{machine}"

    # Get ready to add .exe for Windows.
    ext = ""
    if platform.system() == "Windows":
        ext = ".exe"

    # Ensure the platform is supported, and grab its hash.
    if version not in VERSIONS:
        # If this because a platform support issue, we may need to print errors.
        exit(f"No buildozer available for platform: {version}")
    want_hash = VERSIONS[version]

    # Check if there's a cached file that can be used.
    local_path = cache_dir.joinpath(f"buildozer{ext}")
    if local_path.is_file() and want_hash == get_hash(local_path):
        return str(local_path)

    # Download buildozer.
    url = f"{URL}/buildozer-{version}{ext}"
    with urllib.request.urlopen(url) as response:
        with local_path.open("wb") as f:
            shutil.copyfileobj(response, f)
    local_path.chmod(0o755)

    # Verify the downloaded hash.
    found_hash = get_hash(local_path)
    if want_hash != found_hash:
        exit(
            f"Downloaded buildozer-{version} but found sha256 {found_hash}, "
            f"wanted {want_hash}"
        )

    return str(local_path)


def locate_bazel() -> str:
    """Returns the bazel command.

    We use the `BAZEL` environment variable if present. If not, then we try to
    use `bazelisk` and then `bazel`.
    """
    bazel = os.environ.get("BAZEL")
    if bazel:
        return bazel

    if shutil.which("bazelisk"):
        return "bazelisk"

    if shutil.which("bazel"):
        return "bazel"

    exit("Unable to run Bazel")


def remap_file(label: str) -> str:
    """Remaps a bazel label to a file."""
    repo, _, path = label.partition("//")
    if not repo:
        return path.replace(":", "/")
    assert repo in EXTERNAL_REPOS, repo
    return EXTERNAL_REPOS[repo](path)
    exit(f"Don't know how to remap label '{label}'")


def get_bazel_list(list_child: ElementTree.Element, is_file: bool) -> Set[str]:
    """Returns the contents of a bazel list.

    The return will normally be the full label, unless `is_file` is set, in
    which case the label will be translated to the underlying file.
    """
    results: Set[str] = set()
    for label in list_child:
        assert label.tag in ("label", "output"), label.tag
        value = label.attrib["value"]
        if is_file:
            value = remap_file(value)
        results.add(value)
    return results


def get_rules(targets: str, keep_going: bool) -> Dict[str, Rule]:
    """Queries the specified targets, returning the found rules.

    keep_going will be set to true for external repositories, where sometimes we
    see query errors.

    The return maps rule names to rule data.
    """
    args = [
        "bazel",
        "query",
        "--output=xml",
        f"kind('(cc_binary|cc_library|cc_test|genrule)', set({targets}))",
    ]
    if keep_going:
        args.append("--keep_going")
    p = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8"
    )
    # 3 indicates incomplete results from --keep_going, which is fine here.
    if p.returncode not in {0, 3}:
        print(p.stderr)
        exit(f"bazel query returned {p.returncode}")
    rules: Dict[str, Rule] = {}
    for rule_xml in ElementTree.fromstring(p.stdout):
        assert rule_xml.tag == "rule", rule_xml.tag
        rule_name = rule_xml.attrib["name"]
        hdrs: Set[str] = set()
        srcs: Set[str] = set()
        deps: Set[str] = set()
        outs: Set[str] = set()
        rule_class = rule_xml.attrib["class"]
        for list_child in rule_xml.findall("list"):
            list_name = list_child.attrib["name"]
            if rule_class in ("cc_library", "cc_binary", "cc_test"):
                if list_name in ("hdrs", "textual_hdrs"):
                    hdrs = hdrs.union(get_bazel_list(list_child, True))
                elif list_name == "srcs":
                    srcs = get_bazel_list(list_child, True)
                elif list_name == "deps":
                    deps = get_bazel_list(list_child, False)
            elif rule_class == "genrule":
                if list_name == "outs":
                    outs = get_bazel_list(list_child, True)
            else:
                exit(f"unexpected rule type: {rule_class}")
        rules[rule_name] = Rule(hdrs, srcs, deps, outs)
    return rules


def map_headers(
    header_to_rule_map: Dict[str, Set[str]], rules: Dict[str, Rule]
) -> None:
    """Accumulates headers provided by rules into the map.

    The map maps header paths to rule names.
    """
    for rule_name, rule in rules.items():
        for header in rule.hdrs:
            if header in header_to_rule_map:
                header_to_rule_map[header].add(rule_name)
            else:
                header_to_rule_map[header] = {rule_name}


def get_missing_deps(
    header_to_rule_map: Dict[str, Set[str]],
    generated_files: Set[str],
    rule: Rule,
) -> Tuple[Set[str], bool]:
    """Returns missing dependencies for the rule.

    On return, the set is dependency labels that should be added; the bool
    indicates whether some where omitted due to ambiguity.
    """
    missing_deps: Set[str] = set()
    ambiguous = False
    rule_files = rule.hdrs.union(rule.srcs)
    for source_file in rule_files:
        if source_file in generated_files:
            continue
        with open(source_file, "r") as f:
            for header in re.findall(
                r'^#include "([^"]+)"', f.read(), re.MULTILINE
            ):
                if header in rule_files:
                    continue
                if header not in header_to_rule_map:
                    exit(
                        f"Missing rule for #include '{header}' in "
                        f"'{source_file}'"
                    )
                dep_choices = header_to_rule_map[header]
                if not dep_choices.intersection(rule.deps):
                    if len(dep_choices) > 1:
                        print(
                            f"Ambiguous dependency choice for #include "
                            f"'{header}' in '{source_file}': "
                            f"{', '.join(dep_choices)}"
                        )
                        ambiguous = True
                    # Use the single dep without removing it.
                    missing_deps.add(next(iter(dep_choices)))
    return missing_deps, ambiguous


def main() -> None:
    # Change the working directory to the repository root so that the remaining
    # operations reliably operate relative to that root.
    os.chdir(Path(__file__).parent.parent)

    print("Querying bazel for Carbon targets...")
    carbon_rules = get_rules("//...", False)
    print("Querying bazel for external targets...")
    external_repo_query = " ".join([f"{repo}//..." for repo in EXTERNAL_REPOS])
    external_rules = get_rules(external_repo_query, True)

    print("Building header map...")
    header_to_rule_map: Dict[str, Set[str]] = {}
    map_headers(header_to_rule_map, carbon_rules)
    map_headers(header_to_rule_map, external_rules)

    print("Building generated file list...")
    generated_files: Set[str] = set()
    for rule in carbon_rules.values():
        generated_files = generated_files.union(rule.outs)

    print("Parsing headers from source files...")
    all_missing_deps: List[Tuple[str, Set[str]]] = []
    any_ambiguous = False
    for rule_name, rule in carbon_rules.items():
        missing_deps, ambiguous = get_missing_deps(
            header_to_rule_map, generated_files, rule
        )
        if missing_deps:
            all_missing_deps.append((rule_name, missing_deps))
        if ambiguous:
            any_ambiguous = True
    if any_ambiguous:
        exit("Stopping due to ambiguous dependency choices.")

    if all_missing_deps:
        print("Checking buildozer availability...")
        buildozer = install_buildozer()

        print("Fixing dependencies...")
        SEPARATOR = "\n- "
        for rule_name, missing_deps in sorted(all_missing_deps):
            friendly_missing_deps = SEPARATOR.join(missing_deps)
            print(
                f"Adding deps to {rule_name}:{SEPARATOR}{friendly_missing_deps}"
            )
            args = [
                buildozer,
                f"add deps {' '.join(missing_deps)}",
                rule_name,
            ]
            subprocess.check_call(args)

    print("Done!")


if __name__ == "__main__":
    main()
