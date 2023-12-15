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

import re
import subprocess
from typing import Callable, Dict, List, NamedTuple, Set, Tuple
from xml.etree import ElementTree

import scripts_utils


class ExternalRepo(NamedTuple):
    # A function for remapping files to #include paths.
    remap: Callable[[str], str]
    # The target expression to gather rules for within the repo.
    target: str


# Maps external repository names to a method translating bazel labels to file
# paths for that repository.
EXTERNAL_REPOS: Dict[str, ExternalRepo] = {
    # llvm:include/llvm/Support/Error.h ->llvm/Support/Error.h
    # clang-tools-extra/clangd:URI.h -> clang-tools-extra/clangd/URI.h
    "@llvm-project": ExternalRepo(
        lambda x: re.sub(":", "/", re.sub("^(.*:(lib|include))/", "", x)),
        "...",
    ),
    # :src/google/protobuf/descriptor.h -> google/protobuf/descriptor.h
    # - protobuf_headers is specified because there are multiple overlapping
    #   targets.
    "@com_google_protobuf": ExternalRepo(
        lambda x: re.sub("^(.*:src)/", "", x),
        ":protobuf_headers",
    ),
    # :src/libfuzzer/libfuzzer_macro.h -> libfuzzer/libfuzzer_macro.h
    "@com_google_libprotobuf_mutator": ExternalRepo(
        lambda x: re.sub("^(.*:src)/", "", x), "..."
    ),
    # tools/cpp/runfiles:runfiles.h -> tools/cpp/runfiles/runfiles.h
    "@bazel_tools": ExternalRepo(lambda x: re.sub(":", "/", x), "..."),
    # absl/flags:flag.h -> absl/flags/flag.h
    "@com_google_absl": ExternalRepo(lambda x: re.sub(":", "/", x), "..."),
    # :re2/re2.h -> re2/re2.h
    "@com_googlesource_code_re2": ExternalRepo(
        lambda x: re.sub(":", "", x), ":re2"
    ),
}

# TODO: proto rules are aspect-based and their generated files don't show up in
# `bazel query` output.
# Try using `bazel cquery --output=starlark` to print `target.files`.
# For protobuf, need to add support for `alias` rule kind.
IGNORE_HEADER_REGEX = re.compile("^(.*\\.pb\\.h)$")
IGNORE_SOURCE_FILE_REGEX = re.compile("^third_party/clangd")


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


def remap_file(label: str) -> str:
    """Remaps a bazel label to a file."""
    repo, _, path = label.partition("//")
    if not repo:
        return path.replace(":", "/")
    # Ignore the version, just use the repo name.
    repo = repo.split("~", 1)[0]
    assert repo in EXTERNAL_REPOS, repo
    return EXTERNAL_REPOS[repo].remap(path)


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


def get_rules(bazel: str, targets: str, keep_going: bool) -> Dict[str, Rule]:
    """Queries the specified targets, returning the found rules.

    keep_going will be set to true for external repositories, where sometimes we
    see query errors.

    The return maps rule names to rule data.
    """
    args = [
        bazel,
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
            elif rule_class == "tree_sitter_cc_library":
                continue
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
        repo, _, path = rule_name.partition("//")
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
        if IGNORE_SOURCE_FILE_REGEX.match(source_file):
            continue

        with open(source_file, "r") as f:
            for header_groups in re.findall(
                r'^(#include (?:"([^"]+)"|'
                r"<((?:google|gmock|gtest|libfuzzer)/[^>]+)>))",
                f.read(),
                re.MULTILINE,
            ):
                # Ignore whether the source was a quote or system include.
                header = header_groups[1]
                if not header:
                    header = header_groups[2]

                if header in rule_files:
                    continue
                if header not in header_to_rule_map:
                    if IGNORE_HEADER_REGEX.match(header):
                        print(
                            f"Ignored missing "
                            f"'{header_groups[0]}' in '{source_file}'"
                        )
                        continue
                    else:
                        exit(
                            f"Missing rule for "
                            f"'{header_groups[0]}' in '{source_file}'"
                        )
                dep_choices = header_to_rule_map[header]
                if not dep_choices.intersection(rule.deps):
                    if len(dep_choices) > 1:
                        print(
                            f"Ambiguous dependency choice for "
                            f"'{header_groups[0]}' in '{source_file}': "
                            f"{', '.join(dep_choices)}"
                        )
                        ambiguous = True
                    # Use the single dep without removing it.
                    missing_deps.add(next(iter(dep_choices)))
    return missing_deps, ambiguous


def main() -> None:
    scripts_utils.chdir_repo_root()
    bazel = scripts_utils.locate_bazel()

    print("Querying bazel for Carbon targets...")
    carbon_rules = get_rules(bazel, "//...", False)
    print("Querying bazel for external targets...")
    external_repo_query = " ".join(
        [f"{repo}//{EXTERNAL_REPOS[repo].target}" for repo in EXTERNAL_REPOS]
    )
    external_rules = get_rules(bazel, external_repo_query, True)

    print("Building header map...")
    header_to_rule_map: Dict[str, Set[str]] = {
        "gmock/gmock.h": {"@com_google_googletest//:gtest"},
        "gtest/gtest.h": {"@com_google_googletest//:gtest"},
    }
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
        buildozer = scripts_utils.get_release(scripts_utils.Release.BUILDOZER)

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
