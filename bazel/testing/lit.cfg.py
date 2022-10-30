__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import lit.formats
import os
from pathlib import Path


# This is a provided variable, ignore the undefined name warning.
config = config  # noqa: F821


def fullpath(relative_path):
    return Path(os.environ["TEST_SRCDIR"]).joinpath(relative_path)


def add_substitution(before, after):
    """Adds a substitution to the config. Wraps before as `%{before}`."""
    config.substitutions.append((f"%{{{before}}}", after))


def add_substitutions():
    """Adds required substitutions to the config."""
    tools = {
        "carbon": fullpath("carbon/toolchain/driver/carbon"),
        "explorer": fullpath("carbon/explorer/explorer"),
        "explorer_prelude": fullpath("carbon/explorer/data/prelude.carbon"),
        "filecheck": fullpath("llvm-project/llvm/FileCheck"),
        "not": fullpath("llvm-project/llvm/not"),
        "merge_output": fullpath("carbon/bazel/testing/merge_output"),
    }

    run_carbon = f"{tools['merge_output']} {tools['carbon']}"
    run_explorer = (
        f"{tools['merge_output']} {tools['explorer']} %s "
        f"--prelude={tools['explorer_prelude']}"
    )
    filecheck_allow_unmatched = (
        f"{tools['filecheck']} %s --match-full-lines --strict-whitespace"
    )
    filecheck_strict = (
        f"{filecheck_allow_unmatched} --implicit-check-not={{{{.}}}}"
    )

    add_substitution("carbon", f"{run_carbon}")
    add_substitution(
        "carbon-run-parser",
        f"{run_carbon} dump parse-tree %s | {filecheck_strict}",
    )
    add_substitution(
        "carbon-run-semantics",
        f"{run_carbon} dump semantics-ir %s | {filecheck_strict}",
    )
    add_substitution(
        "carbon-run-tokens", f"{run_carbon} dump tokens %s | {filecheck_strict}"
    )
    add_substitution(
        "explorer-run",
        f"{run_explorer} | {filecheck_strict}",
    )
    add_substitution(
        "explorer-run-trace",
        f"{run_explorer} --parser_debug --trace_file=- | "
        f"{filecheck_allow_unmatched}",
    )
    add_substitution("FileCheck-strict", filecheck_strict)
    add_substitution("not", tools["not"])


config.name = "lit"
config.suffixes = [".carbon"]
config.test_format = lit.formats.ShTest()
add_substitutions()
