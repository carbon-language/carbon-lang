__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import lit.formats
import os


# This is a provided variable, ignore the undefined name warning.
config = config  # noqa: F821


def add_file_substitution(substitution, relative_path):
    """Adds a substitution for a data file path."""
    config.substitutions.append(
        (substitution, os.path.join(os.environ["TEST_SRCDIR"], relative_path))
    )


config.name = "lit"
config.suffixes = [".carbon"]
config.test_format = lit.formats.ShTest()

add_file_substitution(
    "%{prelude}", "carbon/executable_semantics/data/prelude.carbon"
)
add_file_substitution(
    "%{executable_semantics}",
    "carbon/executable_semantics/executable_semantics",
)
add_file_substitution("%{not}", "llvm-project/llvm/not")
add_file_substitution("%{FileCheck}", "llvm-project/llvm/FileCheck")
