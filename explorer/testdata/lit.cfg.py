__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import lit.formats
import os


# This is a provided variable, ignore the undefined name warning.
config = config  # noqa: F821


def fullpath(relative_path):
    return os.path.join(os.environ["TEST_SRCDIR"], relative_path)


config.name = "lit"
config.suffixes = [".carbon"]
config.test_format = lit.formats.ShTest()

config.substitutions.append(
    ("%{prelude}", fullpath("carbon/explorer/data/prelude.carbon"))
)
config.substitutions.append(
    (
        "%{explorer}",
        "%s --prelude=%s"
        % (
            fullpath("carbon/explorer/explorer"),
            fullpath("carbon/explorer/data/prelude.carbon"),
        ),
    )
)
config.substitutions.append(("%{not}", fullpath("llvm-project/llvm/not")))
config.substitutions.append(
    (
        "%{FileCheck}",
        fullpath("llvm-project/llvm/FileCheck --dump-input-filter=all"),
    )
)
