# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Lists (os, cpu) combinations supported by the Carbon build system."""

clang_configs = [
    ("linux", "aarch64"),
    ("linux", "x86_64"),
    ("freebsd", "x86_64"),
    ("macos", "arm64"),
    ("macos", "x86_64"),
    ("windows", "x86_64"),
]
