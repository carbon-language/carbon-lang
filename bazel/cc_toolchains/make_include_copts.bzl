# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Workaround for lack of easy way to add private includes.

This is a workaround for the following Bazel issue:
https://github.com/bazelbuild/bazel/issues/2670
"""

def make_include_copts(include_dirs):
    """Create `copts` to search the provided directories for includes."""
    copts = []
    prefix = ""

    repo_name = native.repository_name()
    if repo_name != "@":
        prefix = "external/{}/".format(repo_name[1:])

    package_name = native.package_name()
    for include_dir in include_dirs:
        copts.append("-I{}{}/{}".format(prefix, package_name, include_dir))
        copts.append("-I$(BINDIR)/{}{}/{}".format(prefix, package_name, include_dir))
        copts.append("-I$(GENDIR)/{}{}/{}".format(prefix, package_name, include_dir))

    return copts
