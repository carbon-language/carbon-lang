# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Installs Carbon to /usr/bin and /usr/lib/carbon.
#
# To install, run:
#   bazel -c opt //installers/local:install
#
# To uninstall, run:
#   bazel //installers/local:uninstall
#
# The build mode is important for installs, as debug versions may be installed.
# It is not relevant for uninstalls.

sh_binary(
    name = "install",
    srcs = ["install.sh"],
    args = [
        "--carbon",
        "$(location :carbon)",
        "$(locations //executable_semantics:standard_libraries)",
    ],
    data = [
        ":carbon",
        "//executable_semantics:standard_libraries",
    ],
)

sh_binary(
    name = "uninstall",
    srcs = ["uninstall.sh"],
)

# A busybox-like binary for Carbon tools; multiple tools are part of this binary
# so that shared code is deduplicated.
cc_binary(
    name = "carbon",
    srcs = ["carbon.cpp"],
    deps = [
        "//executable_semantics:main",
        "@llvm-project//llvm:Support",
    ],
)
