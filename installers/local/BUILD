# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_binary")
load("install.bzl", "install_path_rule")

# Turns `--//installers/local:install_path=arg` into `$(INSTALL_PATH)`.
install_path_rule(
    name = "install_path",
    build_setting_default = "/usr",
)

sh_binary(
    name = "install",
    srcs = ["install.sh"],
    args = [
        "--install_path",
        "$(INSTALL_PATH)",
        "--carbon",
        "$(location :carbon)",
        "$(locations //explorer:standard_libraries)",
    ],
    data = [
        ":carbon",
        "//explorer:standard_libraries",
    ],
    toolchains = [":install_path"],
)

sh_binary(
    name = "uninstall",
    srcs = ["uninstall.sh"],
    args = [
        "--install_path",
        "$(INSTALL_PATH)",
    ],
    toolchains = [":install_path"],
)

# A busybox-like binary for Carbon tools; multiple tools are part of this binary
# so that shared code is deduplicated.
cc_binary(
    name = "carbon",
    srcs = ["carbon.cpp"],
    visibility = ["//bazel/check_deps:__pkg__"],
    deps = [
        "//explorer:main",
        "@llvm-project//llvm:Support",
    ],
)
