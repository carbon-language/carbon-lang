# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load(
    "@llvm-project//:vars.bzl",
    "LLVM_VERSION_MAJOR",
)
load("@rules_python//python:defs.bzl", "py_test")
load("//bazel/cc_rules:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("//bazel/manifest:defs.bzl", "manifest")
load("//toolchain/base:llvm_tools.bzl", "LLVM_MAIN_TOOLS", "LLVM_TOOL_ALIASES")
load(
    "//toolchain/base:runtimes_build_info.bzl",
    "generate_runtimes_build_vars",
)
load("install_filegroups.bzl", "install_filegroup", "install_symlink", "install_target", "make_install_filegroups")
load("pkg_helpers.bzl", "pkg_naming_variables", "pkg_tar_and_test")

package(default_visibility = ["//visibility:public"])

# Build rules supporting the install data tree for the Carbon toolchain.
#
# This populates a synthetic Carbon toolchain installation under the
# `prefix` directory. For details on its layout, see `install_dirs` below.

cc_library(
    name = "busybox_info",
    srcs = ["busybox_info.cpp"],
    hdrs = ["busybox_info.h"],
    deps = [
        "//common:error",
        "//common:exe_path",
        "//common:filesystem",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "busybox_info_test",
    size = "small",
    srcs = ["busybox_info_test.cpp"],
    deps = [
        ":busybox_info",
        "//common:check",
        "//common:filesystem",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

# This target doesn't include prelude libraries. To get a target that has the
# prelude available, use //toolchain.
cc_binary(
    name = "carbon-busybox",
    srcs = ["busybox_main.cpp"],
    deps = [
        ":busybox_info",
        "//common:all_llvm_targets",
        "//common:bazel_working_dir",
        "//common:error",
        "//common:exe_path",
        "//common:init_llvm",
        "//common:version_stamp",
        "//toolchain/base:install_paths",
        "//toolchain/base:llvm_tools_def",
        "//toolchain/driver",
        "@llvm-project//clang:driver",
        "@llvm-project//llvm:Support",
    ],
)

clang_aliases = [
    "clang",
    "clang++",
    "clang-cl",
    "clang-cpp",
]

# TODO: Add remaining aliases of LLD for Windows and WASM when we have support
# for them wired up through the busybox.
lld_aliases = [
    "ld.lld",
    "ld64.lld",
]

llvm_binaries = clang_aliases + lld_aliases + [
    tool.bin_name
    for tool in LLVM_MAIN_TOOLS.values()
] + [
    "llvm-" + alias
    for (_, aliases) in LLVM_TOOL_ALIASES.items()
    for alias in aliases
]

# Generate a Starlark file with all the build variables needed for our runtimes.
generate_runtimes_build_vars(
    name = "staging_bazel/runtimes_build_vars.bzl",
)

# Given a CMake-style install prefix[1], the hierarchy looks like:
#
# - prefix/bin: Binaries intended for direct use.
# - prefix/lib/carbon: Private data and files.
# - prefix/lib/carbon/core: The `Core` package files.
# - prefix/lib/carbon/llvm/bin: LLVM binaries.
#
# This will be how installs are provided on Unix-y platforms, and is loosely
# based on the FHS (Filesystem Hierarchy Standard). See the CMake install prefix
# documentation[1] for more details.
#
# [1]: https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html
install_dirs = {
    "bin": [
        install_symlink(
            "carbon",
            "../lib/carbon/carbon-busybox",
            is_driver = True,
        ),
    ],
    "lib/carbon": [
        install_target("MODULE.bazel", "bazel/install.MODULE.bazel"),
        install_target("BUILD", "bazel/install.BUILD"),
        install_target("carbon_install.txt", "carbon_install.txt"),
        install_target(
            "install_digest.txt",
            ":install_digest.txt",
            executable = False,
            is_digest = True,
            is_driver = True,
        ),
        install_target(
            "carbon-busybox",
            ":carbon-busybox",
            executable = True,
            is_driver = True,
        ),
        install_filegroup("bazel", "//bazel/cc_toolchains:installed_cc_toolchain_starlark"),
        # TODO: Consider if we want to keep `core` here or group it with
        # runtimes. It is a bit of both -- standard library, and runtimes.
        install_filegroup("core", "//core:prelude"),
    ],
    "lib/carbon/bazel": [
        install_target("carbon_cc_toolchain_config.bzl", "bazel/carbon_cc_toolchain_config.bzl"),
        install_target("carbon_detected_variables.tpl.bzl", "bazel/carbon_detected_variables.tpl.bzl"),
        install_target("carbon_runtimes.bzl", "bazel/carbon_runtimes.bzl"),
        install_target("carbon_toolchain.bzl", "bazel/carbon_toolchain.bzl"),
        install_target("make_include_copts.bzl", "bazel/make_include_copts.bzl"),
        install_target("runtimes_build_vars.bzl", "staging_bazel/runtimes_build_vars.bzl"),
        install_target("BUILD", "bazel/empty.BUILD"),
    ],
    "lib/carbon/llvm/bin": [install_symlink(
        name,
        "../../carbon-busybox",
        is_driver = True,
    ) for name in llvm_binaries],
    "lib/carbon/runtimes": [
        install_filegroup(
            "builtins",
            "//toolchain/runtimes:clang_builtins_runtimes",
            remove_prefix = "lib/builtins/",
        ),
        install_filegroup("libcxx", "//toolchain/runtimes:libcxx"),
        install_filegroup("libcxxabi", "//toolchain/runtimes:libcxxabi"),
        install_filegroup("libunwind", "//toolchain/runtimes:libunwind"),
        install_target("BUILD", "bazel/runtimes.BUILD"),
    ],
    "lib/carbon/runtimes/libc": [
        install_filegroup("internal", "//toolchain/runtimes:libc_internal"),
    ],
    "lib/carbon/runtimes/libcxx": [
        install_filegroup(
            "include",
            "//toolchain/runtimes:libcxx_gen_files",
            remove_prefix = "staging_libcxx/include/",
        ),
    ],
    "lib/carbon/llvm/lib/clang/" + LLVM_VERSION_MAJOR: [
        install_target("BUILD", "bazel/clang_resource_dir.BUILD"),
        install_filegroup(
            "include",
            "@llvm-project//clang:builtin_headers_gen",
            label = "installed_clang_headers",
            remove_prefix = "staging/include/",
        ),
    ],
    "lib/carbon/llvm/lib/clang/" + LLVM_VERSION_MAJOR + "/include": [
        install_filegroup(
            "fuzzer",
            "@llvm-project//compiler-rt:fuzzer_installed_hdrs",
            remove_prefix = "include/fuzzer/",
        ),
        install_filegroup(
            "profile",
            "@llvm-project//compiler-rt:profile_installed_hdrs",
            remove_prefix = "include/profile/",
        ),
        install_filegroup(
            "sanitizer",
            "@llvm-project//compiler-rt:sanitizer_installed_hdrs",
            remove_prefix = "include/sanitizer/",
        ),
    ],
}

make_install_filegroups(
    name = "install_data",
    install_dirs = install_dirs,
    no_digest_name = "install_data.no_digest",
    no_driver_name = "install_data.no_driver",
    pkg_name = "pkg_data",
    prefix = "prefix",
)

py_test(
    name = "llvm_symlinks_test",
    size = "small",
    srcs = ["llvm_symlinks_test.py"],
    data = [
        ":install_data",
        "//toolchain/driver:prebuilt_runtimes",
    ],
    deps = ["@bazel_tools//tools/python/runfiles"],
)

manifest(
    name = "install_data_manifest.txt",
    srcs = [":install_data"],
)

cc_binary(
    name = "make-installation-digest",
    srcs = ["make_installation_digest.cpp"],
    deps = [
        "//common:bazel_working_dir",
        "//common:error",
        "//common:exe_path",
        "//common:filesystem",
        "//common:init_llvm",
        "//common:map",
        "//common:vlog",
        "@llvm-project//llvm:Support",
    ],
)

manifest(
    name = "install_digest_manifest.txt",
    srcs = [":install_data.no_digest"],
)

genrule(
    name = "gen_digest",
    srcs = [
        ":install_data.no_digest",
        "install_digest_manifest.txt",
    ],
    outs = [":install_digest.txt"],
    cmd = "$(location :make-installation-digest) " +
          "$(location install_digest_manifest.txt) $@",
    tools = [":make-installation-digest"],
)

# A list of clang's installed builtin header files.
# This is consumed by //toolchain/testing:file_test.
manifest(
    name = "clang_headers_manifest.txt",
    srcs = [":installed_clang_headers"],
    strip_package_dir = True,
)

pkg_naming_variables(
    name = "packaging_variables",
)

# We build both a compressed and uncompressed tar file with the same code here.
# This lets us use the tar file in testing as it is fast to create, but ship the
# compressed version as a release.
#
# For manual tests, the tar rules are `carbon_toolchain_tar_rule` and
# `carbon_toolchain_tar_gz_rule`.
pkg_tar_and_test(
    srcs = [":pkg_data"],
    install_data_manifest = ":install_data_manifest.txt",
    name_base = "carbon_toolchain",
    package_dir = "carbon_toolchain-$(version)",
    package_file_name_base = "carbon_toolchain-$(version)",
    package_variables = ":packaging_variables",
    stamp = -1,  # Allow `--stamp` builds to produce file timestamps.
)
