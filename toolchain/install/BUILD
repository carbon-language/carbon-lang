# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Build rules supporting the install data tree for the Carbon toolchain.
#
# This populates a synthetic Carbon toolchain installation rooted at this
# package in the output tree.
#
# We use a `toolchain_files` rule organize the files in in the installation and
# handle cases where we need to symlink existing files or build outputs
# into the correct location.
#
# This package also includes the logic for building Carbon's runtimes, and Bazel
# `cc_toolchain`s for using the installation as a C++ toolchain in Bazel. This
# allows multi-stage bootstrapping to occur entirely within Bazel. Note that the
# specific stage will be in a _configuration specific_ output tree, but always
# rooted under the install package.
#
# Last but ont least, we provide rules to build packages of the installation in
# conventional layouts for directly extracting and using a binary build. The
# packaged installation includes custom Bazel files to handle a pre-built and
# installed toolchain in the same way as these handle the just-built toolchain.

load(
    "@llvm-project//:vars.bzl",
    "LLVM_VERSION_MAJOR",
)
load(
    "@llvm-project//compiler-rt:compiler-rt.bzl",
    "builtins_copts",
    "crt_copts",
)
load("@llvm-project//libcxx:libcxx_library.bzl", "libcxx_and_abi_copts")
load("@llvm-project//libunwind:libunwind_library.bzl", "libunwind_copts")
load("@rules_python//python:defs.bzl", "py_test")
load("//bazel/cc_rules:defs.bzl", "cc_binary", "cc_library", "cc_test")
load(
    "//bazel/cc_toolchains:carbon_cc_toolchain_config.bzl",
    "carbon_cc_toolchain_suite",
    "set_platform_filegroup",
)
load("//bazel/manifest:defs.bzl", "manifest")
load(
    "//toolchain/base:runtimes_build_info.bzl",
    "generate_runtimes_build_vars",
)
load("//toolchain/runtimes:carbon_runtimes.bzl", "carbon_runtimes")
load("bazel/make_include_copts.bzl", "make_include_copts")
load(
    "install_filegroups.bzl",
    "filtered_toolchain_files",
    "toolchain_files",
    "toolchain_llvm_binaries",
    "toolchain_pkg_filegroup",
)
load("pkg_helpers.bzl", "pkg_naming_variables", "pkg_tar_and_test")

package(default_visibility = ["//visibility:public"])

constraint_setting(
    name = "runtimes_build",
)

constraint_value(
    name = "is_runtimes_build",
    constraint_setting = ":runtimes_build",
)

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

toolchain_files(
    name = "install_marker",
    srcs = ["carbon_install.txt"],
)

toolchain_llvm_binaries(
    name = "llvm_bins",
    carbon_binary = ":carbon-busybox",
)

toolchain_files(
    name = "core",
    srcs = ["//core:prelude"],
    prefix = "core/",
)

toolchain_files(
    name = "clang_builtin_hdrs",
    srcs = ["@llvm-project//clang:builtin_headers_gen"],
    prefix = "llvm/lib/clang/{0}/include/".format(LLVM_VERSION_MAJOR),
    remove_prefix = "staging/include/",
)

toolchain_files(
    name = "fuzzer_runtime_hdrs",
    srcs = ["@llvm-project//compiler-rt:fuzzer_installed_hdrs"],
    prefix = "llvm/lib/clang/{0}/include/".format(LLVM_VERSION_MAJOR),
    remove_prefix = "include/",
)

toolchain_files(
    name = "profile_runtime_hdrs",
    srcs = ["@llvm-project//compiler-rt:profile_installed_hdrs"],
    prefix = "llvm/lib/clang/{0}/include/".format(LLVM_VERSION_MAJOR),
    remove_prefix = "include/",
)

toolchain_files(
    name = "sanitizer_runtime_hdrs",
    srcs = ["@llvm-project//compiler-rt:sanitizer_installed_hdrs"],
    prefix = "llvm/lib/clang/{0}/include/".format(LLVM_VERSION_MAJOR),
    remove_prefix = "include/",
)

filegroup(
    name = "clang_hdrs",
    srcs = [
        ":clang_builtin_hdrs",
        ":fuzzer_runtime_hdrs",
        ":profile_runtime_hdrs",
        ":sanitizer_runtime_hdrs",
    ],
)

filtered_toolchain_files(
    name = "builtins_srcs",
    prefix = "runtimes/builtins/",
    remove_prefix = "lib/builtins/",
    srcs_groups = [
        "@llvm-project//compiler-rt:builtins_aarch64_srcs",
        "@llvm-project//compiler-rt:builtins_aarch64_textual_srcs",
        "@llvm-project//compiler-rt:builtins_crtbegin_src",
        "@llvm-project//compiler-rt:builtins_crtend_src",
        "@llvm-project//compiler-rt:builtins_i386_srcs",
        "@llvm-project//compiler-rt:builtins_i386_textual_srcs",
        "@llvm-project//compiler-rt:builtins_x86_64_srcs",
        "@llvm-project//compiler-rt:builtins_x86_64_textual_srcs",
    ],
)

toolchain_files(
    name = "libcxx_basic_hdrs",
    srcs = ["@llvm-project//libcxx:libcxx_hdrs"],
    prefix = "runtimes/libcxx/include/",
    remove_prefix = "include/",
)

toolchain_files(
    name = "libcxx_gen_hdrs",
    srcs = ["//toolchain/runtimes:libcxx_gen_files"],
    prefix = "runtimes/libcxx/include/",
    remove_prefix = "staging_libcxx/include/",
)

filegroup(
    name = "libcxx_hdrs",
    srcs = [
        ":libcxx_basic_hdrs",
        ":libcxx_gen_hdrs",
    ],
)

filtered_toolchain_files(
    name = "libcxx_srcs",
    prefix = "runtimes/libcxx/src/",
    remove_prefix = "src/",
    srcs_groups = [
        "@llvm-project//libcxx:libcxx_linux_srcs",
        "@llvm-project//libcxx:libcxx_macos_srcs",
        "@llvm-project//libcxx:libcxx_win32_srcs",
    ],
)

toolchain_files(
    name = "libcxxabi_hdrs",
    srcs = ["@llvm-project//libcxxabi:libcxxabi_hdrs"],
    prefix = "runtimes/libcxxabi/include/",
    remove_prefix = "include/",
)

toolchain_files(
    name = "libcxxabi_srcs",
    srcs = ["@llvm-project//libcxxabi:libcxxabi_srcs"],
    prefix = "runtimes/libcxxabi/src/",
    remove_prefix = "src/",
)

toolchain_files(
    name = "libcxxabi_textual_srcs",
    srcs = ["@llvm-project//libcxxabi:libcxxabi_textual_srcs"],
    prefix = "runtimes/libcxxabi/src/",
    remove_prefix = "src/",
)

toolchain_files(
    name = "libc_internal_libcxx_hdrs",
    srcs = ["@llvm-project//libc:libcxx_shared_headers_hdrs"],
    prefix = "runtimes/libc/internal/",
    #remove_prefix = "src",
)

toolchain_files(
    name = "libunwind_hdrs",
    srcs = ["@llvm-project//libunwind:libunwind_hdrs"],
    prefix = "runtimes/libunwind/include/",
    remove_prefix = "include/",
)

toolchain_files(
    name = "libunwind_srcs",
    srcs = ["@llvm-project//libunwind:libunwind_srcs"],
    prefix = "runtimes/libunwind/src/",
    remove_prefix = "src/",
)

filegroup(
    name = "cc_toolchain_all_files",
    srcs = [
        ":carbon-busybox",
        ":clang_hdrs",
        ":core",
        ":install_marker",
        ":libcxx_hdrs",
        ":libcxxabi_hdrs",
        ":libunwind_hdrs",
        ":llvm_bins",
    ],
)

toolchain_files(
    name = "bazel_common_srcs",
    srcs = ["//bazel/cc_toolchains:installed_cc_toolchain_starlark"],
    prefix = "bazel/",
)

toolchain_files(
    name = "bazel_install_srcs",
    srcs = [
        "bazel/carbon_clang_variables.bzl",
        "bazel/carbon_detected_variables.tpl.bzl",
        "bazel/carbon_toolchain.bzl",
        "bazel/make_include_copts.bzl",
    ],
)

toolchain_files(
    name = "carbon_runtimes_installed",
    srcs = ["//toolchain/runtimes:carbon_runtimes.bzl"],
    prefix = "bazel/",
)

toolchain_files(
    name = "bazel_build_and_module",
    srcs = [
        "bazel/empty.BUILD",
        "bazel/install.BUILD",
        "bazel/install.MODULE.bazel",
    ],
    remove_prefix = "bazel/",
    renames = {
        "empty.BUILD": "bazel/BUILD.bazel",
        "install.BUILD": "BUILD.bazel",
        "install.MODULE.bazel": "MODULE.bazel",
    },
)

# Generate a Starlark file with all the build variables needed for our runtimes.
generate_runtimes_build_vars(
    name = "bazel/runtimes_build_vars.bzl",
)

filegroup(
    name = "installed_bazel_files",
    srcs = [
        ":bazel_build_and_module",
        ":bazel_common_srcs",
        ":bazel_install_srcs",
        ":carbon_runtimes_installed",

        # Note that we have to put this here and not in one of the
        # `toolchain_files` because it is a generated file.
        "bazel/runtimes_build_vars.bzl",
    ],
)

filegroup(
    name = "all_data_files",
    srcs = [
        ":builtins_all_srcs",
        ":clang_hdrs",
        ":core",
        ":install_marker",
        ":installed_bazel_files",
        ":libc_internal_libcxx_hdrs",
        ":libcxx_all_srcs",
        ":libcxx_hdrs",
        ":libcxxabi_hdrs",
        ":libcxxabi_srcs",
        ":libunwind_hdrs",
        ":libunwind_srcs",
    ],
)

# A list of clang's installed builtin header files.
# This is consumed by //toolchain/testing:file_test.
manifest(
    name = "clang_headers_manifest.txt",
    srcs = [":clang_hdrs"],
    strip_package_dir = True,
)

filegroup(
    name = "all_digest_files",
    srcs = [
        ":all_data_files",
        ":carbon-busybox",
    ],
)

manifest(
    name = "install_digest_manifest.txt",
    srcs = [":all_digest_files"],
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

genrule(
    name = "gen_digest",
    srcs = [
        ":all_digest_files",
        "install_digest_manifest.txt",
    ],
    outs = ["install_digest.txt"],
    cmd = "$(location :make-installation-digest) " +
          "$(location install_digest_manifest.txt) $@",
    tools = [":make-installation-digest"],
)

filegroup(
    name = "install_data",
    srcs = [
        "install_digest.txt",
        ":all_digest_files",
        ":llvm_bins",
    ],
)

manifest(
    name = "install_data_manifest.txt",
    srcs = [":install_data"],
)

cc_library(
    name = "builtins_internal",
    hdrs_check = "strict",
    textual_hdrs = select({
        "@platforms//cpu:aarch64": [":builtins_aarch64_textual_srcs"],
        "@platforms//cpu:i386": [":builtins_i386_textual_srcs"],
        "@platforms//cpu:x86_64": [":builtins_x86_64_textual_srcs"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "builtins",
    srcs = select({
        "@platforms//cpu:aarch64": [":builtins_aarch64_srcs"],
        "@platforms//cpu:i386": [":builtins_i386_srcs"],
        "@platforms//cpu:x86_64": [":builtins_x86_64_srcs"],
        "//conditions:default": [],
    }),
    copts = builtins_copts + make_include_copts([
        "runtimes/builtins",
    ]) + [
        # TODO: Remove this once we are building sanitizer runtime libraries.
        "-fno-sanitize=all",
    ],
    hdrs_check = "strict",
    target_compatible_with = select({
        "//bazel/cc_toolchains:stage1": [],
        "//bazel/cc_toolchains:stage2": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [":builtins_internal"],
)

filegroup(
    name = "builtins_archive",
    srcs = [":builtins"],
    output_group = "archive",
)

cc_library(
    name = "libunwind",
    srcs = [":libunwind_srcs"],
    hdrs = [":libunwind_hdrs"],
    copts = libunwind_copts + [
        # We disable all warnings as upstream isn't clean with the common
        # warning flags Carbon uses by default.
        "-w",

        # TODO: Remove this once we are building sanitizer runtime libraries.
        "-fno-sanitize=all",
    ],
    hdrs_check = "strict",
    includes = ["runtimes/libunwind/include"],
    linkstatic = 1,
    target_compatible_with = select({
        "//bazel/cc_toolchains:stage1": [],
        "//bazel/cc_toolchains:stage2": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

filegroup(
    name = "libunwind_archive",
    srcs = [":libunwind"],
    output_group = "archive",
)

cc_library(
    name = "libcxxabi_internal",
    hdrs_check = "strict",
    target_compatible_with = select({
        "//bazel/cc_toolchains:stage1": [],
        "//bazel/cc_toolchains:stage2": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    textual_hdrs = [":libcxxabi_textual_srcs"],
)

cc_library(
    name = "libc_internal_libcxx",
    hdrs = [":libc_internal_libcxx_hdrs"],
    hdrs_check = "strict",
    target_compatible_with = select({
        "//bazel/cc_toolchains:stage1": [],
        "//bazel/cc_toolchains:stage2": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

cc_library(
    name = "libcxx",
    srcs = select({
        "@platforms//os:macos": [":libcxx_macos_srcs"],
        "@platforms//os:windows": [":libcxx_win32_srcs"],
        "//conditions:default": [":libcxx_linux_srcs"],
    }) + [":libcxxabi_srcs"],
    hdrs = [
        ":libcxx_hdrs",
        ":libcxxabi_hdrs",
    ],
    copts = libcxx_and_abi_copts + make_include_copts([
        "runtimes/libcxx/src",
        "runtimes/libc/internal",
    ]) + [
        # We disable all warnings as upstream isn't clean with the common
        # warning flags Carbon uses by default.
        "-w",

        # TODO: Remove this once we are building sanitizer runtime libraries.
        "-fno-sanitize=all",
    ],
    hdrs_check = "strict",
    includes = [
        "runtimes/libcxx/include",
        "runtimes/libcxxabi/include",
    ],
    target_compatible_with = select({
        "//bazel/cc_toolchains:stage1": [],
        "//bazel/cc_toolchains:stage2": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":libc_internal_libcxx",
        ":libcxxabi_internal",
    ],
)

filegroup(
    name = "libcxx_archive",
    srcs = [":libcxx"],
    output_group = "archive",
)

carbon_runtimes(
    name = "runtimes",
    builtins_archive = ":builtins_archive",
    clang_hdrs = [":clang_hdrs"],
    clang_hdrs_prefix = "llvm/lib/clang/{0}/include/".format(LLVM_VERSION_MAJOR),
    crt_copts = crt_copts + [
        # Disable all sanitizers for CRT objects.
        "-fno-sanitize=all",
    ],
    crtbegin_src = select({
        "@platforms//os:linux": ":builtins_crtbegin_src",
        "//conditions:default": None,
    }),
    crtend_src = select({
        "@platforms//os:linux": ":builtins_crtend_src",
        "//conditions:default": None,
    }),
    darwin_os_suffix = select({
        # TODO: Add support for tvOS, watchOS, and iOS variants with the
        # relevant Bazel constraints.
        ":is_macos_arm64": "osx",
        ":is_macos_x86_64": "osx",
        "//conditions:default": None,
    }),
    libcxx_archive = ":libcxx_archive",
    libunwind_archive = ":libunwind_archive",
    target_compatible_with = select({
        "//bazel/cc_toolchains:stage1": [],
        "//bazel/cc_toolchains:stage2": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    target_triple = select({
        # TODO: Add other triples (and if needed, constraints) so that we can
        # build the correct Clang resource-dir structure for each.
        ":is_freebsd_x86_64": "x86_64-unknown-freebsd",
        ":is_linux_aarch64": "aarch64-unknown-linux-gnu",
        ":is_linux_x86_64": "x86_64-unknown-linux-gnu",

        # Note that Darwin OSes are handled by the `darwin_os_suffix` attribute.
        "//conditions:default": None,
    }),
)

platforms = {
    "freebsd": ["x86_64"],
    "linux": [
        "aarch64",
        "x86_64",
    ],
    "macos": [
        "arm64",
        "x86_64",
    ],
}

[
    config_setting(
        name = "is_{0}_{1}".format(os, cpu),
        constraint_values = [
            "@platforms//os:" + os,
            "@platforms//cpu:" + cpu,
        ],
    )
    for os, cpus in platforms.items()
    for cpu in cpus
]

carbon_cc_toolchain_suite(
    name = "carbon_stage1",
    all_hdrs = [
        ":clang_hdrs",
        ":libunwind_hdrs",
        ":libcxx_hdrs",
        ":libcxxabi_hdrs",
    ],
    base_files = [
        ":install_marker",
        ":carbon-busybox",
        ":llvm_bins",
    ],
    build_stage = "//bazel/cc_toolchains:stage1",
    clang_hdrs = [":clang_hdrs"],
    platforms = platforms,
    runtimes = ":runtimes",
)

carbon_cc_toolchain_suite(
    name = "carbon_stage2",
    all_hdrs = [
        ":clang_hdrs",
        ":libunwind_hdrs",
        ":libcxx_hdrs",
        ":libcxxabi_hdrs",
    ],
    base_files = [
        ":install_marker",
        ":carbon-busybox",
        ":llvm_bins",
    ],
    base_stage = "//bazel/cc_toolchains:stage1",
    build_stage = "//bazel/cc_toolchains:stage2",
    clang_hdrs = [":clang_hdrs"],
    platforms = platforms,
    runtimes = ":runtimes",
    tags = ["manual"],
)

pkg_naming_variables(
    name = "packaging_variables",
)

toolchain_pkg_filegroup(
    name = "stage1_pkg_data",
    srcs = [
        "install_digest.txt",
        ":all_data_files",
    ],
    carbon_busybox = ":carbon-busybox",
)

# We build both a compressed and uncompressed tar file with the same code here.
# This lets us use the tar file in testing as it is fast to create, but ship the
# compressed version as a release.
#
# For manual tests, the tar rules are `carbon_toolchain_tar_rule` and
# `carbon_toolchain_tar_gz_rule`.
pkg_tar_and_test(
    srcs = [":stage1_pkg_data"],
    install_data_manifest = ":install_data_manifest.txt",
    name_base = "carbon_toolchain",
    package_dir = "carbon_toolchain-$(version)",
    package_file_name_base = "carbon_toolchain-$(version)",
    package_variables = ":packaging_variables",
    stamp = -1,  # Allow `--stamp` builds to produce file timestamps.
)

set_platform_filegroup(
    name = "stage2_pkg_files",
    srcs = [
        ":all_data_files",
        ":gen_digest",
    ],
    platform = select({
        ":is_{}_{}".format(os, cpu): ":carbon_stage2_base_{}_{}_platform".format(os, cpu)
        for os, cpus in platforms.items()
        for cpu in cpus
    }),
    tags = ["manual"],
)

set_platform_filegroup(
    name = "stage2_busybox",
    srcs = [":carbon-busybox"],
    platform = select({
        ":is_{}_{}".format(os, cpu): ":carbon_stage2_base_{}_{}_platform".format(os, cpu)
        for os, cpus in platforms.items()
        for cpu in cpus
    }),
    tags = ["manual"],
)

toolchain_pkg_filegroup(
    name = "stage2_pkg_data",
    srcs = [":stage2_pkg_files"],
    carbon_busybox = ":stage2_busybox",
    tags = ["manual"],
)

pkg_tar_and_test(
    srcs = [":stage2_pkg_data"],
    install_data_manifest = ":install_data_manifest.txt",
    name_base = "carbon_bootstrapped_toolchain",
    package_dir = "carbon_toolchain-$(version)",
    package_file_name_base = "carbon_bootstrapped_toolchain-$(version)",
    package_variables = ":packaging_variables",
    stamp = -1,  # Allow `--stamp` builds to produce file timestamps.
    tags = ["manual"],
)

filegroup(
    name = "built_runtimes",
    srcs = [":carbon_stage1_runtimes"],
)

py_test(
    name = "llvm_symlinks_test",
    size = "small",
    srcs = ["llvm_symlinks_test.py"],
    data = [
        ":built_runtimes",
        ":install_data",
    ],
    deps = ["@bazel_tools//tools/python/runfiles"],
)
