# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@bazel_skylib//rules:common_settings.bzl", "bool_setting", "int_setting")
load("@rules_cc//cc:defs.bzl", "cc_library")
load("//bazel:carbon_cc_toolchain_config.bzl", "carbon_cc_toolchain_suite")
load("//bazel:carbon_runtimes.bzl", "carbon_runtimes_build", "carbon_runtimes_config")
load("//bazel:make_include_copts.bzl", "make_include_copts")
load(
    "//bazel:runtimes_build_vars.bzl",
    "builtins_aarch64_srcs",
    "builtins_aarch64_textual_srcs",
    "builtins_copts",
    "builtins_i386_srcs",
    "builtins_i386_textual_srcs",
    "builtins_x86_64_srcs",
    "builtins_x86_64_textual_srcs",
    "crt_copts",
    "crtbegin_src",
    "crtend_src",
    "libc_internal_libcxx_hdrs",
    "libcxx_copts",
    "libcxx_hdrs",
    "libcxx_linux_srcs",
    "libcxx_macos_srcs",
    "libcxx_win32_srcs",
    "libcxxabi_hdrs",
    "libcxxabi_srcs",
    "libcxxabi_textual_srcs",
    "libunwind_copts",
    "libunwind_hdrs",
    "libunwind_srcs",
    "llvm_version_major",
)

_libcxx_hdrs = libcxx_hdrs + [
    "runtimes/libcxx/include/__config_site",
    "runtimes/libcxx/include/__assertion_handler",
]

package(default_visibility = ["//visibility:public"])

bool_setting(
    name = "runtimes_build",
    build_setting_default = False,
)

config_setting(
    name = "is_runtimes_build",
    flag_values = {":runtimes_build": "True"},
)

config_setting(
    name = "not_runtimes_build",
    flag_values = {":runtimes_build": "False"},
)

int_setting(
    name = "bootstrap_stage",
    build_setting_default = 1,
)

config_setting(
    name = "is_bootstrap_stage_1",
    flag_values = {":bootstrap_stage": "1"},
)

filegroup(
    name = "llvm_bins",
    srcs = glob(["llvm/bin/*"]),
)

filegroup(
    name = "clang_hdrs",
    srcs = glob(["llvm/lib/clang/{0}/include/*".format(llvm_version_major)]),
)

filegroup(
    name = "libcxx_hdrs",
    srcs = _libcxx_hdrs + libcxxabi_hdrs,
)

filegroup(
    name = "libunwind_hdrs",
    srcs = libunwind_hdrs,
)

cc_library(
    name = "builtins_internal",
    hdrs_check = "strict",
    textual_hdrs = select({
        "@platforms//cpu:aarch64": builtins_aarch64_textual_srcs,
        "@platforms//cpu:i386": builtins_i386_textual_srcs,
        "@platforms//cpu:x86_64": builtins_x86_64_textual_srcs,
        "//conditions:default": [],
    }),
)

cc_library(
    name = "builtins",
    srcs = select({
        "@platforms//cpu:aarch64": builtins_aarch64_srcs,
        "@platforms//cpu:i386": builtins_i386_srcs,
        "@platforms//cpu:x86_64": builtins_x86_64_srcs,
        "//conditions:default": [],
    }),
    copts = builtins_copts + make_include_copts([
        "runtimes/builtins",
    ]),
    hdrs_check = "strict",
    target_compatible_with = select({
        ":is_runtimes_build": [],
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
    srcs = libunwind_srcs,
    hdrs = [":libunwind_hdrs"],
    copts = libunwind_copts + [
        # We disable all warnings as upstream isn't clean with the common
        # warning flags Carbon uses by default.
        "-w",
    ],
    hdrs_check = "strict",
    includes = ["runtimes/libunwind/include"],
    linkstatic = 1,
    target_compatible_with = select({
        ":is_runtimes_build": [],
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
        ":is_runtimes_build": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    textual_hdrs = libcxxabi_textual_srcs,
)

cc_library(
    name = "libc_internal_libcxx",
    hdrs = libc_internal_libcxx_hdrs,
    hdrs_check = "strict",
    target_compatible_with = select({
        ":is_runtimes_build": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

cc_library(
    name = "libcxx",
    srcs = select({
        "@platforms//os:macos": libcxx_macos_srcs,
        "@platforms//os:windows": libcxx_win32_srcs,
        "//conditions:default": libcxx_linux_srcs,
    }) + libcxxabi_srcs,
    hdrs = [":libcxx_hdrs"],
    copts = libcxx_copts + make_include_copts([
        "runtimes/libcxx/src",
        "runtimes/libc/internal",
    ]) + [
        # We disable all warnings as upstream isn't clean with the common
        # warning flags Carbon uses by default.
        "-w",
    ],
    hdrs_check = "strict",
    includes = [
        "runtimes/libcxx/include",
        "runtimes/libcxxabi/include",
    ],
    target_compatible_with = select({
        ":is_runtimes_build": [],
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

carbon_runtimes_config(
    name = "runtimes_cfg",
    builtins_archive = ":builtins_archive",
    clang_hdrs_prefix = "llvm/lib/clang/{0}/include/".format(llvm_version_major),
    crt_copts = crt_copts,
    crtbegin_src = select({
        "@platforms//os:linux": crtbegin_src,
        "//conditions:default": None,
    }),
    crtend_src = select({
        "@platforms//os:linux": crtend_src,
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
        ":is_runtimes_build": [],
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

carbon_runtimes_build(
    name = "runtimes",
    clang_hdrs = [":clang_hdrs"],
    config = ":runtimes_cfg",
)

filegroup(
    name = "carbon_install_digest_file",
    srcs = ["install_digest.txt"],
)

filegroup(
    name = "carbon_install_marker_file",
    srcs = ["carbon_install.txt"],
)

filegroup(
    name = "carbon_busybox_file",
    srcs = ["carbon-busybox"],
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
            "@platforms//os:{}".format(os),
            "@platforms//cpu:{}".format(cpu),
        ],
    )
    for os, cpus in platforms.items()
    for cpu in cpus
]

carbon_cc_toolchain_suite(
    name = "carbon",
    all_hdrs = [
        ":clang_hdrs",
        ":libunwind_hdrs",
        ":libcxx_hdrs",
    ],
    base_files = [
        ":carbon_install_digest_file",
        ":carbon_install_marker_file",
        ":carbon_busybox_file",
        ":llvm_bins",
    ],
    clang_hdrs = [":clang_hdrs"],
    platforms = platforms,
    runtimes_cfg = ":runtimes_cfg",
)
