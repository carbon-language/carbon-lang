# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_library")
load("//bazel:carbon_runtimes.bzl", "carbon_runtimes")
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

package(default_visibility = ["//visibility:public"])

# Clang looks in a target-triple subdirectory of its resource directory for the
# builtins and CRT files. Triples capture a large range of settings and aren't
# even consistent structurally between platforms, so we expect to have roughly
# one setting per supported target triple. By convention, we name them after the
# target triple.
#
# TODO: Add constraints and other settings so we can select the correct target
# triple for different `libc`s.
#
# TODO: Add other OS and CPU triples.
config_setting(
    name = "x86_64-unknown-linux-gnu",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
)

config_setting(
    name = "aarch64-unknown-linux-gnu",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:aarch64",
    ],
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
        "builtins",
    ]),
    hdrs_check = "strict",
    deps = [":builtins_internal"],
)

filegroup(
    name = "builtins_archive",
    srcs = [":builtins"],
    output_group = "archive",
)

filegroup(
    name = "libunwind_hdrs",
    srcs = libunwind_hdrs,
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
    includes = ["libunwind/include"],
    linkstatic = 1,
)

filegroup(
    name = "libunwind_archive",
    srcs = [":libunwind"],
    output_group = "archive",
)

cc_library(
    name = "libcxxabi_internal",
    hdrs_check = "strict",
    textual_hdrs = libcxxabi_textual_srcs,
)

cc_library(
    name = "libc_internal_libcxx",
    hdrs = libc_internal_libcxx_hdrs,
    hdrs_check = "strict",
)

filegroup(
    name = "libcxx_hdrs",
    srcs = [
        "libcxx/include/__assertion_handler",
        "libcxx/include/__config_site",
    ] + libcxx_hdrs + libcxxabi_hdrs,
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
        "libcxx/src",
        "libc/internal",
    ]) + [
        # We disable all warnings as upstream isn't clean with the common
        # warning flags Carbon uses by default.
        "-w",
    ],
    hdrs_check = "strict",
    includes = [
        "libcxx/include",
        "libcxxabi/include",
    ],
    linkstatic = 1,
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
    name = "carbon_runtimes",
    builtins_archive = ":builtins_archive",
    clang_hdrs = [
        "//llvm/lib/clang/{0}:clang_hdrs".format(llvm_version_major),
    ],
    crt_copts = crt_copts,
    crtbegin_src = select({
        "@platforms//os:linux": crtbegin_src,
        "//conditions:default": None,
    }),
    crtend_src = select({
        "@platforms//os:linux": crtend_src,
        "//conditions:default": None,
    }),
    libcxx_archive = ":libcxx_archive",
    libunwind_archive = ":libunwind_archive",
    target_triple = select({
        ":aarch64-unknown-linux-gnu": "aarch64-unknown-linux-gnu",
        ":x86_64-unknown-linux-gnu": "x86_64-unknown-linux-gnu",
        "//conditions:default": "",
    }),
)
