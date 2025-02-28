# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("//bazel/version:rules.bzl", "expand_version_build_info")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "array_stack",
    hdrs = ["array_stack.h"],
    deps = [
        ":check",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "array_stack_test",
    size = "small",
    srcs = ["array_stack_test.cpp"],
    deps = [
        ":array_stack",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "bazel_working_dir",
    hdrs = ["bazel_working_dir.h"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "command_line",
    srcs = ["command_line.cpp"],
    hdrs = ["command_line.h"],
    deps = [
        ":check",
        ":error",
        ":ostream",
        ":raw_string_ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "command_line_test",
    size = "small",
    srcs = ["command_line_test.cpp"],
    deps = [
        ":command_line",
        ":error_test_helpers",
        ":raw_string_ostream",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "check",
    srcs = [
        "check_internal.cpp",
        "check_internal.h",
    ],
    hdrs = ["check.h"],
    deps = [
        ":ostream",
        ":template_string",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "check_test",
    size = "small",
    srcs = ["check_test.cpp"],
    deps = [
        ":check",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "enum_base",
    hdrs = ["enum_base.h"],
    deps = [
        "//common:ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "enum_base_test_def",
    testonly = 1,
    textual_hdrs = ["enum_base_test.def"],
)

cc_test(
    name = "enum_base_test",
    size = "small",
    srcs = ["enum_base_test.cpp"],
    deps = [
        ":enum_base",
        ":enum_base_test_def",
        ":raw_string_ostream",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "error",
    hdrs = ["error.h"],
    deps = [
        ":check",
        ":ostream",
        ":raw_string_ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "error_test_helpers",
    testonly = 1,
    hdrs = ["error_test_helpers.h"],
    deps = [
        ":error",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "error_test",
    size = "small",
    srcs = ["error_test.cpp"],
    deps = [
        ":error",
        ":error_test_helpers",
        ":raw_string_ostream",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "exe_path",
    srcs = ["exe_path.cpp"],
    hdrs = ["exe_path.h"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "exe_path_test",
    size = "small",
    srcs = ["exe_path_test.cpp"],
    deps = [
        ":exe_path",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "hashing",
    srcs = ["hashing.cpp"],
    hdrs = ["hashing.h"],
    deps = [
        ":check",
        ":ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "hashing_test",
    size = "small",
    srcs = ["hashing_test.cpp"],
    deps = [
        ":hashing",
        ":raw_string_ostream",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_binary(
    name = "hashing_benchmark",
    testonly = 1,
    srcs = ["hashing_benchmark.cpp"],
    deps = [
        ":check",
        ":hashing",
        "//testing/base:benchmark_main",
        "@abseil-cpp//absl/hash",
        "@abseil-cpp//absl/random",
        "@google_benchmark//:benchmark",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "hashtable_key_context",
    hdrs = ["hashtable_key_context.h"],
    deps = [
        ":hashing",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "hashtable_key_context_test",
    size = "small",
    srcs = ["hashtable_key_context_test.cpp"],
    deps = [
        ":hashtable_key_context",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "indirect_value",
    hdrs = ["indirect_value.h"],
)

cc_test(
    name = "indirect_value_test",
    size = "small",
    srcs = ["indirect_value_test.cpp"],
    deps = [
        ":indirect_value",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "init_llvm",
    srcs = ["init_llvm.cpp"],
    hdrs = ["init_llvm.h"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)

# Link against this to cause `:init_llvm` to pull in all LLVM targets.
#
# Be careful when depending on this: it pulls in several hundred megabytes of
# LLVM binary size in -c fastbuild. This should only be depended on by a
# `cc_binary` or `cc_test` target, never a `cc_library`.
cc_library(
    name = "all_llvm_targets",
    srcs = ["all_llvm_targets.cpp"],
    deps = [
        ":init_llvm",
        "@llvm-project//llvm:AllTargetsAsmParsers",
        "@llvm-project//llvm:AllTargetsCodeGens",
        "@llvm-project//llvm:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "map",
    hdrs = ["map.h"],
    deps = [
        ":check",
        ":hashtable_key_context",
        ":raw_hashtable",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "map_test",
    size = "small",
    srcs = ["map_test.cpp"],
    deps = [
        ":map",
        ":raw_hashtable_test_helpers",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
    ],
)

cc_binary(
    name = "map_benchmark",
    testonly = 1,
    srcs = ["map_benchmark.cpp"],
    deps = [
        ":map",
        ":raw_hashtable_benchmark_helpers",
        "//testing/base:benchmark_main",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/random",
        "@boost_unordered",
        "@google_benchmark//:benchmark",
        "@llvm-project//llvm:Support",
    ],
)

sh_test(
    name = "map_benchmark_test",
    # The benchmark allocates a large amount of memory.
    size = "enormous",
    # We configure the test to run somewhat quickly.
    timeout = "moderate",
    srcs = [":map_benchmark"],
    args = [
        "--benchmark_min_time=1x",
        # The `$$` is repeated for Bazel escaping of `$`.
        "--benchmark_filter=^[^/]*/[1-9][0-9]{0,3}(/[0-9]+)?$$",
    ],
)

cc_library(
    name = "ostream",
    hdrs = ["ostream.h"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "raw_hashtable",
    srcs = ["raw_hashtable.cpp"],
    hdrs = ["raw_hashtable.h"],
    deps = [
        ":check",
        ":hashing",
        ":hashtable_key_context",
        ":raw_hashtable_metadata_group",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "raw_hashtable_metadata_group",
    srcs = ["raw_hashtable_metadata_group.cpp"],
    hdrs = ["raw_hashtable_metadata_group.h"],
    deps = [
        ":check",
        ":ostream",
        "@llvm-project//llvm:Support",
    ],
)

cc_binary(
    name = "raw_hashtable_metadata_group_benchmark",
    testonly = 1,
    srcs = ["raw_hashtable_metadata_group_benchmark.cpp"],
    deps = [
        ":raw_hashtable_metadata_group",
        "//testing/base:benchmark_main",
        "@abseil-cpp//absl/random",
        "@google_benchmark//:benchmark",
        "@llvm-project//llvm:Support",
    ],
)

sh_test(
    name = "raw_hashtable_metadata_group_benchmark_test",
    size = "small",
    srcs = ["raw_hashtable_metadata_group_benchmark"],
    args = [
        "--benchmark_min_time=1x",
    ],
)

cc_library(
    name = "raw_hashtable_benchmark_helpers",
    testonly = 1,
    srcs = ["raw_hashtable_benchmark_helpers.cpp"],
    hdrs = ["raw_hashtable_benchmark_helpers.h"],
    copts = [
        "-O2",  # Always optimize to make testing benchmarks faster.
    ],
    deps = [
        ":check",
        ":hashing",
        ":raw_hashtable",
        ":set",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/hash",
        "@abseil-cpp//absl/random",
        "@google_benchmark//:benchmark",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "raw_hashtable_test_helpers",
    testonly = 1,
    hdrs = ["raw_hashtable_test_helpers.h"],
    deps = [
        ":check",
        ":hashing",
        ":hashtable_key_context",
        ":ostream",
    ],
)

cc_library(
    name = "raw_string_ostream",
    hdrs = ["raw_string_ostream.h"],
    deps = [
        ":check",
        ":ostream",
    ],
)

cc_test(
    name = "raw_string_ostream_test",
    size = "small",
    srcs = ["raw_string_ostream_test.cpp"],
    deps = [
        ":raw_string_ostream",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "set",
    hdrs = ["set.h"],
    deps = [
        ":check",
        ":hashtable_key_context",
        ":raw_hashtable",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "set_test",
    size = "small",
    srcs = ["set_test.cpp"],
    deps = [
        ":raw_hashtable_test_helpers",
        ":set",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
    ],
)

cc_binary(
    name = "set_benchmark",
    testonly = 1,
    srcs = ["set_benchmark.cpp"],
    deps = [
        ":raw_hashtable_benchmark_helpers",
        ":set",
        "//testing/base:benchmark_main",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@google_benchmark//:benchmark",
        "@llvm-project//llvm:Support",
    ],
)

sh_test(
    name = "set_benchmark_test",
    # The benchmark allocates a large amount of memory.
    size = "enormous",
    # We configure the test to run somewhat quickly.
    timeout = "moderate",
    srcs = [":set_benchmark"],
    args = [
        "--benchmark_min_time=1x",
        # The `$$` is repeated for Bazel escaping of `$`.
        "--benchmark_filter=^[^/]*/[1-9][0-9]{0,3}(/[0-9]+)?$$",
    ],
)

cc_library(
    name = "string_helpers",
    srcs = ["string_helpers.cpp"],
    hdrs = ["string_helpers.h"],
    deps = [
        ":check",
        ":error",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "string_helpers_test",
    size = "small",
    srcs = ["string_helpers_test.cpp"],
    deps = [
        ":string_helpers",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "struct_reflection",
    hdrs = ["struct_reflection.h"],
)

cc_test(
    name = "struct_reflection_test",
    size = "small",
    srcs = ["struct_reflection_test.cpp"],
    deps = [
        ":struct_reflection",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "template_string",
    hdrs = ["template_string.h"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "template_string_test",
    size = "small",
    srcs = ["template_string_test.cpp"],
    deps = [
        ":template_string",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "variant_helpers",
    hdrs = ["variant_helpers.h"],
    deps = [
        ":error",
        "@llvm-project//llvm:Support",
    ],
)

# The base version source file only uses non-stamped parts of the version
# information so we expand it once here without any stamping.
expand_version_build_info(
    name = "version_cpp_gen",
    out = "version.cpp",
    stamp = 0,
    template = "version.tmpl.cpp",
)

# Build a nostamp version of the stamp source, but mark its definitions as weak.
# We'll include this in the library to satisfy definitions of library and test
# users, but still allow binaries that want full build stamping to depend on the
# stamp library below to override with strong, stamped definitions.
expand_version_build_info(
    name = "version_nostamp_cpp_gen",
    out = "version_nostamp.cpp",
    stamp = 0,
    substitutions = {"MAKE_WEAK": "1"},
    template = "version_stamp.tmpl.cpp",
)

# Provides APIs for accessing Carbon version information.
#
# These provide full access to the major, minor, and patch version. It also
# provides an API for querying version strings that may contain detailed build
# information such as the commit SHA.
#
# By default, this provides the API and an *unstamped* implementations of
# version strings. As a consequence, depending on this library doesn't introduce
# any dependency on the commit SHA or loss of build caching.
#
# Targets that want full build info stamping in the data produced by these APIs
# should additionally depend on `:version_stamp` below -- the data these APIs
# return will be overridden in any binaries depending on that rule with the
# fully stamped details.
cc_library(
    name = "version",
    srcs = [
        "version.cpp",
        "version_nostamp.cpp",
    ],
    hdrs = ["version.h"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)

# Generate the fully stamped sourcefile if stamping is enabled in the build.
expand_version_build_info(
    name = "version_stamp_cpp_gen",
    out = "version_stamp.cpp",
    template = "version_stamp.tmpl.cpp",
)

# Depend on this library to enable fully-stamped build information in the
# version API provided by `:version`. This doesn't provide the API, it injects
# an override of stamped versions of the data.
#
# Note that depending on this will significantly reduce build caching with
# `--stamp` builds. It should be used sparingly, typically in user-facing
# binaries or systems that need to render a maximally detailed version string
# with build information stamped into it.
cc_library(
    name = "version_stamp",
    srcs = ["version_stamp.cpp"],
    deps = [
        ":version",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "vlog",
    hdrs = ["vlog.h"],
    deps = [
        ":ostream",
        ":template_string",
        "@llvm-project//llvm:Support",
    ],
)

cc_test(
    name = "vlog_test",
    size = "small",
    srcs = ["vlog_test.cpp"],
    deps = [
        ":raw_string_ostream",
        ":vlog",
        "//testing/base:gtest_main",
        "@googletest//:gtest",
    ],
)
