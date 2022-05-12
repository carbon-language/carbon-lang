# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# libprotobuf_mutator uses cmake and doesn't provide a bazel BUILD file.
# See https://github.com/google/libprotobuf-mutator/issues/91.

cc_library(
    name = "libprotobuf_mutator",
    srcs = glob(
        [
            "src/**/*.cc",
            "src/**/*.h",
            "port/protobuf.h",
        ],
        exclude = ["**/*_test.cc"],
    ),
    hdrs = ["src/libfuzzer/libfuzzer_macro.h"],
    include_prefix = "libprotobuf_mutator",
    visibility = ["//visibility:public"],
    deps = ["@com_google_protobuf//:protobuf"],
)
