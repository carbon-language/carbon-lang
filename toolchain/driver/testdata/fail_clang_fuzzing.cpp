// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: --include-diagnostic-kind --fuzzing clang foo.cpp
//
// SET-CAPTURE-CONSOLE-OUTPUT
// clang-format off
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/fail_clang_fuzzing.cpp
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/fail_clang_fuzzing.cpp
// CHECK:STDERR: error: preventing fuzzing of `clang` subcommand due to library crashes [ClangFuzzingDisallowed]
// CHECK:STDERR:
