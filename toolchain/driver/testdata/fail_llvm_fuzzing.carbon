// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: --include-diagnostic-kind --fuzzing llvm ar -- --version
//
// SET-CAPTURE-CONSOLE-OUTPUT
// clang-format off
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/fail_llvm_fuzzing.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/fail_llvm_fuzzing.carbon
// CHECK:STDERR: error: preventing fuzzing of `llvm` subcommand due to external library [ToolFuzzingDisallowed]
// CHECK:STDERR:
