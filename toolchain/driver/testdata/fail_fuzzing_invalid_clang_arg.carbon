// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: --include-diagnostic-kind --fuzzing compile nonexistent -- --target=mips64-a-ios
//
// SET-CAPTURE-CONSOLE-OUTPUT
// clang-format off
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/fail_fuzzing_invalid_clang_arg.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/fail_fuzzing_invalid_clang_arg.carbon
// CHECK:STDERR: error: preventing fuzzing of `compile` subcommand due to external library [ToolFuzzingDisallowed]
// CHECK:STDERR:
