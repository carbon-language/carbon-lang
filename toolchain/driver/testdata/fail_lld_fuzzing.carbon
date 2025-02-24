// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: --include-diagnostic-kind --fuzzing lld -- -o foo foo.o bar.o
//
// SET-CAPTURE-CONSOLE-OUTPUT
// clang-format off
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/fail_lld_fuzzing.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/fail_lld_fuzzing.carbon
// CHECK:STDERR: error: preventing fuzzing of `lld` subcommand due to external library [LLDFuzzingDisallowed]
// CHECK:STDERR:
