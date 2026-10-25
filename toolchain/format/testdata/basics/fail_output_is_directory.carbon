// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: --include-diagnostic-kind format --output=/ %s
//
// NOAUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/basics/fail_output_is_directory.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/basics/fail_output_is_directory.carbon

// Writing the formatted output fails because `/` is a directory. Don't depend
// on a specific filesystem error.
// CHECK:STDERR: error: could not open output file `/`: {{.*}} [FormatOutputFileOpenError]
// CHECK:STDERR:

fn F() {}
