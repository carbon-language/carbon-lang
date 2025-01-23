// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SET-CAPTURE-CONSOLE-OUTPUT
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //testing/file_test:file_test_base_test --test_arg=--file_tests=testing/file_test/testdata/capture_console_output.carbon
// TIP: To dump output, run:
// TIP:   bazel run //testing/file_test:file_test_base_test -- --dump_output --file_tests=testing/file_test/testdata/capture_console_output.carbon
// CHECK:STDERR: params.error_stream
// CHECK:STDERR: llvm::errs

// CHECK:STDOUT: 2 args: `default_args`, `capture_console_output.carbon`
// CHECK:STDOUT: params.output_stream
// CHECK:STDOUT: llvm::outs
