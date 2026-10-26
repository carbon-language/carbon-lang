// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --phase=lex --stream-errors %s
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/fail_errors_streamed.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/fail_errors_streamed.carbon

fn run(String program) {
  return True;

// CHECK:STDERR: fail_errors_streamed.carbon:[[@LINE+5]]:10: error: invalid digit 'a' in decimal numeric literal
// CHECK:STDERR:
// CHECK:STDERR: fail_errors_streamed.carbon:[[@LINE-5]]:24: error: opening symbol without a corresponding closing symbol
// CHECK:STDERR: fail_errors_streamed.carbon:[[@LINE-4]]:1: note: possibly missing `}` here
// CHECK:STDERR:
var x = 3a;
