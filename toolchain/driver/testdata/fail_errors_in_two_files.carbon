// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --phase=lex %s
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/fail_errors_in_two_files.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/fail_errors_in_two_files.carbon

// --- fail_file1.carbon

// CHECK:STDERR: fail_file1.carbon:[[@LINE+4]]:24: error: opening symbol without a corresponding closing symbol
// CHECK:STDERR: fn run(String program) {
// CHECK:STDERR:                        ^
// CHECK:STDERR:
fn run(String program) {
  return True;

// --- fail_file2.carbon

// CHECK:STDERR: fail_file2.carbon:[[@LINE+4]]:10: error: invalid digit 'a' in decimal numeric literal
// CHECK:STDERR: var x = 3a;
// CHECK:STDERR:          ^
// CHECK:STDERR:
var x = 3a;
