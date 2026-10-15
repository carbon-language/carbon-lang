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

// The `}` that recovery suggests goes at the start of the line after
// `return True;`, so the `CHECK`s must live in their own split: autoupdate would
// otherwise write them at that very position, and the note would quote its own
// `CHECK` line.

// --- fail_file1.carbon

fn run(String program) {
  return True;

// --- fail_file2.carbon

var x = 3a;

// --- AUTOUPDATE-SPLIT

// CHECK:STDERR: fail_file1.carbon:2:24: error: opening symbol without a corresponding closing symbol
// CHECK:STDERR: fail_file1.carbon:4:1: note: possibly missing `}` here
// CHECK:STDERR:
// CHECK:STDERR: fail_file2.carbon:2:10: error: invalid digit 'a' in decimal numeric literal
// CHECK:STDERR:
