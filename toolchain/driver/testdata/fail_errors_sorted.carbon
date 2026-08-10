// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --phase=lex %s
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/fail_errors_sorted.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/fail_errors_sorted.carbon

// The `}` that recovery suggests goes at the start of the line after
// `return True;`, so the `CHECK`s must live in their own split: autoupdate would
// otherwise write them at that very position, and the note would quote its own
// `CHECK` line.

// --- fail_errors_sorted.carbon

fn run(String program) {
  return True;

var x = 3a;

// --- AUTOUPDATE-SPLIT

// CHECK:STDERR: fail_errors_sorted.carbon:2:24: error: opening symbol without a corresponding closing symbol
// CHECK:STDERR: fail_errors_sorted.carbon:4:1: note: possibly missing `}` here
// CHECK:STDERR:
// CHECK:STDERR: fail_errors_sorted.carbon:5:10: error: invalid digit 'a' in decimal numeric literal
// CHECK:STDERR:
