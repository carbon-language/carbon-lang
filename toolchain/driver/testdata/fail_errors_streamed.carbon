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

// CHECK:STDERR:        ╭─┤ fail_errors_streamed.carbon:[[@LINE+17]]:10
// CHECK:STDERR: ->     │ var x = 3a;
// CHECK:STDERR:        ·          ┳
// CHECK:STDERR:        ·          ┗━━┫ error: invalid digit 'a' in decimal numeric literal
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_errors_streamed.carbon:[[@LINE-9]]:24
// CHECK:STDERR: ->     │ fn run(String program) {
// CHECK:STDERR:        ·                        ┳
// CHECK:STDERR:        ·                        ┗━━┫ error: opening symbol without a corresponding
// CHECK:STDERR:        ·                           ┃        closing symbol
// CHECK:STDERR:        │   return True;
// CHECK:STDERR:        │
// CHECK:STDERR:        · ┬
// CHECK:STDERR:        · ╰──┤ possibly missing `}` here
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var x = 3a;
