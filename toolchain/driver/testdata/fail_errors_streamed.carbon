// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --phase=lex --stream-errors %s
//
// AUTOUPDATE

fn run(String program) {
  return True;

// CHECK:STDERR: fail_errors_streamed.carbon:[[@LINE+6]]:10: ERROR: Invalid digit 'a' in decimal numeric literal.
// CHECK:STDERR: var x = 3a;
// CHECK:STDERR:          ^
// CHECK:STDERR: fail_errors_streamed.carbon:[[@LINE-6]]:24: ERROR: Opening symbol without a corresponding closing symbol.
// CHECK:STDERR: fn run(String program) {
// CHECK:STDERR:                        ^
var x = 3a;
