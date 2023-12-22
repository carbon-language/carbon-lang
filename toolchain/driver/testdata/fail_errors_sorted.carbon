// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --phase=lex %s
//
// AUTOUPDATE

// CHECK:STDERR: fail_errors_sorted.carbon:[[@LINE+3]]:24: ERROR: Opening symbol without a corresponding closing symbol.
// CHECK:STDERR: fn run(String program) {
// CHECK:STDERR:                        ^
fn run(String program) {
  return True;

// CHECK:STDERR: fail_errors_sorted.carbon:[[@LINE+3]]:10: ERROR: Invalid digit 'a' in decimal numeric literal.
// CHECK:STDERR: var x = 3a;
// CHECK:STDERR:          ^
var x = 3a;
