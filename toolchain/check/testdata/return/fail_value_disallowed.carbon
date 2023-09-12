// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_value_disallowed.carbon" {

fn Main() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
  // CHECK:STDERR: fail_value_disallowed.carbon:[[@LINE+6]]:11: No return expression should be provided in this context.
  // CHECK:STDERR:   return 0;
  // CHECK:STDERR:           ^
  // CHECK:STDERR: fail_value_disallowed.carbon:[[@LINE-9]]:1: There was no return type provided.
  // CHECK:STDERR: fn Main() {
  // CHECK:STDERR: ^
  return 0;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]: i32 = int_literal 0
  // CHECK:STDOUT:   return %.loc[[@LINE-2]]
  // CHECK:STDOUT: }
}
