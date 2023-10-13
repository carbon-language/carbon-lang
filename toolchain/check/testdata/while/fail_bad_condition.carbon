// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn While() {
  // CHECK:STDERR: fail_bad_condition.carbon:[[@LINE+3]]:17: ERROR: Cannot implicitly convert from `String` to `bool`.
  // CHECK:STDERR:   while ("Hello") {}
  // CHECK:STDERR:                 ^
  while ("Hello") {}
}

// CHECK:STDOUT: file "fail_bad_condition.carbon" {
// CHECK:STDOUT:   %While: <function> = fn_decl @While
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @While() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   br !while.cond
// CHECK:STDOUT:
// CHECK:STDOUT: !while.cond:
// CHECK:STDOUT:   %.1: type = ptr_type String
// CHECK:STDOUT:   %.loc11: String = string_literal "Hello"
// CHECK:STDOUT:   if <error> br !while.body else br !while.done
// CHECK:STDOUT:
// CHECK:STDOUT: !while.body:
// CHECK:STDOUT:   br !while.cond
// CHECK:STDOUT:
// CHECK:STDOUT: !while.done:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
