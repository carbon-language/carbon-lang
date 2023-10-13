// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Run() {
  // CHECK:STDERR: fail_not_callable.carbon:[[@LINE+6]]:24: ERROR: Semantics TODO: `Not a callable name`.
  // CHECK:STDERR:   var x: i32 = "hello"();
  // CHECK:STDERR:                        ^
  // CHECK:STDERR: fail_not_callable.carbon:[[@LINE+3]]:25: ERROR: Cannot implicitly convert from `String` to `i32`.
  // CHECK:STDERR:   var x: i32 = "hello"();
  // CHECK:STDERR:                         ^
  var x: i32 = "hello"();
}

// CHECK:STDOUT: file "fail_not_callable.carbon" {
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x: ref i32 = var "x"
// CHECK:STDOUT:   %.1: type = ptr_type String
// CHECK:STDOUT:   %.loc14: String = string_literal "hello"
// CHECK:STDOUT:   assign %x, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
