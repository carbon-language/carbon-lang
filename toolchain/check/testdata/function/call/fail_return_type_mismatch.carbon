// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_return_type_mismatch.carbon" {

fn Foo() -> f64 { return 1.0; }
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Foo

fn Run() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Run
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo() -> f64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc[[@LINE-9]]: f64 = real_literal 10e-1
// CHECK:STDOUT:   return %.loc[[@LINE-10]]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x: ref i32 = var "x"
  // CHECK:STDERR: fail_return_type_mismatch.carbon:[[@LINE+3]]:21: Cannot implicitly convert from `f64` to `i32`.
  // CHECK:STDERR:   var x: i32 = Foo();
  // CHECK:STDERR:                     ^
  var x: i32 = Foo();
  // CHECK:STDOUT:   %.loc[[@LINE-1]]: init f64 = call @Foo()
  // CHECK:STDOUT:   assign %x, <error>
  // CHECK:STDOUT:   return
  // CHECK:STDOUT: }
}
