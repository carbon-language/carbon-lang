// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Run(a: i32) {}

fn Main() {
  // CHECK:STDERR: fail_param_type.carbon:[[@LINE+6]]:6: Cannot implicitly convert from `f64` to `i32`.
  // CHECK:STDERR:   Run(1.0);
  // CHECK:STDERR:      ^
  // CHECK:STDERR: fail_param_type.carbon:[[@LINE-6]]:1: Initializing parameter 1 of function declared here.
  // CHECK:STDERR: fn Run(a: i32) {}
  // CHECK:STDERR: ^
  Run(1.0);
}

// CHECK:STDOUT: file "fail_param_type.carbon" {
// CHECK:STDOUT:   %.loc7 = fn_decl @Run
// CHECK:STDOUT:   %.loc9 = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run(%a: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc16_7.1: f64 = real_literal 10e-1
// CHECK:STDOUT:   %.loc16_7.2: f64 = stub_reference %.loc16_7.1
// CHECK:STDOUT:   %.loc16_6: type = tuple_type ()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
