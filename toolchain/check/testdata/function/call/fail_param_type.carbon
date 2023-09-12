// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_param_type.carbon" {

fn Run(a: i32) {}
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Run

fn Main() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run(%a: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
  // CHECK:STDERR: fail_param_type.carbon:[[@LINE+6]]:6: Cannot implicitly convert from `f64` to `i32`.
  // CHECK:STDERR:   Run(1.0);
  // CHECK:STDERR:      ^
  // CHECK:STDERR: fail_param_type.carbon:[[@LINE-17]]:1: Initializing parameter 1 of function declared here.
  // CHECK:STDERR: fn Run(a: i32) {}
  // CHECK:STDERR: ^
  Run(1.0);
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_7.1: f64 = real_literal 10e-1
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_7.2: f64 = stub_reference %.loc[[@LINE-2]]_7.1
  // CHECK:STDOUT:   return
  // CHECK:STDOUT: }
}
