// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(a: i32) {}

fn G() {
  // CHECK:STDERR: fail_param_type.carbon:[[@LINE+6]]:4: ERROR: Cannot implicitly convert from `f64` to `i32`.
  // CHECK:STDERR:   F(1.0);
  // CHECK:STDERR:    ^
  // CHECK:STDERR: fail_param_type.carbon:[[@LINE-6]]:1: Initializing parameter 1 of function declared here.
  // CHECK:STDERR: fn F(a: i32) {}
  // CHECK:STDERR: ^
  F(1.0);
}

// CHECK:STDOUT: file "fail_param_type.carbon" {
// CHECK:STDOUT:   %F = fn_decl @F
// CHECK:STDOUT:   %G = fn_decl @G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc16_5: f64 = real_literal 10e-1
// CHECK:STDOUT:   %.loc16_4: type = tuple_type ()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
