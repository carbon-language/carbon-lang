// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn G(a: i32) {}

fn F() {
  // CHECK:STDERR: fail_param_type.carbon:[[@LINE+6]]:4: ERROR: Cannot implicitly convert from `f64` to `i32`.
  // CHECK:STDERR:   G(1.0);
  // CHECK:STDERR:    ^
  // CHECK:STDERR: fail_param_type.carbon:[[@LINE-6]]:1: Initializing parameter 1 of function declared here.
  // CHECK:STDERR: fn G(a: i32) {}
  // CHECK:STDERR: ^
  G(1.0);
}

// CHECK:STDOUT: file "fail_param_type.carbon" {
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: <function> = name_reference "G", package.%G
// CHECK:STDOUT:   %.loc16_5: f64 = real_literal 10e-1
// CHECK:STDOUT:   %.loc16_4: type = tuple_type ()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
