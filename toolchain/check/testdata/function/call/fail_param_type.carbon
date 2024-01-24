// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn G(a: i32) {}

fn F() {
  // CHECK:STDERR: fail_param_type.carbon:[[@LINE+6]]:3: ERROR: Cannot implicitly convert from `f64` to `i32`.
  // CHECK:STDERR:   G(1.0);
  // CHECK:STDERR:   ^~
  // CHECK:STDERR: fail_param_type.carbon:[[@LINE-6]]:1: Initializing parameter 1 of function declared here.
  // CHECK:STDERR: fn G(a: i32) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~
  G(1.0);
}

// CHECK:STDOUT: --- fail_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: f64 = real_literal 10e-1 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.G = %G, .F = %F} [template]
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: <function> = name_ref G, file.%G [template = file.%G]
// CHECK:STDOUT:   %.loc16_5: f64 = real_literal 10e-1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc16_4: init () = call %G.ref(<invalid>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
