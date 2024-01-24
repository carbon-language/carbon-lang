// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() {
  // CHECK:STDERR: fail_init_type_mismatch.carbon:[[@LINE+3]]:3: ERROR: Cannot implicitly convert from `f64` to `i32`.
  // CHECK:STDERR:   var x: i32 = 1.0;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  var x: i32 = 1.0;
}

// CHECK:STDOUT: --- fail_init_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: f64 = real_literal 10e-1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Main = %Main} [template]
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref i32 = var x
// CHECK:STDOUT:   %x: ref i32 = bind_name x, %x.var
// CHECK:STDOUT:   %.loc11: f64 = real_literal 10e-1 [template = constants.%.1]
// CHECK:STDOUT:   assign %x.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
