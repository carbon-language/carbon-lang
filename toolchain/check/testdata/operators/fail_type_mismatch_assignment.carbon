// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() {
  var a: i32 = 3;
  // CHECK:STDERR: fail_type_mismatch_assignment.carbon:[[@LINE+3]]:3: ERROR: Cannot implicitly convert from `f64` to `i32`.
  // CHECK:STDERR:   a = 5.6;
  // CHECK:STDERR:   ^~~~~~~
  a = 5.6;
}

// CHECK:STDOUT: --- fail_type_mismatch_assignment.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.2: f64 = real_literal 56e-1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Main = %Main} [template]
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref i32 = var a
// CHECK:STDOUT:   %a: ref i32 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc8: i32 = int_literal 3 [template = constants.%.1]
// CHECK:STDOUT:   assign %a.var, %.loc8
// CHECK:STDOUT:   %a.ref: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc12: f64 = real_literal 56e-1 [template = constants.%.2]
// CHECK:STDOUT:   assign %a.ref, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
