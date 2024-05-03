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
// CHECK:STDOUT:   %Main: type = fn_type @Main [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Main = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.3: f64 = float_literal 5.6000000000000005 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Main.decl: Main = fn_decl @Main [template = constants.%struct] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref i32 = var a
// CHECK:STDOUT:   %a: ref i32 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc8: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   assign %a.var, %.loc8
// CHECK:STDOUT:   %a.ref: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc12: f64 = float_literal 5.6000000000000005 [template = constants.%.3]
// CHECK:STDOUT:   assign %a.ref, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
