// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: {.a: f64} = {.a = 4.0};
// CHECK:STDERR: fail_member_access_type.carbon:[[@LINE+3]]:14: ERROR: Type `{.a: f64}` does not have a member `b`.
// CHECK:STDERR: var y: i32 = x.b;
// CHECK:STDERR:              ^~~
var y: i32 = x.b;

// CHECK:STDOUT: --- fail_member_access_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {.a: f64} [template]
// CHECK:STDOUT:   %.2: f64 = real_literal 40e-1 [template]
// CHECK:STDOUT:   %.3: {.a: f64} = struct_value (%.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.x = %x, .y = %y} [template]
// CHECK:STDOUT:   %.loc7_16: type = struct_type {.a: f64} [template = constants.%.1]
// CHECK:STDOUT:   %x.var: ref {.a: f64} = var x
// CHECK:STDOUT:   %x: ref {.a: f64} = bind_name x, %x.var
// CHECK:STDOUT:   %.loc7_26: f64 = real_literal 40e-1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc7_29.1: {.a: f64} = struct_literal (%.loc7_26)
// CHECK:STDOUT:   %.loc7_29.2: init {.a: f64} = struct_init (%.loc7_26) to %x.var [template = constants.%.3]
// CHECK:STDOUT:   %.loc7_29.3: init {.a: f64} = converted %.loc7_29.1, %.loc7_29.2 [template = constants.%.3]
// CHECK:STDOUT:   assign %x.var, %.loc7_29.3
// CHECK:STDOUT:   %y.var: ref i32 = var y
// CHECK:STDOUT:   %y: ref i32 = bind_name y, %y.var
// CHECK:STDOUT:   %x.ref: ref {.a: f64} = name_ref x, %x
// CHECK:STDOUT:   assign %y.var, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
