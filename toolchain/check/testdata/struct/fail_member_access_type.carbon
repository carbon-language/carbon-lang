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
// CHECK:STDOUT:   %.2: f64 = float_literal 4 [template]
// CHECK:STDOUT:   %struct: {.a: f64} = struct_value (%.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc7: type = struct_type {.a: f64} [template = constants.%.1]
// CHECK:STDOUT:   %x.var: ref {.a: f64} = var x
// CHECK:STDOUT:   %x: ref {.a: f64} = bind_name x, %x.var
// CHECK:STDOUT:   %y.var: ref i32 = var y
// CHECK:STDOUT:   %y: ref i32 = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_26: f64 = float_literal 4 [template = constants.%.2]
// CHECK:STDOUT:   %.loc7_29.1: {.a: f64} = struct_literal (%.loc7_26)
// CHECK:STDOUT:   %.loc7_29.2: init {.a: f64} = struct_init (%.loc7_26) to file.%x.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc7_30: init {.a: f64} = converted %.loc7_29.1, %.loc7_29.2 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc7_30
// CHECK:STDOUT:   %x.ref: ref {.a: f64} = name_ref x, file.%x
// CHECK:STDOUT:   assign file.%y.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
