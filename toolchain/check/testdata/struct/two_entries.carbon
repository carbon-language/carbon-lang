// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

let v: {.a: i32, .b: i32} = {.a = 1, .b = 2};
let w: {.a: i32, .b: i32} = v;

var x: {.a: i32, .b: i32} = {.a = 1, .b = 2};
var y: {.a: i32, .b: i32} = x;

// CHECK:STDOUT: --- two_entries.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.2: type = ptr_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %struct: {.a: i32, .b: i32} = struct_value (%.3, %.4) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc7_25: type = struct_type {.a: i32, .b: i32} [template = constants.%.1]
// CHECK:STDOUT:   %.loc7_35: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc7_43: i32 = int_literal 2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_44: {.a: i32, .b: i32} = struct_literal (%.loc7_35, %.loc7_43)
// CHECK:STDOUT:   %struct: {.a: i32, .b: i32} = struct_value (%.loc7_35, %.loc7_43) [template = constants.%struct]
// CHECK:STDOUT:   %.loc7_45: {.a: i32, .b: i32} = converted %.loc7_44, %struct [template = constants.%struct]
// CHECK:STDOUT:   %v: {.a: i32, .b: i32} = bind_name v, %.loc7_45
// CHECK:STDOUT:   %.loc8: type = struct_type {.a: i32, .b: i32} [template = constants.%.1]
// CHECK:STDOUT:   %v.ref: {.a: i32, .b: i32} = name_ref v, %v
// CHECK:STDOUT:   %w: {.a: i32, .b: i32} = bind_name w, %v.ref
// CHECK:STDOUT:   %.loc10: type = struct_type {.a: i32, .b: i32} [template = constants.%.1]
// CHECK:STDOUT:   %x.var: ref {.a: i32, .b: i32} = var x
// CHECK:STDOUT:   %x: ref {.a: i32, .b: i32} = bind_name x, %x.var
// CHECK:STDOUT:   %.loc11: type = struct_type {.a: i32, .b: i32} [template = constants.%.1]
// CHECK:STDOUT:   %y.var: ref {.a: i32, .b: i32} = var y
// CHECK:STDOUT:   %y: ref {.a: i32, .b: i32} = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_35: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc10_43: i32 = int_literal 2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc10_44.1: {.a: i32, .b: i32} = struct_literal (%.loc10_35, %.loc10_43)
// CHECK:STDOUT:   %.loc10_44.2: ref i32 = struct_access file.%x.var, element0
// CHECK:STDOUT:   %.loc10_44.3: init i32 = initialize_from %.loc10_35 to %.loc10_44.2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc10_44.4: ref i32 = struct_access file.%x.var, element1
// CHECK:STDOUT:   %.loc10_44.5: init i32 = initialize_from %.loc10_43 to %.loc10_44.4 [template = constants.%.4]
// CHECK:STDOUT:   %.loc10_44.6: init {.a: i32, .b: i32} = struct_init (%.loc10_44.3, %.loc10_44.5) to file.%x.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc10_45: init {.a: i32, .b: i32} = converted %.loc10_44.1, %.loc10_44.6 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc10_45
// CHECK:STDOUT:   %x.ref: ref {.a: i32, .b: i32} = name_ref x, file.%x
// CHECK:STDOUT:   %.loc11_29.1: ref i32 = struct_access %x.ref, element0
// CHECK:STDOUT:   %.loc11_29.2: i32 = bind_value %.loc11_29.1
// CHECK:STDOUT:   %.loc11_29.3: ref i32 = struct_access file.%y.var, element0
// CHECK:STDOUT:   %.loc11_29.4: init i32 = initialize_from %.loc11_29.2 to %.loc11_29.3
// CHECK:STDOUT:   %.loc11_29.5: ref i32 = struct_access %x.ref, element1
// CHECK:STDOUT:   %.loc11_29.6: i32 = bind_value %.loc11_29.5
// CHECK:STDOUT:   %.loc11_29.7: ref i32 = struct_access file.%y.var, element1
// CHECK:STDOUT:   %.loc11_29.8: init i32 = initialize_from %.loc11_29.6 to %.loc11_29.7
// CHECK:STDOUT:   %.loc11_29.9: init {.a: i32, .b: i32} = struct_init (%.loc11_29.4, %.loc11_29.8) to file.%y.var
// CHECK:STDOUT:   %.loc11_30: init {.a: i32, .b: i32} = converted %x.ref, %.loc11_29.9
// CHECK:STDOUT:   assign file.%y.var, %.loc11_30
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
