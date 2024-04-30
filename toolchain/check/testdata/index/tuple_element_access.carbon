// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: (i32,) = (12,);
var b: (i32,) = a;
var c: i32 = b[0];

// CHECK:STDOUT: --- tuple_element_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 12 [template]
// CHECK:STDOUT:   %tuple: (i32,) = tuple_value (%.3) [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc7_13.1: (type,) = tuple_literal (i32)
// CHECK:STDOUT:   %.loc7_13.2: type = converted %.loc7_13.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref (i32,) = var a
// CHECK:STDOUT:   %a: ref (i32,) = bind_name a, %a.var
// CHECK:STDOUT:   %.loc8_13.1: (type,) = tuple_literal (i32)
// CHECK:STDOUT:   %.loc8_13.2: type = converted %.loc8_13.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %b.var: ref (i32,) = var b
// CHECK:STDOUT:   %b: ref (i32,) = bind_name b, %b.var
// CHECK:STDOUT:   %c.var: ref i32 = var c
// CHECK:STDOUT:   %c: ref i32 = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_18: i32 = int_literal 12 [template = constants.%.3]
// CHECK:STDOUT:   %.loc7_21.1: (i32,) = tuple_literal (%.loc7_18)
// CHECK:STDOUT:   %.loc7_21.2: init (i32,) = tuple_init (%.loc7_18) to file.%a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_22: init (i32,) = converted %.loc7_21.1, %.loc7_21.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc7_22
// CHECK:STDOUT:   %a.ref: ref (i32,) = name_ref a, file.%a
// CHECK:STDOUT:   %.loc8_17.1: ref i32 = tuple_access %a.ref, element0
// CHECK:STDOUT:   %.loc8_17.2: i32 = bind_value %.loc8_17.1
// CHECK:STDOUT:   %.loc8_17.3: init (i32,) = tuple_init (%.loc8_17.2) to file.%b.var
// CHECK:STDOUT:   %.loc8_18: init (i32,) = converted %a.ref, %.loc8_17.3
// CHECK:STDOUT:   assign file.%b.var, %.loc8_18
// CHECK:STDOUT:   %b.ref: ref (i32,) = name_ref b, file.%b
// CHECK:STDOUT:   %.loc9_16: i32 = int_literal 0 [template = constants.%.4]
// CHECK:STDOUT:   %.loc9_17.1: ref i32 = tuple_index %b.ref, %.loc9_16
// CHECK:STDOUT:   %.loc9_17.2: i32 = bind_value %.loc9_17.1
// CHECK:STDOUT:   assign file.%c.var, %.loc9_17.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
