// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: {.a: i32, .b: (i32,)} = {.a = 1, .b = (2,)};
var y: {.a: i32, .b: (i32,)} = x;

// CHECK:STDOUT: --- tuple_as_element.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: i32, .b: (i32,)} [template]
// CHECK:STDOUT:   %.4: type = ptr_type {.a: i32, .b: (i32,)} [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.7: (i32,) = tuple_value (%.6) [template]
// CHECK:STDOUT:   %.8: {.a: i32, .b: (i32,)} = struct_value (%.5, %.7) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.x = %x, .y = %y} [template]
// CHECK:STDOUT:   %.loc7_27.1: (type,) = tuple_literal (i32)
// CHECK:STDOUT:   %.loc7_27.2: type = converted %.loc7_27.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc7_28: type = struct_type {.a: i32, .b: (i32,)} [template = constants.%.3]
// CHECK:STDOUT:   %x.var: ref {.a: i32, .b: (i32,)} = var x
// CHECK:STDOUT:   %x: ref {.a: i32, .b: (i32,)} = bind_name x, %x.var
// CHECK:STDOUT:   %.loc7_38: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_47: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc7_49.1: (i32,) = tuple_literal (%.loc7_47)
// CHECK:STDOUT:   %.loc7_50.1: {.a: i32, .b: (i32,)} = struct_literal (%.loc7_38, %.loc7_49.1)
// CHECK:STDOUT:   %.loc7_50.2: ref i32 = struct_access %x.var, element0
// CHECK:STDOUT:   %.loc7_50.3: init i32 = initialize_from %.loc7_38 to %.loc7_50.2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_50.4: ref (i32,) = struct_access %x.var, element1
// CHECK:STDOUT:   %.loc7_49.2: init (i32,) = tuple_init (%.loc7_47) to %.loc7_50.4 [template = constants.%.7]
// CHECK:STDOUT:   %.loc7_49.3: init (i32,) = converted %.loc7_49.1, %.loc7_49.2 [template = constants.%.7]
// CHECK:STDOUT:   %.loc7_50.5: init (i32,) = initialize_from %.loc7_49.3 to %.loc7_50.4 [template = constants.%.7]
// CHECK:STDOUT:   %.loc7_50.6: init {.a: i32, .b: (i32,)} = struct_init (%.loc7_50.3, %.loc7_50.5) to %x.var [template = constants.%.8]
// CHECK:STDOUT:   %.loc7_50.7: init {.a: i32, .b: (i32,)} = converted %.loc7_50.1, %.loc7_50.6 [template = constants.%.8]
// CHECK:STDOUT:   assign %x.var, %.loc7_50.7
// CHECK:STDOUT:   %.loc8_27.1: (type,) = tuple_literal (i32)
// CHECK:STDOUT:   %.loc8_27.2: type = converted %.loc8_27.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_28: type = struct_type {.a: i32, .b: (i32,)} [template = constants.%.3]
// CHECK:STDOUT:   %y.var: ref {.a: i32, .b: (i32,)} = var y
// CHECK:STDOUT:   %y: ref {.a: i32, .b: (i32,)} = bind_name y, %y.var
// CHECK:STDOUT:   %x.ref: ref {.a: i32, .b: (i32,)} = name_ref x, %x
// CHECK:STDOUT:   %.loc8_32.1: ref i32 = struct_access %x.ref, element0
// CHECK:STDOUT:   %.loc8_32.2: i32 = bind_value %.loc8_32.1
// CHECK:STDOUT:   %.loc8_32.3: ref i32 = struct_access %y.var, element0
// CHECK:STDOUT:   %.loc8_32.4: init i32 = initialize_from %.loc8_32.2 to %.loc8_32.3
// CHECK:STDOUT:   %.loc8_32.5: ref (i32,) = struct_access %x.ref, element1
// CHECK:STDOUT:   %.loc8_32.6: ref i32 = tuple_access %.loc8_32.5, element0
// CHECK:STDOUT:   %.loc8_32.7: i32 = bind_value %.loc8_32.6
// CHECK:STDOUT:   %.loc8_32.8: ref (i32,) = struct_access %y.var, element1
// CHECK:STDOUT:   %.loc8_32.9: init (i32,) = tuple_init (%.loc8_32.7) to %.loc8_32.8
// CHECK:STDOUT:   %.loc8_32.10: init (i32,) = converted %.loc8_32.5, %.loc8_32.9
// CHECK:STDOUT:   %.loc8_32.11: init (i32,) = initialize_from %.loc8_32.10 to %.loc8_32.8
// CHECK:STDOUT:   %.loc8_32.12: init {.a: i32, .b: (i32,)} = struct_init (%.loc8_32.4, %.loc8_32.11) to %y.var
// CHECK:STDOUT:   %.loc8_32.13: init {.a: i32, .b: (i32,)} = converted %x.ref, %.loc8_32.12
// CHECK:STDOUT:   assign %y.var, %.loc8_32.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
