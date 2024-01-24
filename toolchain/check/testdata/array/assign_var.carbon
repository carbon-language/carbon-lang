// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: (i32, i32, i32) = (1, 2, 3);
var b: [i32; 3] = a;

// CHECK:STDOUT: --- assign_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.3: type = ptr_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.7: (i32, i32, i32) = tuple_value (%.4, %.5, %.6) [template]
// CHECK:STDOUT:   %.8: type = array_type %.6, i32 [template]
// CHECK:STDOUT:   %.9: type = ptr_type [i32; 3] [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.11: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.12: i32 = int_literal 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.a = %a, .b = %b} [template]
// CHECK:STDOUT:   %.loc7_22.1: (type, type, type) = tuple_literal (i32, i32, i32)
// CHECK:STDOUT:   %.loc7_22.2: type = converted %.loc7_22.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref (i32, i32, i32) = var a
// CHECK:STDOUT:   %a: ref (i32, i32, i32) = bind_name a, %a.var
// CHECK:STDOUT:   %.loc7_27: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_30: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_33: i32 = int_literal 3 [template = constants.%.6]
// CHECK:STDOUT:   %.loc7_34.1: (i32, i32, i32) = tuple_literal (%.loc7_27, %.loc7_30, %.loc7_33)
// CHECK:STDOUT:   %.loc7_34.2: ref i32 = tuple_access %a.var, element0
// CHECK:STDOUT:   %.loc7_34.3: init i32 = initialize_from %.loc7_27 to %.loc7_34.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_34.4: ref i32 = tuple_access %a.var, element1
// CHECK:STDOUT:   %.loc7_34.5: init i32 = initialize_from %.loc7_30 to %.loc7_34.4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_34.6: ref i32 = tuple_access %a.var, element2
// CHECK:STDOUT:   %.loc7_34.7: init i32 = initialize_from %.loc7_33 to %.loc7_34.6 [template = constants.%.6]
// CHECK:STDOUT:   %.loc7_34.8: init (i32, i32, i32) = tuple_init (%.loc7_34.3, %.loc7_34.5, %.loc7_34.7) to %a.var [template = constants.%.7]
// CHECK:STDOUT:   %.loc7_34.9: init (i32, i32, i32) = converted %.loc7_34.1, %.loc7_34.8 [template = constants.%.7]
// CHECK:STDOUT:   assign %a.var, %.loc7_34.9
// CHECK:STDOUT:   %.loc8_14: i32 = int_literal 3 [template = constants.%.6]
// CHECK:STDOUT:   %.loc8_15: type = array_type %.loc8_14, i32 [template = constants.%.8]
// CHECK:STDOUT:   %b.var: ref [i32; 3] = var b
// CHECK:STDOUT:   %b: ref [i32; 3] = bind_name b, %b.var
// CHECK:STDOUT:   %a.ref: ref (i32, i32, i32) = name_ref a, %a
// CHECK:STDOUT:   %.loc8_19.1: ref i32 = tuple_access %a.ref, element0
// CHECK:STDOUT:   %.loc8_19.2: i32 = bind_value %.loc8_19.1
// CHECK:STDOUT:   %.loc8_19.3: i32 = int_literal 0 [template = constants.%.10]
// CHECK:STDOUT:   %.loc8_19.4: ref i32 = array_index %b.var, %.loc8_19.3
// CHECK:STDOUT:   %.loc8_19.5: init i32 = initialize_from %.loc8_19.2 to %.loc8_19.4
// CHECK:STDOUT:   %.loc8_19.6: ref i32 = tuple_access %a.ref, element1
// CHECK:STDOUT:   %.loc8_19.7: i32 = bind_value %.loc8_19.6
// CHECK:STDOUT:   %.loc8_19.8: i32 = int_literal 1 [template = constants.%.11]
// CHECK:STDOUT:   %.loc8_19.9: ref i32 = array_index %b.var, %.loc8_19.8
// CHECK:STDOUT:   %.loc8_19.10: init i32 = initialize_from %.loc8_19.7 to %.loc8_19.9
// CHECK:STDOUT:   %.loc8_19.11: ref i32 = tuple_access %a.ref, element2
// CHECK:STDOUT:   %.loc8_19.12: i32 = bind_value %.loc8_19.11
// CHECK:STDOUT:   %.loc8_19.13: i32 = int_literal 2 [template = constants.%.12]
// CHECK:STDOUT:   %.loc8_19.14: ref i32 = array_index %b.var, %.loc8_19.13
// CHECK:STDOUT:   %.loc8_19.15: init i32 = initialize_from %.loc8_19.12 to %.loc8_19.14
// CHECK:STDOUT:   %.loc8_19.16: init [i32; 3] = array_init (%.loc8_19.5, %.loc8_19.10, %.loc8_19.15) to %b.var
// CHECK:STDOUT:   %.loc8_19.17: init [i32; 3] = converted %a.ref, %.loc8_19.16
// CHECK:STDOUT:   assign %b.var, %.loc8_19.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
