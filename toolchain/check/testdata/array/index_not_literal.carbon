// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: [i32; 3] = (1, 2, 3);
var b: i32 = a[{.index = 2}.index];

// CHECK:STDOUT: --- index_not_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.2: type = array_type %.1, i32 [template]
// CHECK:STDOUT:   %.3: type = ptr_type [i32; 3] [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.6: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.10: [i32; 3] = tuple_value (%.4, %.5, %.1) [template]
// CHECK:STDOUT:   %.11: type = struct_type {.index: i32} [template]
// CHECK:STDOUT:   %.12: {.index: i32} = struct_value (%.5) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.a = %a, .b = %b} [template]
// CHECK:STDOUT:   %.loc7_14: i32 = int_literal 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc7_15: type = array_type %.loc7_14, i32 [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref [i32; 3] = var a
// CHECK:STDOUT:   %a: ref [i32; 3] = bind_name a, %a.var
// CHECK:STDOUT:   %.loc7_20: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_23: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_26: i32 = int_literal 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc7_27.1: (i32, i32, i32) = tuple_literal (%.loc7_20, %.loc7_23, %.loc7_26)
// CHECK:STDOUT:   %.loc7_27.2: i32 = int_literal 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc7_27.3: ref i32 = array_index %a.var, %.loc7_27.2
// CHECK:STDOUT:   %.loc7_27.4: init i32 = initialize_from %.loc7_20 to %.loc7_27.3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_27.5: i32 = int_literal 1 [template = constants.%.8]
// CHECK:STDOUT:   %.loc7_27.6: ref i32 = array_index %a.var, %.loc7_27.5
// CHECK:STDOUT:   %.loc7_27.7: init i32 = initialize_from %.loc7_23 to %.loc7_27.6 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_27.8: i32 = int_literal 2 [template = constants.%.9]
// CHECK:STDOUT:   %.loc7_27.9: ref i32 = array_index %a.var, %.loc7_27.8
// CHECK:STDOUT:   %.loc7_27.10: init i32 = initialize_from %.loc7_26 to %.loc7_27.9 [template = constants.%.1]
// CHECK:STDOUT:   %.loc7_27.11: init [i32; 3] = array_init (%.loc7_27.4, %.loc7_27.7, %.loc7_27.10) to %a.var [template = constants.%.10]
// CHECK:STDOUT:   %.loc7_27.12: init [i32; 3] = converted %.loc7_27.1, %.loc7_27.11 [template = constants.%.10]
// CHECK:STDOUT:   assign %a.var, %.loc7_27.12
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %a.ref: ref [i32; 3] = name_ref a, %a
// CHECK:STDOUT:   %.loc8_26: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc8_27.1: {.index: i32} = struct_literal (%.loc8_26)
// CHECK:STDOUT:   %.loc8_27.2: {.index: i32} = struct_value (%.loc8_26) [template = constants.%.12]
// CHECK:STDOUT:   %.loc8_27.3: {.index: i32} = converted %.loc8_27.1, %.loc8_27.2 [template = constants.%.12]
// CHECK:STDOUT:   %.loc8_28: i32 = struct_access %.loc8_27.3, element0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc8_34.1: ref i32 = array_index %a.ref, %.loc8_28
// CHECK:STDOUT:   %.loc8_34.2: i32 = bind_value %.loc8_34.1
// CHECK:STDOUT:   assign %b.var, %.loc8_34.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
