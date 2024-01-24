// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: [i32; 2] = (12, 24);
var b: i32 = 1;
var c: i32 = a[0];
var d: i32 = a[b];

// CHECK:STDOUT: --- array_element_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.2: type = array_type %.1, i32 [template]
// CHECK:STDOUT:   %.3: type = ptr_type [i32; 2] [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 12 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 24 [template]
// CHECK:STDOUT:   %.6: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.9: [i32; 2] = tuple_value (%.4, %.5) [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.11: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.a = %a, .b = %b, .c = %c, .d = %d} [template]
// CHECK:STDOUT:   %.loc7_14: i32 = int_literal 2 [template = constants.%.1]
// CHECK:STDOUT:   %.loc7_15: type = array_type %.loc7_14, i32 [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref [i32; 2] = var a
// CHECK:STDOUT:   %a: ref [i32; 2] = bind_name a, %a.var
// CHECK:STDOUT:   %.loc7_20: i32 = int_literal 12 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_24: i32 = int_literal 24 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_26.1: (i32, i32) = tuple_literal (%.loc7_20, %.loc7_24)
// CHECK:STDOUT:   %.loc7_26.2: i32 = int_literal 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc7_26.3: ref i32 = array_index %a.var, %.loc7_26.2
// CHECK:STDOUT:   %.loc7_26.4: init i32 = initialize_from %.loc7_20 to %.loc7_26.3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_26.5: i32 = int_literal 1 [template = constants.%.8]
// CHECK:STDOUT:   %.loc7_26.6: ref i32 = array_index %a.var, %.loc7_26.5
// CHECK:STDOUT:   %.loc7_26.7: init i32 = initialize_from %.loc7_24 to %.loc7_26.6 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_26.8: init [i32; 2] = array_init (%.loc7_26.4, %.loc7_26.7) to %a.var [template = constants.%.9]
// CHECK:STDOUT:   %.loc7_26.9: init [i32; 2] = converted %.loc7_26.1, %.loc7_26.8 [template = constants.%.9]
// CHECK:STDOUT:   assign %a.var, %.loc7_26.9
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc8: i32 = int_literal 1 [template = constants.%.10]
// CHECK:STDOUT:   assign %b.var, %.loc8
// CHECK:STDOUT:   %c.var: ref i32 = var c
// CHECK:STDOUT:   %c: ref i32 = bind_name c, %c.var
// CHECK:STDOUT:   %a.ref.loc9: ref [i32; 2] = name_ref a, %a
// CHECK:STDOUT:   %.loc9_16: i32 = int_literal 0 [template = constants.%.11]
// CHECK:STDOUT:   %.loc9_17.1: ref i32 = array_index %a.ref.loc9, %.loc9_16
// CHECK:STDOUT:   %.loc9_17.2: i32 = bind_value %.loc9_17.1
// CHECK:STDOUT:   assign %c.var, %.loc9_17.2
// CHECK:STDOUT:   %d.var: ref i32 = var d
// CHECK:STDOUT:   %d: ref i32 = bind_name d, %d.var
// CHECK:STDOUT:   %a.ref.loc10: ref [i32; 2] = name_ref a, %a
// CHECK:STDOUT:   %b.ref: ref i32 = name_ref b, %b
// CHECK:STDOUT:   %.loc10_16: i32 = bind_value %b.ref
// CHECK:STDOUT:   %.loc10_17.1: ref i32 = array_index %a.ref.loc10, %.loc10_16
// CHECK:STDOUT:   %.loc10_17.2: i32 = bind_value %.loc10_17.1
// CHECK:STDOUT:   assign %d.var, %.loc10_17.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
