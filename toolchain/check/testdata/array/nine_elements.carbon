// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: [i32; 9] = (1, 2, 3, 4, 5, 6, 7, 8, 9);

// CHECK:STDOUT: --- nine_elements.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 9 [template]
// CHECK:STDOUT:   %.2: type = array_type %.1, i32 [template]
// CHECK:STDOUT:   %.3: type = ptr_type [i32; 9] [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 6 [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 7 [template]
// CHECK:STDOUT:   %.11: i32 = int_literal 8 [template]
// CHECK:STDOUT:   %.12: type = tuple_type (i32, i32, i32, i32, i32, i32, i32, i32, i32) [template]
// CHECK:STDOUT:   %.13: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.14: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.15: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.16: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.17: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.18: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.19: i32 = int_literal 6 [template]
// CHECK:STDOUT:   %.20: i32 = int_literal 7 [template]
// CHECK:STDOUT:   %.21: i32 = int_literal 8 [template]
// CHECK:STDOUT:   %.22: [i32; 9] = tuple_value (%.4, %.5, %.6, %.7, %.8, %.9, %.10, %.11, %.1) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.a = %a} [template]
// CHECK:STDOUT:   %.loc7_14: i32 = int_literal 9 [template = constants.%.1]
// CHECK:STDOUT:   %.loc7_15: type = array_type %.loc7_14, i32 [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref [i32; 9] = var a
// CHECK:STDOUT:   %a: ref [i32; 9] = bind_name a, %a.var
// CHECK:STDOUT:   %.loc7_20: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_23: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_26: i32 = int_literal 3 [template = constants.%.6]
// CHECK:STDOUT:   %.loc7_29: i32 = int_literal 4 [template = constants.%.7]
// CHECK:STDOUT:   %.loc7_32: i32 = int_literal 5 [template = constants.%.8]
// CHECK:STDOUT:   %.loc7_35: i32 = int_literal 6 [template = constants.%.9]
// CHECK:STDOUT:   %.loc7_38: i32 = int_literal 7 [template = constants.%.10]
// CHECK:STDOUT:   %.loc7_41: i32 = int_literal 8 [template = constants.%.11]
// CHECK:STDOUT:   %.loc7_44: i32 = int_literal 9 [template = constants.%.1]
// CHECK:STDOUT:   %.loc7_45.1: (i32, i32, i32, i32, i32, i32, i32, i32, i32) = tuple_literal (%.loc7_20, %.loc7_23, %.loc7_26, %.loc7_29, %.loc7_32, %.loc7_35, %.loc7_38, %.loc7_41, %.loc7_44)
// CHECK:STDOUT:   %.loc7_45.2: i32 = int_literal 0 [template = constants.%.13]
// CHECK:STDOUT:   %.loc7_45.3: ref i32 = array_index %a.var, %.loc7_45.2
// CHECK:STDOUT:   %.loc7_45.4: init i32 = initialize_from %.loc7_20 to %.loc7_45.3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_45.5: i32 = int_literal 1 [template = constants.%.14]
// CHECK:STDOUT:   %.loc7_45.6: ref i32 = array_index %a.var, %.loc7_45.5
// CHECK:STDOUT:   %.loc7_45.7: init i32 = initialize_from %.loc7_23 to %.loc7_45.6 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_45.8: i32 = int_literal 2 [template = constants.%.15]
// CHECK:STDOUT:   %.loc7_45.9: ref i32 = array_index %a.var, %.loc7_45.8
// CHECK:STDOUT:   %.loc7_45.10: init i32 = initialize_from %.loc7_26 to %.loc7_45.9 [template = constants.%.6]
// CHECK:STDOUT:   %.loc7_45.11: i32 = int_literal 3 [template = constants.%.16]
// CHECK:STDOUT:   %.loc7_45.12: ref i32 = array_index %a.var, %.loc7_45.11
// CHECK:STDOUT:   %.loc7_45.13: init i32 = initialize_from %.loc7_29 to %.loc7_45.12 [template = constants.%.7]
// CHECK:STDOUT:   %.loc7_45.14: i32 = int_literal 4 [template = constants.%.17]
// CHECK:STDOUT:   %.loc7_45.15: ref i32 = array_index %a.var, %.loc7_45.14
// CHECK:STDOUT:   %.loc7_45.16: init i32 = initialize_from %.loc7_32 to %.loc7_45.15 [template = constants.%.8]
// CHECK:STDOUT:   %.loc7_45.17: i32 = int_literal 5 [template = constants.%.18]
// CHECK:STDOUT:   %.loc7_45.18: ref i32 = array_index %a.var, %.loc7_45.17
// CHECK:STDOUT:   %.loc7_45.19: init i32 = initialize_from %.loc7_35 to %.loc7_45.18 [template = constants.%.9]
// CHECK:STDOUT:   %.loc7_45.20: i32 = int_literal 6 [template = constants.%.19]
// CHECK:STDOUT:   %.loc7_45.21: ref i32 = array_index %a.var, %.loc7_45.20
// CHECK:STDOUT:   %.loc7_45.22: init i32 = initialize_from %.loc7_38 to %.loc7_45.21 [template = constants.%.10]
// CHECK:STDOUT:   %.loc7_45.23: i32 = int_literal 7 [template = constants.%.20]
// CHECK:STDOUT:   %.loc7_45.24: ref i32 = array_index %a.var, %.loc7_45.23
// CHECK:STDOUT:   %.loc7_45.25: init i32 = initialize_from %.loc7_41 to %.loc7_45.24 [template = constants.%.11]
// CHECK:STDOUT:   %.loc7_45.26: i32 = int_literal 8 [template = constants.%.21]
// CHECK:STDOUT:   %.loc7_45.27: ref i32 = array_index %a.var, %.loc7_45.26
// CHECK:STDOUT:   %.loc7_45.28: init i32 = initialize_from %.loc7_44 to %.loc7_45.27 [template = constants.%.1]
// CHECK:STDOUT:   %.loc7_45.29: init [i32; 9] = array_init (%.loc7_45.4, %.loc7_45.7, %.loc7_45.10, %.loc7_45.13, %.loc7_45.16, %.loc7_45.19, %.loc7_45.22, %.loc7_45.25, %.loc7_45.28) to %a.var [template = constants.%.22]
// CHECK:STDOUT:   %.loc7_45.30: init [i32; 9] = converted %.loc7_45.1, %.loc7_45.29 [template = constants.%.22]
// CHECK:STDOUT:   assign %a.var, %.loc7_45.30
// CHECK:STDOUT: }
// CHECK:STDOUT:
