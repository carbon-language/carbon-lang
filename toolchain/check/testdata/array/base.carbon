// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: [i32; 1] = (1,);
var b: [f64; 2] = (11.1, 2.2,);
var c: [(); 5] = ((), (), (), (), (),);

// CHECK:STDOUT: --- base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.2: type = array_type %.1, i32 [template]
// CHECK:STDOUT:   %.3: type = ptr_type [i32; 1] [template]
// CHECK:STDOUT:   %.4: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.6: [i32; 1] = tuple_value (%.1) [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.8: type = array_type %.7, f64 [template]
// CHECK:STDOUT:   %.9: type = ptr_type [f64; 2] [template]
// CHECK:STDOUT:   %.10: f64 = real_literal 111e-1 [template]
// CHECK:STDOUT:   %.11: f64 = real_literal 22e-1 [template]
// CHECK:STDOUT:   %.12: type = tuple_type (f64, f64) [template]
// CHECK:STDOUT:   %.13: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.14: [f64; 2] = tuple_value (%.10, %.11) [template]
// CHECK:STDOUT:   %.15: type = tuple_type () [template]
// CHECK:STDOUT:   %.16: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.17: type = array_type %.16, () [template]
// CHECK:STDOUT:   %.18: type = ptr_type [(); 5] [template]
// CHECK:STDOUT:   %.19: type = tuple_type ((), (), (), (), ()) [template]
// CHECK:STDOUT:   %.20: () = tuple_value () [template]
// CHECK:STDOUT:   %.21: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.22: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.23: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.24: [(); 5] = tuple_value (%.20, %.20, %.20, %.20, %.20) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.a = %a, .b = %b, .c = %c} [template]
// CHECK:STDOUT:   %.loc7_14: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc7_15: type = array_type %.loc7_14, i32 [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref [i32; 1] = var a
// CHECK:STDOUT:   %a: ref [i32; 1] = bind_name a, %a.var
// CHECK:STDOUT:   %.loc7_20: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc7_22.1: (i32,) = tuple_literal (%.loc7_20)
// CHECK:STDOUT:   %.loc7_22.2: i32 = int_literal 0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_22.3: ref i32 = array_index %a.var, %.loc7_22.2
// CHECK:STDOUT:   %.loc7_22.4: init i32 = initialize_from %.loc7_20 to %.loc7_22.3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc7_22.5: init [i32; 1] = array_init (%.loc7_22.4) to %a.var [template = constants.%.6]
// CHECK:STDOUT:   %.loc7_22.6: init [i32; 1] = converted %.loc7_22.1, %.loc7_22.5 [template = constants.%.6]
// CHECK:STDOUT:   assign %a.var, %.loc7_22.6
// CHECK:STDOUT:   %.loc8_14: i32 = int_literal 2 [template = constants.%.7]
// CHECK:STDOUT:   %.loc8_15: type = array_type %.loc8_14, f64 [template = constants.%.8]
// CHECK:STDOUT:   %b.var: ref [f64; 2] = var b
// CHECK:STDOUT:   %b: ref [f64; 2] = bind_name b, %b.var
// CHECK:STDOUT:   %.loc8_20: f64 = real_literal 111e-1 [template = constants.%.10]
// CHECK:STDOUT:   %.loc8_26: f64 = real_literal 22e-1 [template = constants.%.11]
// CHECK:STDOUT:   %.loc8_30.1: (f64, f64) = tuple_literal (%.loc8_20, %.loc8_26)
// CHECK:STDOUT:   %.loc8_30.2: i32 = int_literal 0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc8_30.3: ref f64 = array_index %b.var, %.loc8_30.2
// CHECK:STDOUT:   %.loc8_30.4: init f64 = initialize_from %.loc8_20 to %.loc8_30.3 [template = constants.%.10]
// CHECK:STDOUT:   %.loc8_30.5: i32 = int_literal 1 [template = constants.%.13]
// CHECK:STDOUT:   %.loc8_30.6: ref f64 = array_index %b.var, %.loc8_30.5
// CHECK:STDOUT:   %.loc8_30.7: init f64 = initialize_from %.loc8_26 to %.loc8_30.6 [template = constants.%.11]
// CHECK:STDOUT:   %.loc8_30.8: init [f64; 2] = array_init (%.loc8_30.4, %.loc8_30.7) to %b.var [template = constants.%.14]
// CHECK:STDOUT:   %.loc8_30.9: init [f64; 2] = converted %.loc8_30.1, %.loc8_30.8 [template = constants.%.14]
// CHECK:STDOUT:   assign %b.var, %.loc8_30.9
// CHECK:STDOUT:   %.loc9_10.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_13: i32 = int_literal 5 [template = constants.%.16]
// CHECK:STDOUT:   %.loc9_10.2: type = converted %.loc9_10.1, constants.%.15 [template = constants.%.15]
// CHECK:STDOUT:   %.loc9_14: type = array_type %.loc9_13, () [template = constants.%.17]
// CHECK:STDOUT:   %c.var: ref [(); 5] = var c
// CHECK:STDOUT:   %c: ref [(); 5] = bind_name c, %c.var
// CHECK:STDOUT:   %.loc9_20.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_24.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_28.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_32.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_36.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_38.1: ((), (), (), (), ()) = tuple_literal (%.loc9_20.1, %.loc9_24.1, %.loc9_28.1, %.loc9_32.1, %.loc9_36.1)
// CHECK:STDOUT:   %.loc9_38.2: i32 = int_literal 0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_38.3: ref () = array_index %c.var, %.loc9_38.2
// CHECK:STDOUT:   %.loc9_20.2: init () = tuple_init () to %.loc9_38.3 [template = constants.%.20]
// CHECK:STDOUT:   %.loc9_20.3: init () = converted %.loc9_20.1, %.loc9_20.2 [template = constants.%.20]
// CHECK:STDOUT:   %.loc9_38.4: i32 = int_literal 1 [template = constants.%.13]
// CHECK:STDOUT:   %.loc9_38.5: ref () = array_index %c.var, %.loc9_38.4
// CHECK:STDOUT:   %.loc9_24.2: init () = tuple_init () to %.loc9_38.5 [template = constants.%.20]
// CHECK:STDOUT:   %.loc9_24.3: init () = converted %.loc9_24.1, %.loc9_24.2 [template = constants.%.20]
// CHECK:STDOUT:   %.loc9_38.6: i32 = int_literal 2 [template = constants.%.21]
// CHECK:STDOUT:   %.loc9_38.7: ref () = array_index %c.var, %.loc9_38.6
// CHECK:STDOUT:   %.loc9_28.2: init () = tuple_init () to %.loc9_38.7 [template = constants.%.20]
// CHECK:STDOUT:   %.loc9_28.3: init () = converted %.loc9_28.1, %.loc9_28.2 [template = constants.%.20]
// CHECK:STDOUT:   %.loc9_38.8: i32 = int_literal 3 [template = constants.%.22]
// CHECK:STDOUT:   %.loc9_38.9: ref () = array_index %c.var, %.loc9_38.8
// CHECK:STDOUT:   %.loc9_32.2: init () = tuple_init () to %.loc9_38.9 [template = constants.%.20]
// CHECK:STDOUT:   %.loc9_32.3: init () = converted %.loc9_32.1, %.loc9_32.2 [template = constants.%.20]
// CHECK:STDOUT:   %.loc9_38.10: i32 = int_literal 4 [template = constants.%.23]
// CHECK:STDOUT:   %.loc9_38.11: ref () = array_index %c.var, %.loc9_38.10
// CHECK:STDOUT:   %.loc9_36.2: init () = tuple_init () to %.loc9_38.11 [template = constants.%.20]
// CHECK:STDOUT:   %.loc9_36.3: init () = converted %.loc9_36.1, %.loc9_36.2 [template = constants.%.20]
// CHECK:STDOUT:   %.loc9_38.12: init [(); 5] = array_init (%.loc9_20.3, %.loc9_24.3, %.loc9_28.3, %.loc9_32.3, %.loc9_36.3) to %c.var [template = constants.%.24]
// CHECK:STDOUT:   %.loc9_38.13: init [(); 5] = converted %.loc9_38.1, %.loc9_38.12 [template = constants.%.24]
// CHECK:STDOUT:   assign %c.var, %.loc9_38.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
