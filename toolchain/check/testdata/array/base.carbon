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
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, i32 [template]
// CHECK:STDOUT:   %.4: type = ptr_type [i32; 1] [template]
// CHECK:STDOUT:   %.5: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %array.1: [i32; 1] = tuple_value (%.2) [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 64 [template]
// CHECK:STDOUT:   %Float: type = fn_type @Float [template]
// CHECK:STDOUT:   %struct.2: Float = struct_value () [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.9: type = array_type %.8, f64 [template]
// CHECK:STDOUT:   %.10: type = ptr_type [f64; 2] [template]
// CHECK:STDOUT:   %.11: f64 = float_literal 11.100000000000001 [template]
// CHECK:STDOUT:   %.12: f64 = float_literal 2.2000000000000002 [template]
// CHECK:STDOUT:   %.13: type = tuple_type (f64, f64) [template]
// CHECK:STDOUT:   %array.2: [f64; 2] = tuple_value (%.11, %.12) [template]
// CHECK:STDOUT:   %.14: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.15: type = array_type %.14, () [template]
// CHECK:STDOUT:   %.16: type = ptr_type [(); 5] [template]
// CHECK:STDOUT:   %.17: type = tuple_type ((), (), (), (), ()) [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT:   %.18: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.19: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %array.3: [(); 5] = tuple_value (%tuple, %tuple, %tuple, %tuple, %tuple) [template]
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
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc7_14: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc7_9.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc7_9.2: type = converted %int.make_type_32, %.loc7_9.1 [template = i32]
// CHECK:STDOUT:   %.loc7_15: type = array_type %.loc7_14, i32 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref [i32; 1] = var a
// CHECK:STDOUT:   %a: ref [i32; 1] = bind_name a, %a.var
// CHECK:STDOUT:   %.loc8_9.1: i32 = int_literal 64 [template = constants.%.7]
// CHECK:STDOUT:   %import_ref.2: Float = import_ref ir4, inst+30, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %float.make_type: init type = call constants.%struct.2(%.loc8_9.1) [template = f64]
// CHECK:STDOUT:   %.loc8_14: i32 = int_literal 2 [template = constants.%.8]
// CHECK:STDOUT:   %.loc8_9.2: type = value_of_initializer %float.make_type [template = f64]
// CHECK:STDOUT:   %.loc8_9.3: type = converted %float.make_type, %.loc8_9.2 [template = f64]
// CHECK:STDOUT:   %.loc8_15: type = array_type %.loc8_14, f64 [template = constants.%.9]
// CHECK:STDOUT:   %b.var: ref [f64; 2] = var b
// CHECK:STDOUT:   %b: ref [f64; 2] = bind_name b, %b.var
// CHECK:STDOUT:   %.loc9_10.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_13: i32 = int_literal 5 [template = constants.%.14]
// CHECK:STDOUT:   %.loc9_10.2: type = converted %.loc9_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc9_14: type = array_type %.loc9_13, () [template = constants.%.15]
// CHECK:STDOUT:   %c.var: ref [(); 5] = var c
// CHECK:STDOUT:   %c: ref [(); 5] = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size: i32) -> type = "float.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_20: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc7_22.1: (i32,) = tuple_literal (%.loc7_20)
// CHECK:STDOUT:   %.loc7_22.2: i32 = int_literal 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc7_22.3: ref i32 = array_index file.%a.var, %.loc7_22.2
// CHECK:STDOUT:   %.loc7_22.4: init i32 = initialize_from %.loc7_20 to %.loc7_22.3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc7_22.5: init [i32; 1] = array_init (%.loc7_22.4) to file.%a.var [template = constants.%array.1]
// CHECK:STDOUT:   %.loc7_23: init [i32; 1] = converted %.loc7_22.1, %.loc7_22.5 [template = constants.%array.1]
// CHECK:STDOUT:   assign file.%a.var, %.loc7_23
// CHECK:STDOUT:   %.loc8_20: f64 = float_literal 11.100000000000001 [template = constants.%.11]
// CHECK:STDOUT:   %.loc8_26: f64 = float_literal 2.2000000000000002 [template = constants.%.12]
// CHECK:STDOUT:   %.loc8_30.1: (f64, f64) = tuple_literal (%.loc8_20, %.loc8_26)
// CHECK:STDOUT:   %.loc8_30.2: i32 = int_literal 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc8_30.3: ref f64 = array_index file.%b.var, %.loc8_30.2
// CHECK:STDOUT:   %.loc8_30.4: init f64 = initialize_from %.loc8_20 to %.loc8_30.3 [template = constants.%.11]
// CHECK:STDOUT:   %.loc8_30.5: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_30.6: ref f64 = array_index file.%b.var, %.loc8_30.5
// CHECK:STDOUT:   %.loc8_30.7: init f64 = initialize_from %.loc8_26 to %.loc8_30.6 [template = constants.%.12]
// CHECK:STDOUT:   %.loc8_30.8: init [f64; 2] = array_init (%.loc8_30.4, %.loc8_30.7) to file.%b.var [template = constants.%array.2]
// CHECK:STDOUT:   %.loc8_31: init [f64; 2] = converted %.loc8_30.1, %.loc8_30.8 [template = constants.%array.2]
// CHECK:STDOUT:   assign file.%b.var, %.loc8_31
// CHECK:STDOUT:   %.loc9_20.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_24.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_28.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_32.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_36.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_38.1: ((), (), (), (), ()) = tuple_literal (%.loc9_20.1, %.loc9_24.1, %.loc9_28.1, %.loc9_32.1, %.loc9_36.1)
// CHECK:STDOUT:   %.loc9_38.2: i32 = int_literal 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_38.3: ref () = array_index file.%c.var, %.loc9_38.2
// CHECK:STDOUT:   %.loc9_20.2: init () = tuple_init () to %.loc9_38.3 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc9_38.4: init () = converted %.loc9_20.1, %.loc9_20.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc9_38.5: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc9_38.6: ref () = array_index file.%c.var, %.loc9_38.5
// CHECK:STDOUT:   %.loc9_24.2: init () = tuple_init () to %.loc9_38.6 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc9_38.7: init () = converted %.loc9_24.1, %.loc9_24.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc9_38.8: i32 = int_literal 2 [template = constants.%.8]
// CHECK:STDOUT:   %.loc9_38.9: ref () = array_index file.%c.var, %.loc9_38.8
// CHECK:STDOUT:   %.loc9_28.2: init () = tuple_init () to %.loc9_38.9 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc9_38.10: init () = converted %.loc9_28.1, %.loc9_28.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc9_38.11: i32 = int_literal 3 [template = constants.%.18]
// CHECK:STDOUT:   %.loc9_38.12: ref () = array_index file.%c.var, %.loc9_38.11
// CHECK:STDOUT:   %.loc9_32.2: init () = tuple_init () to %.loc9_38.12 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc9_38.13: init () = converted %.loc9_32.1, %.loc9_32.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc9_38.14: i32 = int_literal 4 [template = constants.%.19]
// CHECK:STDOUT:   %.loc9_38.15: ref () = array_index file.%c.var, %.loc9_38.14
// CHECK:STDOUT:   %.loc9_36.2: init () = tuple_init () to %.loc9_38.15 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc9_38.16: init () = converted %.loc9_36.1, %.loc9_36.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc9_38.17: init [(); 5] = array_init (%.loc9_38.4, %.loc9_38.7, %.loc9_38.10, %.loc9_38.13, %.loc9_38.16) to file.%c.var [template = constants.%array.3]
// CHECK:STDOUT:   %.loc9_39: init [(); 5] = converted %.loc9_38.1, %.loc9_38.17 [template = constants.%array.3]
// CHECK:STDOUT:   assign file.%c.var, %.loc9_39
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
