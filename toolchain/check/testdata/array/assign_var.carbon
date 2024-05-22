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
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.4: type = ptr_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %tuple: (i32, i32, i32) = tuple_value (%.5, %.6, %.7) [template]
// CHECK:STDOUT:   %.8: type = array_type %.7, i32 [template]
// CHECK:STDOUT:   %.9: type = ptr_type [i32; 3] [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc7_9: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc7_14: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc7_19: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc7_22.1: (type, type, type) = tuple_literal (%int.make_type_32.loc7_9, %int.make_type_32.loc7_14, %int.make_type_32.loc7_19)
// CHECK:STDOUT:   %.loc7_22.2: type = value_of_initializer %int.make_type_32.loc7_9 [template = i32]
// CHECK:STDOUT:   %.loc7_22.3: type = converted %int.make_type_32.loc7_9, %.loc7_22.2 [template = i32]
// CHECK:STDOUT:   %.loc7_22.4: type = value_of_initializer %int.make_type_32.loc7_14 [template = i32]
// CHECK:STDOUT:   %.loc7_22.5: type = converted %int.make_type_32.loc7_14, %.loc7_22.4 [template = i32]
// CHECK:STDOUT:   %.loc7_22.6: type = value_of_initializer %int.make_type_32.loc7_19 [template = i32]
// CHECK:STDOUT:   %.loc7_22.7: type = converted %int.make_type_32.loc7_19, %.loc7_22.6 [template = i32]
// CHECK:STDOUT:   %.loc7_22.8: type = converted %.loc7_22.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref (i32, i32, i32) = var a
// CHECK:STDOUT:   %a: ref (i32, i32, i32) = bind_name a, %a.var
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc8: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc8_14: i32 = int_literal 3 [template = constants.%.7]
// CHECK:STDOUT:   %.loc8_9.1: type = value_of_initializer %int.make_type_32.loc8 [template = i32]
// CHECK:STDOUT:   %.loc8_9.2: type = converted %int.make_type_32.loc8, %.loc8_9.1 [template = i32]
// CHECK:STDOUT:   %.loc8_15: type = array_type %.loc8_14, i32 [template = constants.%.8]
// CHECK:STDOUT:   %b.var: ref [i32; 3] = var b
// CHECK:STDOUT:   %b: ref [i32; 3] = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_27: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_30: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc7_33: i32 = int_literal 3 [template = constants.%.7]
// CHECK:STDOUT:   %.loc7_34.1: (i32, i32, i32) = tuple_literal (%.loc7_27, %.loc7_30, %.loc7_33)
// CHECK:STDOUT:   %.loc7_34.2: ref i32 = tuple_access file.%a.var, element0
// CHECK:STDOUT:   %.loc7_34.3: init i32 = initialize_from %.loc7_27 to %.loc7_34.2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_34.4: ref i32 = tuple_access file.%a.var, element1
// CHECK:STDOUT:   %.loc7_34.5: init i32 = initialize_from %.loc7_30 to %.loc7_34.4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc7_34.6: ref i32 = tuple_access file.%a.var, element2
// CHECK:STDOUT:   %.loc7_34.7: init i32 = initialize_from %.loc7_33 to %.loc7_34.6 [template = constants.%.7]
// CHECK:STDOUT:   %.loc7_34.8: init (i32, i32, i32) = tuple_init (%.loc7_34.3, %.loc7_34.5, %.loc7_34.7) to file.%a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_35: init (i32, i32, i32) = converted %.loc7_34.1, %.loc7_34.8 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc7_35
// CHECK:STDOUT:   %a.ref: ref (i32, i32, i32) = name_ref a, file.%a
// CHECK:STDOUT:   %.loc8_19.1: ref i32 = tuple_access %a.ref, element0
// CHECK:STDOUT:   %.loc8_19.2: i32 = bind_value %.loc8_19.1
// CHECK:STDOUT:   %.loc8_19.3: i32 = int_literal 0 [template = constants.%.10]
// CHECK:STDOUT:   %.loc8_19.4: ref i32 = array_index file.%b.var, %.loc8_19.3
// CHECK:STDOUT:   %.loc8_19.5: init i32 = initialize_from %.loc8_19.2 to %.loc8_19.4
// CHECK:STDOUT:   %.loc8_19.6: ref i32 = tuple_access %a.ref, element1
// CHECK:STDOUT:   %.loc8_19.7: i32 = bind_value %.loc8_19.6
// CHECK:STDOUT:   %.loc8_19.8: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc8_19.9: ref i32 = array_index file.%b.var, %.loc8_19.8
// CHECK:STDOUT:   %.loc8_19.10: init i32 = initialize_from %.loc8_19.7 to %.loc8_19.9
// CHECK:STDOUT:   %.loc8_19.11: ref i32 = tuple_access %a.ref, element2
// CHECK:STDOUT:   %.loc8_19.12: i32 = bind_value %.loc8_19.11
// CHECK:STDOUT:   %.loc8_19.13: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc8_19.14: ref i32 = array_index file.%b.var, %.loc8_19.13
// CHECK:STDOUT:   %.loc8_19.15: init i32 = initialize_from %.loc8_19.12 to %.loc8_19.14
// CHECK:STDOUT:   %.loc8_19.16: init [i32; 3] = array_init (%.loc8_19.5, %.loc8_19.10, %.loc8_19.15) to file.%b.var
// CHECK:STDOUT:   %.loc8_20: init [i32; 3] = converted %a.ref, %.loc8_19.16
// CHECK:STDOUT:   assign file.%b.var, %.loc8_20
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
