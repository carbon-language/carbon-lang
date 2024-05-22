// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn G() {
  // These should have two different constant values.
  var a: [i32; 3] = (1, 2, 3);
  var b: (i32, i32, i32) = (1, 2, 3);
}

// CHECK:STDOUT: --- array_vs_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: G = struct_value () [template]
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %struct.2: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, i32 [template]
// CHECK:STDOUT:   %.4: type = ptr_type [i32; 3] [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.7: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %array: [i32; 3] = tuple_value (%.5, %.6, %.2) [template]
// CHECK:STDOUT:   %.9: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %.10: type = ptr_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %tuple: (i32, i32, i32) = tuple_value (%.5, %.6, %.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.1] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc9: init type = call constants.%struct.2() [template = i32]
// CHECK:STDOUT:   %.loc9_16: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc9_11.1: type = value_of_initializer %int.make_type_32.loc9 [template = i32]
// CHECK:STDOUT:   %.loc9_11.2: type = converted %int.make_type_32.loc9, %.loc9_11.1 [template = i32]
// CHECK:STDOUT:   %.loc9_17: type = array_type %.loc9_16, i32 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref [i32; 3] = var a
// CHECK:STDOUT:   %a: ref [i32; 3] = bind_name a, %a.var
// CHECK:STDOUT:   %.loc9_22: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_25: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_28: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc9_29.1: (i32, i32, i32) = tuple_literal (%.loc9_22, %.loc9_25, %.loc9_28)
// CHECK:STDOUT:   %.loc9_29.2: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc9_29.3: ref i32 = array_index %a.var, %.loc9_29.2
// CHECK:STDOUT:   %.loc9_29.4: init i32 = initialize_from %.loc9_22 to %.loc9_29.3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_29.5: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_29.6: ref i32 = array_index %a.var, %.loc9_29.5
// CHECK:STDOUT:   %.loc9_29.7: init i32 = initialize_from %.loc9_25 to %.loc9_29.6 [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_29.8: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_29.9: ref i32 = array_index %a.var, %.loc9_29.8
// CHECK:STDOUT:   %.loc9_29.10: init i32 = initialize_from %.loc9_28 to %.loc9_29.9 [template = constants.%.2]
// CHECK:STDOUT:   %.loc9_29.11: init [i32; 3] = array_init (%.loc9_29.4, %.loc9_29.7, %.loc9_29.10) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc9_30: init [i32; 3] = converted %.loc9_29.1, %.loc9_29.11 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc9_30
// CHECK:STDOUT:   %int.make_type_32.loc10_11: init type = call constants.%struct.2() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc10_16: init type = call constants.%struct.2() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc10_21: init type = call constants.%struct.2() [template = i32]
// CHECK:STDOUT:   %.loc10_24.1: (type, type, type) = tuple_literal (%int.make_type_32.loc10_11, %int.make_type_32.loc10_16, %int.make_type_32.loc10_21)
// CHECK:STDOUT:   %.loc10_24.2: type = value_of_initializer %int.make_type_32.loc10_11 [template = i32]
// CHECK:STDOUT:   %.loc10_24.3: type = converted %int.make_type_32.loc10_11, %.loc10_24.2 [template = i32]
// CHECK:STDOUT:   %.loc10_24.4: type = value_of_initializer %int.make_type_32.loc10_16 [template = i32]
// CHECK:STDOUT:   %.loc10_24.5: type = converted %int.make_type_32.loc10_16, %.loc10_24.4 [template = i32]
// CHECK:STDOUT:   %.loc10_24.6: type = value_of_initializer %int.make_type_32.loc10_21 [template = i32]
// CHECK:STDOUT:   %.loc10_24.7: type = converted %int.make_type_32.loc10_21, %.loc10_24.6 [template = i32]
// CHECK:STDOUT:   %.loc10_24.8: type = converted %.loc10_24.1, constants.%.7 [template = constants.%.7]
// CHECK:STDOUT:   %b.var: ref (i32, i32, i32) = var b
// CHECK:STDOUT:   %b: ref (i32, i32, i32) = bind_name b, %b.var
// CHECK:STDOUT:   %.loc10_29: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc10_32: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc10_35: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_36.1: (i32, i32, i32) = tuple_literal (%.loc10_29, %.loc10_32, %.loc10_35)
// CHECK:STDOUT:   %.loc10_36.2: ref i32 = tuple_access %b.var, element0
// CHECK:STDOUT:   %.loc10_36.3: init i32 = initialize_from %.loc10_29 to %.loc10_36.2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc10_36.4: ref i32 = tuple_access %b.var, element1
// CHECK:STDOUT:   %.loc10_36.5: init i32 = initialize_from %.loc10_32 to %.loc10_36.4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc10_36.6: ref i32 = tuple_access %b.var, element2
// CHECK:STDOUT:   %.loc10_36.7: init i32 = initialize_from %.loc10_35 to %.loc10_36.6 [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_36.8: init (i32, i32, i32) = tuple_init (%.loc10_36.3, %.loc10_36.5, %.loc10_36.7) to %b.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc10_37: init (i32, i32, i32) = converted %.loc10_36.1, %.loc10_36.8 [template = constants.%tuple]
// CHECK:STDOUT:   assign %b.var, %.loc10_37
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
