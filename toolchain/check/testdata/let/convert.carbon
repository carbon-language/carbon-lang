// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> i32 {
  var v: (i32, i32, i32) = (1, 2, 3);
  // Convert from object representation to value representation.
  let w: (i32, i32, i32) = v;
  return w[1];
}

// CHECK:STDOUT: --- convert.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.2: F = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.4: type = ptr_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %tuple: (i32, i32, i32) = tuple_value (%.5, %.6, %.7) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.2] {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc7_11.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc7_11.2: type = converted %int.make_type_32, %.loc7_11.1 [template = i32]
// CHECK:STDOUT:     @F.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.5: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.6: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.7: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc8_11: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc8_16: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc8_21: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc8_24.1: (type, type, type) = tuple_literal (%int.make_type_32.loc8_11, %int.make_type_32.loc8_16, %int.make_type_32.loc8_21)
// CHECK:STDOUT:   %.loc8_24.2: type = value_of_initializer %int.make_type_32.loc8_11 [template = i32]
// CHECK:STDOUT:   %.loc8_24.3: type = converted %int.make_type_32.loc8_11, %.loc8_24.2 [template = i32]
// CHECK:STDOUT:   %.loc8_24.4: type = value_of_initializer %int.make_type_32.loc8_16 [template = i32]
// CHECK:STDOUT:   %.loc8_24.5: type = converted %int.make_type_32.loc8_16, %.loc8_24.4 [template = i32]
// CHECK:STDOUT:   %.loc8_24.6: type = value_of_initializer %int.make_type_32.loc8_21 [template = i32]
// CHECK:STDOUT:   %.loc8_24.7: type = converted %int.make_type_32.loc8_21, %.loc8_24.6 [template = i32]
// CHECK:STDOUT:   %.loc8_24.8: type = converted %.loc8_24.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %v.var: ref (i32, i32, i32) = var v
// CHECK:STDOUT:   %v: ref (i32, i32, i32) = bind_name v, %v.var
// CHECK:STDOUT:   %.loc8_29: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc8_32: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc8_35: i32 = int_literal 3 [template = constants.%.7]
// CHECK:STDOUT:   %.loc8_36.1: (i32, i32, i32) = tuple_literal (%.loc8_29, %.loc8_32, %.loc8_35)
// CHECK:STDOUT:   %.loc8_36.2: ref i32 = tuple_access %v.var, element0
// CHECK:STDOUT:   %.loc8_36.3: init i32 = initialize_from %.loc8_29 to %.loc8_36.2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc8_36.4: ref i32 = tuple_access %v.var, element1
// CHECK:STDOUT:   %.loc8_36.5: init i32 = initialize_from %.loc8_32 to %.loc8_36.4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc8_36.6: ref i32 = tuple_access %v.var, element2
// CHECK:STDOUT:   %.loc8_36.7: init i32 = initialize_from %.loc8_35 to %.loc8_36.6 [template = constants.%.7]
// CHECK:STDOUT:   %.loc8_36.8: init (i32, i32, i32) = tuple_init (%.loc8_36.3, %.loc8_36.5, %.loc8_36.7) to %v.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc8_37: init (i32, i32, i32) = converted %.loc8_36.1, %.loc8_36.8 [template = constants.%tuple]
// CHECK:STDOUT:   assign %v.var, %.loc8_37
// CHECK:STDOUT:   %int.make_type_32.loc10_11: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc10_16: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc10_21: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc10_24.1: (type, type, type) = tuple_literal (%int.make_type_32.loc10_11, %int.make_type_32.loc10_16, %int.make_type_32.loc10_21)
// CHECK:STDOUT:   %.loc10_24.2: type = value_of_initializer %int.make_type_32.loc10_11 [template = i32]
// CHECK:STDOUT:   %.loc10_24.3: type = converted %int.make_type_32.loc10_11, %.loc10_24.2 [template = i32]
// CHECK:STDOUT:   %.loc10_24.4: type = value_of_initializer %int.make_type_32.loc10_16 [template = i32]
// CHECK:STDOUT:   %.loc10_24.5: type = converted %int.make_type_32.loc10_16, %.loc10_24.4 [template = i32]
// CHECK:STDOUT:   %.loc10_24.6: type = value_of_initializer %int.make_type_32.loc10_21 [template = i32]
// CHECK:STDOUT:   %.loc10_24.7: type = converted %int.make_type_32.loc10_21, %.loc10_24.6 [template = i32]
// CHECK:STDOUT:   %.loc10_24.8: type = converted %.loc10_24.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %v.ref: ref (i32, i32, i32) = name_ref v, %v
// CHECK:STDOUT:   %.loc10_28.1: ref i32 = tuple_access %v.ref, element0
// CHECK:STDOUT:   %.loc10_28.2: i32 = bind_value %.loc10_28.1
// CHECK:STDOUT:   %.loc10_28.3: ref i32 = tuple_access %v.ref, element1
// CHECK:STDOUT:   %.loc10_28.4: i32 = bind_value %.loc10_28.3
// CHECK:STDOUT:   %.loc10_28.5: ref i32 = tuple_access %v.ref, element2
// CHECK:STDOUT:   %.loc10_28.6: i32 = bind_value %.loc10_28.5
// CHECK:STDOUT:   %tuple: (i32, i32, i32) = tuple_value (%.loc10_28.2, %.loc10_28.4, %.loc10_28.6)
// CHECK:STDOUT:   %.loc10_29: (i32, i32, i32) = converted %v.ref, %tuple
// CHECK:STDOUT:   %w: (i32, i32, i32) = bind_name w, %.loc10_29
// CHECK:STDOUT:   %w.ref: (i32, i32, i32) = name_ref w, %w
// CHECK:STDOUT:   %.loc11_12: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_13: i32 = tuple_index %w.ref, %.loc11_12
// CHECK:STDOUT:   return %.loc11_13
// CHECK:STDOUT: }
// CHECK:STDOUT:
