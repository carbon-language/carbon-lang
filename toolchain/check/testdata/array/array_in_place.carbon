// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> (i32, i32, i32);

fn G() {
  var v: [(i32, i32, i32); 2] = (F(), F());
}

// CHECK:STDOUT: --- array_in_place.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.2: F = struct_value () [template]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.3: G = struct_value () [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.5: type = array_type %.4, (i32, i32, i32) [template]
// CHECK:STDOUT:   %.6: type = ptr_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.7: type = ptr_type [(i32, i32, i32); 2] [template]
// CHECK:STDOUT:   %.8: type = tuple_type ((i32, i32, i32), (i32, i32, i32)) [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.2] {
// CHECK:STDOUT:     %int.make_type_32.loc7_12: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc7_17: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc7_22: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc7_25.1: (type, type, type) = tuple_literal (%int.make_type_32.loc7_12, %int.make_type_32.loc7_17, %int.make_type_32.loc7_22)
// CHECK:STDOUT:     %.loc7_25.2: type = value_of_initializer %int.make_type_32.loc7_12 [template = i32]
// CHECK:STDOUT:     %.loc7_25.3: type = converted %int.make_type_32.loc7_12, %.loc7_25.2 [template = i32]
// CHECK:STDOUT:     %.loc7_25.4: type = value_of_initializer %int.make_type_32.loc7_17 [template = i32]
// CHECK:STDOUT:     %.loc7_25.5: type = converted %int.make_type_32.loc7_17, %.loc7_25.4 [template = i32]
// CHECK:STDOUT:     %.loc7_25.6: type = value_of_initializer %int.make_type_32.loc7_22 [template = i32]
// CHECK:STDOUT:     %.loc7_25.7: type = converted %int.make_type_32.loc7_22, %.loc7_25.6 [template = i32]
// CHECK:STDOUT:     %.loc7_25.8: type = converted %.loc7_25.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     @F.%return: ref (i32, i32, i32) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.3] {}
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.5: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.6: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> (i32, i32, i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc10_12: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc10_17: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc10_22: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc10_25.1: (type, type, type) = tuple_literal (%int.make_type_32.loc10_12, %int.make_type_32.loc10_17, %int.make_type_32.loc10_22)
// CHECK:STDOUT:   %.loc10_28: i32 = int_literal 2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc10_25.2: type = value_of_initializer %int.make_type_32.loc10_12 [template = i32]
// CHECK:STDOUT:   %.loc10_25.3: type = converted %int.make_type_32.loc10_12, %.loc10_25.2 [template = i32]
// CHECK:STDOUT:   %.loc10_25.4: type = value_of_initializer %int.make_type_32.loc10_17 [template = i32]
// CHECK:STDOUT:   %.loc10_25.5: type = converted %int.make_type_32.loc10_17, %.loc10_25.4 [template = i32]
// CHECK:STDOUT:   %.loc10_25.6: type = value_of_initializer %int.make_type_32.loc10_22 [template = i32]
// CHECK:STDOUT:   %.loc10_25.7: type = converted %int.make_type_32.loc10_22, %.loc10_25.6 [template = i32]
// CHECK:STDOUT:   %.loc10_25.8: type = converted %.loc10_25.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc10_29: type = array_type %.loc10_28, (i32, i32, i32) [template = constants.%.5]
// CHECK:STDOUT:   %v.var: ref [(i32, i32, i32); 2] = var v
// CHECK:STDOUT:   %v: ref [(i32, i32, i32); 2] = bind_name v, %v.var
// CHECK:STDOUT:   %F.ref.loc10_34: F = name_ref F, file.%F.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc10_42.3: ref (i32, i32, i32) = splice_block %.loc10_42.2 {
// CHECK:STDOUT:     %.loc10_42.1: i32 = int_literal 0 [template = constants.%.9]
// CHECK:STDOUT:     %.loc10_42.2: ref (i32, i32, i32) = array_index %v.var, %.loc10_42.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.call.loc10_35: init (i32, i32, i32) = call %F.ref.loc10_34() to %.loc10_42.3
// CHECK:STDOUT:   %F.ref.loc10_39: F = name_ref F, file.%F.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc10_42.6: ref (i32, i32, i32) = splice_block %.loc10_42.5 {
// CHECK:STDOUT:     %.loc10_42.4: i32 = int_literal 1 [template = constants.%.10]
// CHECK:STDOUT:     %.loc10_42.5: ref (i32, i32, i32) = array_index %v.var, %.loc10_42.4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.call.loc10_40: init (i32, i32, i32) = call %F.ref.loc10_39() to %.loc10_42.6
// CHECK:STDOUT:   %.loc10_42.7: ((i32, i32, i32), (i32, i32, i32)) = tuple_literal (%F.call.loc10_35, %F.call.loc10_40)
// CHECK:STDOUT:   %.loc10_42.8: init [(i32, i32, i32); 2] = array_init (%F.call.loc10_35, %F.call.loc10_40) to %v.var
// CHECK:STDOUT:   %.loc10_43: init [(i32, i32, i32); 2] = converted %.loc10_42.7, %.loc10_42.8
// CHECK:STDOUT:   assign %v.var, %.loc10_43
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
