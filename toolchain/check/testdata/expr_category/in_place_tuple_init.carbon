// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> (i32, i32);

fn G() -> (i32, i32) {
  var v: (i32, i32) = F();
  v = F();
  return F();
}

fn H() -> i32 {
  return G()[0];
}

// CHECK:STDOUT: --- in_place_tuple_init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.2: F = struct_value () [template]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.3: G = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type (i32, i32) [template]
// CHECK:STDOUT:   %H: type = fn_type @H [template]
// CHECK:STDOUT:   %struct.4: H = struct_value () [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.2] {
// CHECK:STDOUT:     %int.make_type_32.loc7_12: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc7_17: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc7_20.1: (type, type) = tuple_literal (%int.make_type_32.loc7_12, %int.make_type_32.loc7_17)
// CHECK:STDOUT:     %.loc7_20.2: type = value_of_initializer %int.make_type_32.loc7_12 [template = i32]
// CHECK:STDOUT:     %.loc7_20.3: type = converted %int.make_type_32.loc7_12, %.loc7_20.2 [template = i32]
// CHECK:STDOUT:     %.loc7_20.4: type = value_of_initializer %int.make_type_32.loc7_17 [template = i32]
// CHECK:STDOUT:     %.loc7_20.5: type = converted %int.make_type_32.loc7_17, %.loc7_20.4 [template = i32]
// CHECK:STDOUT:     %.loc7_20.6: type = converted %.loc7_20.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     @F.%return: ref (i32, i32) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.3] {
// CHECK:STDOUT:     %int.make_type_32.loc9_12: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc9_17: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc9_20.1: (type, type) = tuple_literal (%int.make_type_32.loc9_12, %int.make_type_32.loc9_17)
// CHECK:STDOUT:     %.loc9_20.2: type = value_of_initializer %int.make_type_32.loc9_12 [template = i32]
// CHECK:STDOUT:     %.loc9_20.3: type = converted %int.make_type_32.loc9_12, %.loc9_20.2 [template = i32]
// CHECK:STDOUT:     %.loc9_20.4: type = value_of_initializer %int.make_type_32.loc9_17 [template = i32]
// CHECK:STDOUT:     %.loc9_20.5: type = converted %int.make_type_32.loc9_17, %.loc9_20.4 [template = i32]
// CHECK:STDOUT:     %.loc9_20.6: type = converted %.loc9_20.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     @G.%return: ref (i32, i32) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.6: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.7: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %H.decl: H = fn_decl @H [template = constants.%struct.4] {
// CHECK:STDOUT:     %int.make_type_32.loc15: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc15_11.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:     %.loc15_11.2: type = converted %int.make_type_32.loc15, %.loc15_11.1 [template = i32]
// CHECK:STDOUT:     @H.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> (i32, i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return: (i32, i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc10_11: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc10_16: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc10_19.1: (type, type) = tuple_literal (%int.make_type_32.loc10_11, %int.make_type_32.loc10_16)
// CHECK:STDOUT:   %.loc10_19.2: type = value_of_initializer %int.make_type_32.loc10_11 [template = i32]
// CHECK:STDOUT:   %.loc10_19.3: type = converted %int.make_type_32.loc10_11, %.loc10_19.2 [template = i32]
// CHECK:STDOUT:   %.loc10_19.4: type = value_of_initializer %int.make_type_32.loc10_16 [template = i32]
// CHECK:STDOUT:   %.loc10_19.5: type = converted %int.make_type_32.loc10_16, %.loc10_19.4 [template = i32]
// CHECK:STDOUT:   %.loc10_19.6: type = converted %.loc10_19.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %v.var: ref (i32, i32) = var v
// CHECK:STDOUT:   %v: ref (i32, i32) = bind_name v, %v.var
// CHECK:STDOUT:   %F.ref.loc10: F = name_ref F, file.%F.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc10_7: ref (i32, i32) = splice_block %v.var {}
// CHECK:STDOUT:   %F.call.loc10: init (i32, i32) = call %F.ref.loc10() to %.loc10_7
// CHECK:STDOUT:   assign %v.var, %F.call.loc10
// CHECK:STDOUT:   %v.ref: ref (i32, i32) = name_ref v, %v
// CHECK:STDOUT:   %F.ref.loc11: F = name_ref F, file.%F.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc11: ref (i32, i32) = splice_block %v.ref {}
// CHECK:STDOUT:   %F.call.loc11: init (i32, i32) = call %F.ref.loc11() to %.loc11
// CHECK:STDOUT:   assign %v.ref, %F.call.loc11
// CHECK:STDOUT:   %F.ref.loc12: F = name_ref F, file.%F.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc9: ref (i32, i32) = splice_block %return {}
// CHECK:STDOUT:   %F.call.loc12: init (i32, i32) = call %F.ref.loc12() to %.loc9
// CHECK:STDOUT:   return %F.call.loc12 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: G = name_ref G, file.%G.decl [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc16_11.1: ref (i32, i32) = temporary_storage
// CHECK:STDOUT:   %G.call: init (i32, i32) = call %G.ref() to %.loc16_11.1
// CHECK:STDOUT:   %.loc16_14: i32 = int_literal 0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc16_11.2: ref (i32, i32) = temporary %.loc16_11.1, %G.call
// CHECK:STDOUT:   %.loc16_15.1: ref i32 = tuple_index %.loc16_11.2, %.loc16_14
// CHECK:STDOUT:   %.loc16_15.2: i32 = bind_value %.loc16_15.1
// CHECK:STDOUT:   return %.loc16_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
