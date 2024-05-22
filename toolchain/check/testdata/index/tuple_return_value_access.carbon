// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> (i32,) { return (0,); }

fn Run() -> i32 {
  return F()[0];
}

// CHECK:STDOUT: --- tuple_return_value_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.2: F = struct_value () [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %tuple: (i32,) = tuple_value (%.4) [template]
// CHECK:STDOUT:   %Run: type = fn_type @Run [template]
// CHECK:STDOUT:   %struct.3: Run = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.2] {
// CHECK:STDOUT:     %int.make_type_32.loc7: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc7_16.1: (type,) = tuple_literal (%int.make_type_32.loc7)
// CHECK:STDOUT:     %.loc7_16.2: type = value_of_initializer %int.make_type_32.loc7 [template = i32]
// CHECK:STDOUT:     %.loc7_16.3: type = converted %int.make_type_32.loc7, %.loc7_16.2 [template = i32]
// CHECK:STDOUT:     %.loc7_16.4: type = converted %.loc7_16.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     @F.%return: ref (i32,) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %Run.decl: Run = fn_decl @Run [template = constants.%struct.3] {
// CHECK:STDOUT:     %int.make_type_32.loc9: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc9_13.1: type = value_of_initializer %int.make_type_32.loc9 [template = i32]
// CHECK:STDOUT:     %.loc9_13.2: type = converted %int.make_type_32.loc9, %.loc9_13.1 [template = i32]
// CHECK:STDOUT:     @Run.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> (i32,) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_28: i32 = int_literal 0 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_30: (i32,) = tuple_literal (%.loc7_28)
// CHECK:STDOUT:   %tuple: (i32,) = tuple_value (%.loc7_28) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_31: (i32,) = converted %.loc7_30, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   return %.loc7_31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: F = name_ref F, file.%F.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %F.call: init (i32,) = call %F.ref()
// CHECK:STDOUT:   %.loc10_14: i32 = int_literal 0 [template = constants.%.4]
// CHECK:STDOUT:   %.loc10_11.1: ref (i32,) = temporary_storage
// CHECK:STDOUT:   %.loc10_11.2: ref (i32,) = temporary %.loc10_11.1, %F.call
// CHECK:STDOUT:   %.loc10_15.1: ref i32 = tuple_index %.loc10_11.2, %.loc10_14
// CHECK:STDOUT:   %.loc10_15.2: i32 = bind_value %.loc10_15.1
// CHECK:STDOUT:   return %.loc10_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
