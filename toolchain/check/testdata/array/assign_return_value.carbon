// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> (i32,) { return (0,); }

fn Run() {
  var t: [i32; 1] = F();
}

// CHECK:STDOUT: --- assign_return_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %tuple: (i32,) = tuple_value (%.4) [template]
// CHECK:STDOUT:   %Run: type = fn_type @Run [template]
// CHECK:STDOUT:   %struct.2: Run = struct_value () [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: type = array_type %.5, i32 [template]
// CHECK:STDOUT:   %.7: type = ptr_type [i32; 1] [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.1] {
// CHECK:STDOUT:     %.loc7_16.1: (type,) = tuple_literal (i32)
// CHECK:STDOUT:     %.loc7_16.2: type = converted %.loc7_16.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:     @F.%return: ref (i32,) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Run.decl: Run = fn_decl @Run [template = constants.%struct.2] {}
// CHECK:STDOUT: }
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
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_16: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc10_17: type = array_type %.loc10_16, i32 [template = constants.%.6]
// CHECK:STDOUT:   %t.var: ref [i32; 1] = var t
// CHECK:STDOUT:   %t: ref [i32; 1] = bind_name t, %t.var
// CHECK:STDOUT:   %F.ref: F = name_ref F, file.%F.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %F.call: init (i32,) = call %F.ref()
// CHECK:STDOUT:   %.loc10_22.1: ref (i32,) = temporary_storage
// CHECK:STDOUT:   %.loc10_22.2: ref (i32,) = temporary %.loc10_22.1, %F.call
// CHECK:STDOUT:   %.loc10_22.3: ref i32 = tuple_access %.loc10_22.2, element0
// CHECK:STDOUT:   %.loc10_22.4: i32 = bind_value %.loc10_22.3
// CHECK:STDOUT:   %.loc10_22.5: i32 = int_literal 0 [template = constants.%.4]
// CHECK:STDOUT:   %.loc10_22.6: ref i32 = array_index %t.var, %.loc10_22.5
// CHECK:STDOUT:   %.loc10_22.7: init i32 = initialize_from %.loc10_22.4 to %.loc10_22.6
// CHECK:STDOUT:   %.loc10_22.8: init [i32; 1] = array_init (%.loc10_22.7) to %t.var
// CHECK:STDOUT:   %.loc10_24: init [i32; 1] = converted %F.call, %.loc10_22.8
// CHECK:STDOUT:   assign %t.var, %.loc10_24
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
