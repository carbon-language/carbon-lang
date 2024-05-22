// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

let v: (i32, i32) = (4, 102);
let w: (i32, i32) = v;

var x: (i32, i32) = (4, 102);
var y: (i32, i32) = x;

// CHECK:STDOUT: --- two_elements.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.4: type = ptr_type (i32, i32) [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 102 [template]
// CHECK:STDOUT:   %tuple: (i32, i32) = tuple_value (%.5, %.6) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc7_9: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc7_14: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc7_17.1: (type, type) = tuple_literal (%int.make_type_32.loc7_9, %int.make_type_32.loc7_14)
// CHECK:STDOUT:   %.loc7_17.2: type = value_of_initializer %int.make_type_32.loc7_9 [template = i32]
// CHECK:STDOUT:   %.loc7_17.3: type = converted %int.make_type_32.loc7_9, %.loc7_17.2 [template = i32]
// CHECK:STDOUT:   %.loc7_17.4: type = value_of_initializer %int.make_type_32.loc7_14 [template = i32]
// CHECK:STDOUT:   %.loc7_17.5: type = converted %int.make_type_32.loc7_14, %.loc7_17.4 [template = i32]
// CHECK:STDOUT:   %.loc7_17.6: type = converted %.loc7_17.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc7_22: i32 = int_literal 4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_25: i32 = int_literal 102 [template = constants.%.6]
// CHECK:STDOUT:   %.loc7_28: (i32, i32) = tuple_literal (%.loc7_22, %.loc7_25)
// CHECK:STDOUT:   %tuple: (i32, i32) = tuple_value (%.loc7_22, %.loc7_25) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_29: (i32, i32) = converted %.loc7_28, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %v: (i32, i32) = bind_name v, %.loc7_29
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc8_9: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc8_14: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc8_17.1: (type, type) = tuple_literal (%int.make_type_32.loc8_9, %int.make_type_32.loc8_14)
// CHECK:STDOUT:   %.loc8_17.2: type = value_of_initializer %int.make_type_32.loc8_9 [template = i32]
// CHECK:STDOUT:   %.loc8_17.3: type = converted %int.make_type_32.loc8_9, %.loc8_17.2 [template = i32]
// CHECK:STDOUT:   %.loc8_17.4: type = value_of_initializer %int.make_type_32.loc8_14 [template = i32]
// CHECK:STDOUT:   %.loc8_17.5: type = converted %int.make_type_32.loc8_14, %.loc8_17.4 [template = i32]
// CHECK:STDOUT:   %.loc8_17.6: type = converted %.loc8_17.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %v.ref: (i32, i32) = name_ref v, %v
// CHECK:STDOUT:   %w: (i32, i32) = bind_name w, %v.ref
// CHECK:STDOUT:   %import_ref.5: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc10_9: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %import_ref.6: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc10_14: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc10_17.1: (type, type) = tuple_literal (%int.make_type_32.loc10_9, %int.make_type_32.loc10_14)
// CHECK:STDOUT:   %.loc10_17.2: type = value_of_initializer %int.make_type_32.loc10_9 [template = i32]
// CHECK:STDOUT:   %.loc10_17.3: type = converted %int.make_type_32.loc10_9, %.loc10_17.2 [template = i32]
// CHECK:STDOUT:   %.loc10_17.4: type = value_of_initializer %int.make_type_32.loc10_14 [template = i32]
// CHECK:STDOUT:   %.loc10_17.5: type = converted %int.make_type_32.loc10_14, %.loc10_17.4 [template = i32]
// CHECK:STDOUT:   %.loc10_17.6: type = converted %.loc10_17.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %x.var: ref (i32, i32) = var x
// CHECK:STDOUT:   %x: ref (i32, i32) = bind_name x, %x.var
// CHECK:STDOUT:   %import_ref.7: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc11_9: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %import_ref.8: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc11_14: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc11_17.1: (type, type) = tuple_literal (%int.make_type_32.loc11_9, %int.make_type_32.loc11_14)
// CHECK:STDOUT:   %.loc11_17.2: type = value_of_initializer %int.make_type_32.loc11_9 [template = i32]
// CHECK:STDOUT:   %.loc11_17.3: type = converted %int.make_type_32.loc11_9, %.loc11_17.2 [template = i32]
// CHECK:STDOUT:   %.loc11_17.4: type = value_of_initializer %int.make_type_32.loc11_14 [template = i32]
// CHECK:STDOUT:   %.loc11_17.5: type = converted %int.make_type_32.loc11_14, %.loc11_17.4 [template = i32]
// CHECK:STDOUT:   %.loc11_17.6: type = converted %.loc11_17.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %y.var: ref (i32, i32) = var y
// CHECK:STDOUT:   %y: ref (i32, i32) = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_22: i32 = int_literal 4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc10_25: i32 = int_literal 102 [template = constants.%.6]
// CHECK:STDOUT:   %.loc10_28.1: (i32, i32) = tuple_literal (%.loc10_22, %.loc10_25)
// CHECK:STDOUT:   %.loc10_28.2: ref i32 = tuple_access file.%x.var, element0
// CHECK:STDOUT:   %.loc10_28.3: init i32 = initialize_from %.loc10_22 to %.loc10_28.2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc10_28.4: ref i32 = tuple_access file.%x.var, element1
// CHECK:STDOUT:   %.loc10_28.5: init i32 = initialize_from %.loc10_25 to %.loc10_28.4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc10_28.6: init (i32, i32) = tuple_init (%.loc10_28.3, %.loc10_28.5) to file.%x.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc10_29: init (i32, i32) = converted %.loc10_28.1, %.loc10_28.6 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%x.var, %.loc10_29
// CHECK:STDOUT:   %x.ref: ref (i32, i32) = name_ref x, file.%x
// CHECK:STDOUT:   %.loc11_21.1: ref i32 = tuple_access %x.ref, element0
// CHECK:STDOUT:   %.loc11_21.2: i32 = bind_value %.loc11_21.1
// CHECK:STDOUT:   %.loc11_21.3: ref i32 = tuple_access file.%y.var, element0
// CHECK:STDOUT:   %.loc11_21.4: init i32 = initialize_from %.loc11_21.2 to %.loc11_21.3
// CHECK:STDOUT:   %.loc11_21.5: ref i32 = tuple_access %x.ref, element1
// CHECK:STDOUT:   %.loc11_21.6: i32 = bind_value %.loc11_21.5
// CHECK:STDOUT:   %.loc11_21.7: ref i32 = tuple_access file.%y.var, element1
// CHECK:STDOUT:   %.loc11_21.8: init i32 = initialize_from %.loc11_21.6 to %.loc11_21.7
// CHECK:STDOUT:   %.loc11_21.9: init (i32, i32) = tuple_init (%.loc11_21.4, %.loc11_21.8) to file.%y.var
// CHECK:STDOUT:   %.loc11_22: init (i32, i32) = converted %x.ref, %.loc11_21.9
// CHECK:STDOUT:   assign file.%y.var, %.loc11_22
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
