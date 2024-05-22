// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: (i32,) = (4,);
var y: (i32,) = x;

// CHECK:STDOUT: --- one_element.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %tuple: (i32,) = tuple_value (%.4) [template]
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
// CHECK:STDOUT:   %int.make_type_32.loc7: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc7_13.1: (type,) = tuple_literal (%int.make_type_32.loc7)
// CHECK:STDOUT:   %.loc7_13.2: type = value_of_initializer %int.make_type_32.loc7 [template = i32]
// CHECK:STDOUT:   %.loc7_13.3: type = converted %int.make_type_32.loc7, %.loc7_13.2 [template = i32]
// CHECK:STDOUT:   %.loc7_13.4: type = converted %.loc7_13.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %x.var: ref (i32,) = var x
// CHECK:STDOUT:   %x: ref (i32,) = bind_name x, %x.var
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc8: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc8_13.1: (type,) = tuple_literal (%int.make_type_32.loc8)
// CHECK:STDOUT:   %.loc8_13.2: type = value_of_initializer %int.make_type_32.loc8 [template = i32]
// CHECK:STDOUT:   %.loc8_13.3: type = converted %int.make_type_32.loc8, %.loc8_13.2 [template = i32]
// CHECK:STDOUT:   %.loc8_13.4: type = converted %.loc8_13.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %y.var: ref (i32,) = var y
// CHECK:STDOUT:   %y: ref (i32,) = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_18: i32 = int_literal 4 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_20.1: (i32,) = tuple_literal (%.loc7_18)
// CHECK:STDOUT:   %.loc7_20.2: init (i32,) = tuple_init (%.loc7_18) to file.%x.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_21: init (i32,) = converted %.loc7_20.1, %.loc7_20.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%x.var, %.loc7_21
// CHECK:STDOUT:   %x.ref: ref (i32,) = name_ref x, file.%x
// CHECK:STDOUT:   %.loc8_17.1: ref i32 = tuple_access %x.ref, element0
// CHECK:STDOUT:   %.loc8_17.2: i32 = bind_value %.loc8_17.1
// CHECK:STDOUT:   %.loc8_17.3: init (i32,) = tuple_init (%.loc8_17.2) to file.%y.var
// CHECK:STDOUT:   %.loc8_18: init (i32,) = converted %x.ref, %.loc8_17.3
// CHECK:STDOUT:   assign file.%y.var, %.loc8_18
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
