// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

let t: type = (i32, i32) as type;

// CHECK:STDOUT: --- as_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc7_16: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc7_21: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc7_24: (type, type) = tuple_literal (%int.make_type_32.loc7_16, %int.make_type_32.loc7_21)
// CHECK:STDOUT:   %.loc7_26.1: type = value_of_initializer %int.make_type_32.loc7_16 [template = i32]
// CHECK:STDOUT:   %.loc7_26.2: type = converted %int.make_type_32.loc7_16, %.loc7_26.1 [template = i32]
// CHECK:STDOUT:   %.loc7_26.3: type = value_of_initializer %int.make_type_32.loc7_21 [template = i32]
// CHECK:STDOUT:   %.loc7_26.4: type = converted %int.make_type_32.loc7_21, %.loc7_26.3 [template = i32]
// CHECK:STDOUT:   %.loc7_26.5: type = converted %.loc7_24, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %t: type = bind_name t, %.loc7_26.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
