// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn A() { if (true) { var n: i32 = 1; } if (true) { var n: i32 = 2; } }

// CHECK:STDOUT: --- duplicate_name_same_line.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: A = struct_value () [template]
// CHECK:STDOUT:   %.2: bool = bool_literal true [template]
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %struct.2: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.decl: A = fn_decl @A [template = constants.%struct.1] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_14: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   if %.loc7_14 br !if.then.loc7_18 else br !if.else.loc7_18
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc7_18:
// CHECK:STDOUT:   %int.make_type_32.loc7_29: init type = call constants.%struct.2() [template = i32]
// CHECK:STDOUT:   %.loc7_29.1: type = value_of_initializer %int.make_type_32.loc7_29 [template = i32]
// CHECK:STDOUT:   %.loc7_29.2: type = converted %int.make_type_32.loc7_29, %.loc7_29.1 [template = i32]
// CHECK:STDOUT:   %n.var.loc7_26: ref i32 = var n
// CHECK:STDOUT:   %n.loc7_26: ref i32 = bind_name n, %n.var.loc7_26
// CHECK:STDOUT:   %.loc7_35: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   assign %n.var.loc7_26, %.loc7_35
// CHECK:STDOUT:   br !if.else.loc7_18
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc7_18:
// CHECK:STDOUT:   %.loc7_44: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   if %.loc7_44 br !if.then.loc7_48 else br !if.else.loc7_48
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc7_48:
// CHECK:STDOUT:   %int.make_type_32.loc7_59: init type = call constants.%struct.2() [template = i32]
// CHECK:STDOUT:   %.loc7_59.1: type = value_of_initializer %int.make_type_32.loc7_59 [template = i32]
// CHECK:STDOUT:   %.loc7_59.2: type = converted %int.make_type_32.loc7_59, %.loc7_59.1 [template = i32]
// CHECK:STDOUT:   %n.var.loc7_56: ref i32 = var n
// CHECK:STDOUT:   %n.loc7_56: ref i32 = bind_name n, %n.var.loc7_56
// CHECK:STDOUT:   %.loc7_65: i32 = int_literal 2 [template = constants.%.4]
// CHECK:STDOUT:   assign %n.var.loc7_56, %.loc7_65
// CHECK:STDOUT:   br !if.else.loc7_48
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc7_48:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
