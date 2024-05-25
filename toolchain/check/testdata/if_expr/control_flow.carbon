// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn A() -> i32 { return 1; }
fn B() -> i32 { return 2; }

fn F(b: bool) -> i32 {
  return if b then A() else B();
}

// CHECK:STDOUT: --- control_flow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %struct.2: A = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %B: type = fn_type @B [template]
// CHECK:STDOUT:   %struct.3: B = struct_value () [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %Bool: type = fn_type @Bool [template]
// CHECK:STDOUT:   %struct.4: Bool = struct_value () [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.5: F = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %A.decl: A = fn_decl @A [template = constants.%struct.2] {
// CHECK:STDOUT:     %int.make_type_32.loc7: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc7_11.1: type = value_of_initializer %int.make_type_32.loc7 [template = i32]
// CHECK:STDOUT:     %.loc7_11.2: type = converted %int.make_type_32.loc7, %.loc7_11.1 [template = i32]
// CHECK:STDOUT:     @A.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %B.decl: B = fn_decl @B [template = constants.%struct.3] {
// CHECK:STDOUT:     %int.make_type_32.loc8: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc8_11.1: type = value_of_initializer %int.make_type_32.loc8 [template = i32]
// CHECK:STDOUT:     %.loc8_11.2: type = converted %int.make_type_32.loc8, %.loc8_11.1 [template = i32]
// CHECK:STDOUT:     @B.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3: Bool = import_ref ir7, inst+2, loaded [template = constants.%struct.4]
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.5] {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%struct.4() [template = bool]
// CHECK:STDOUT:     %.loc10_9.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc10_9.2: type = converted %bool.make_type, %.loc10_9.1 [template = bool]
// CHECK:STDOUT:     %b.loc10_6.1: bool = param b
// CHECK:STDOUT:     @F.%b: bool = bind_name b, %b.loc10_6.1
// CHECK:STDOUT:     %int.make_type_32.loc10: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc10_18.1: type = value_of_initializer %int.make_type_32.loc10 [template = i32]
// CHECK:STDOUT:     %.loc10_18.2: type = converted %int.make_type_32.loc10, %.loc10_18.1 [template = i32]
// CHECK:STDOUT:     @F.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   return %.loc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   return %.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%b: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %A.ref: A = name_ref A, file.%A.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %A.call: init i32 = call %A.ref()
// CHECK:STDOUT:   %.loc11_21.1: i32 = value_of_initializer %A.call
// CHECK:STDOUT:   %.loc11_21.2: i32 = converted %A.call, %.loc11_21.1
// CHECK:STDOUT:   br !if.expr.result(%.loc11_21.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %B.ref: B = name_ref B, file.%B.decl [template = constants.%struct.3]
// CHECK:STDOUT:   %B.call: init i32 = call %B.ref()
// CHECK:STDOUT:   %.loc11_24.1: i32 = value_of_initializer %B.call
// CHECK:STDOUT:   %.loc11_24.2: i32 = converted %B.call, %.loc11_24.1
// CHECK:STDOUT:   br !if.expr.result(%.loc11_24.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc11_10: i32 = block_arg !if.expr.result
// CHECK:STDOUT:   return %.loc11_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
