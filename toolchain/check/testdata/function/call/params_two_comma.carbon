// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Foo(a: i32, b: i32,) {}

fn Main() {
  Foo(1, 2);
  Foo(1, 2,);
}

// CHECK:STDOUT: --- params_two_comma.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %Foo: type = fn_type @Foo [template]
// CHECK:STDOUT:   %struct.2: Foo = struct_value () [template]
// CHECK:STDOUT:   %Main: type = fn_type @Main [template]
// CHECK:STDOUT:   %struct.3: Main = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %Foo.decl: Foo = fn_decl @Foo [template = constants.%struct.2] {
// CHECK:STDOUT:     %int.make_type_32.loc7_11: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc7_11.1: type = value_of_initializer %int.make_type_32.loc7_11 [template = i32]
// CHECK:STDOUT:     %.loc7_11.2: type = converted %int.make_type_32.loc7_11, %.loc7_11.1 [template = i32]
// CHECK:STDOUT:     %a.loc7_8.1: i32 = param a
// CHECK:STDOUT:     @Foo.%a: i32 = bind_name a, %a.loc7_8.1
// CHECK:STDOUT:     %int.make_type_32.loc7_19: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc7_19.1: type = value_of_initializer %int.make_type_32.loc7_19 [template = i32]
// CHECK:STDOUT:     %.loc7_19.2: type = converted %int.make_type_32.loc7_19, %.loc7_19.1 [template = i32]
// CHECK:STDOUT:     %b.loc7_16.1: i32 = param b
// CHECK:STDOUT:     @Foo.%b: i32 = bind_name b, %b.loc7_16.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: Main = fn_decl @Main [template = constants.%struct.3] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%a: i32, %b: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Foo.ref.loc10: Foo = name_ref Foo, file.%Foo.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc10_7: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_10: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %Foo.call.loc10: init () = call %Foo.ref.loc10(%.loc10_7, %.loc10_10)
// CHECK:STDOUT:   %Foo.ref.loc11: Foo = name_ref Foo, file.%Foo.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc11_7: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_10: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %Foo.call.loc11: init () = call %Foo.ref.loc11(%.loc11_7, %.loc11_10)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
