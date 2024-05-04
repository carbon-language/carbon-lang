// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Echo(a: i32) -> i32 {
  return a;
}

fn Main() {
  var b: i32 = Echo(1);
}

// CHECK:STDOUT: --- i32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Echo: type = fn_type @Echo [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Echo = struct_value () [template]
// CHECK:STDOUT:   %Main: type = fn_type @Main [template]
// CHECK:STDOUT:   %struct.2: Main = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Echo = %Echo.decl
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Echo.decl: Echo = fn_decl @Echo [template = constants.%struct.1] {
// CHECK:STDOUT:     %a.loc7_9.1: i32 = param a
// CHECK:STDOUT:     @Echo.%a: i32 = bind_name a, %a.loc7_9.1
// CHECK:STDOUT:     @Echo.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: Main = fn_decl @Main [template = constants.%struct.2] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Echo(%a: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   return %a.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %Echo.ref: Echo = name_ref Echo, file.%Echo.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc12: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %Echo.call: init i32 = call %Echo.ref(%.loc12)
// CHECK:STDOUT:   assign %b.var, %Echo.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
