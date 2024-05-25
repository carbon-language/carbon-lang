// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn A() -> () { return (); }

alias B = A;

fn Main() {
  var b: () = B();
}

// CHECK:STDOUT: --- alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %struct.1: A = struct_value () [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT:   %Main: type = fn_type @Main [template]
// CHECK:STDOUT:   %struct.2: Main = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: A = fn_decl @A [template = constants.%struct.1] {
// CHECK:STDOUT:     %.loc7_12.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc7_12.2: type = converted %.loc7_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     @A.%return: ref () = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.ref: A = name_ref A, %A.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %B: A = bind_alias B, %A.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %Main.decl: Main = fn_decl @Main [template = constants.%struct.2] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() -> () {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_24: () = tuple_literal ()
// CHECK:STDOUT:   %tuple: () = tuple_value () [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_25: () = converted %.loc7_24, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   return %.loc7_25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_11.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc12_11.2: type = converted %.loc12_11.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %b.var: ref () = var b
// CHECK:STDOUT:   %b: ref () = bind_name b, %b.var
// CHECK:STDOUT:   %B.ref: A = name_ref B, file.%B [template = constants.%struct.1]
// CHECK:STDOUT:   %A.call: init () = call %B.ref()
// CHECK:STDOUT:   assign %b.var, %A.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
