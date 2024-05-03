// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() {
  var x: () = ();
  x;
}

// CHECK:STDOUT: --- lookup.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main: type = fn_type @Main [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Main = struct_value () [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: Main = fn_decl @Main [template = constants.%struct] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_11.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc8_11.2: type = converted %.loc8_11.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %x.var: ref () = var x
// CHECK:STDOUT:   %x: ref () = bind_name x, %x.var
// CHECK:STDOUT:   %.loc8_16.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc8_16.2: init () = tuple_init () to %x.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc8_17: init () = converted %.loc8_16.1, %.loc8_16.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign %x.var, %.loc8_17
// CHECK:STDOUT:   %x.ref: ref () = name_ref x, %x
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
