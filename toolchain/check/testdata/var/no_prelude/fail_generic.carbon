// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() {
  // CHECK:STDERR: fail_generic.carbon:[[@LINE+3]]:11: ERROR: `var` declaration cannot declare a compile-time binding.
  // CHECK:STDERR:   var x:! () = ();
  // CHECK:STDERR:           ^~
  var x:! () = ();
}

// CHECK:STDOUT: --- fail_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main: type = fn_type @Main [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Main = struct_value () [template]
// CHECK:STDOUT:   %x: () = bind_symbolic_name x 0 [symbolic]
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
// CHECK:STDOUT:   %.loc11_12.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc11_12.2: type = converted %.loc11_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %x.var: ref () = var x
// CHECK:STDOUT:   %x: () = bind_symbolic_name x 0, %x.var [symbolic = constants.%x]
// CHECK:STDOUT:   %.loc11_17.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc11_17.2: init () = tuple_init () to %x.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_18: init () = converted %.loc11_17.1, %.loc11_17.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign %x.var, %.loc11_18
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
