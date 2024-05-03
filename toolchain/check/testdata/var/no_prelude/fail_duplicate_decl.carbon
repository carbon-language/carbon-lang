// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE


fn Main() {
  var x: () = ();
  // CHECK:STDERR: fail_duplicate_decl.carbon:[[@LINE+6]]:7: ERROR: Duplicate name being declared in the same scope.
  // CHECK:STDERR:   var x: () = ();
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_duplicate_decl.carbon:[[@LINE-4]]:7: Name is previously declared here.
  // CHECK:STDERR:   var x: () = ();
  // CHECK:STDERR:       ^
  var x: () = ();
}

// CHECK:STDOUT: --- fail_duplicate_decl.carbon
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
// CHECK:STDOUT:   %.loc9_11.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_11.2: type = converted %.loc9_11.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %x.var.loc9: ref () = var x
// CHECK:STDOUT:   %x.loc9: ref () = bind_name x, %x.var.loc9
// CHECK:STDOUT:   %.loc9_16.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_16.2: init () = tuple_init () to %x.var.loc9 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc9_17: init () = converted %.loc9_16.1, %.loc9_16.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign %x.var.loc9, %.loc9_17
// CHECK:STDOUT:   %.loc16_11.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc16_11.2: type = converted %.loc16_11.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %x.var.loc16: ref () = var x
// CHECK:STDOUT:   %x.loc16: ref () = bind_name x, %x.var.loc16
// CHECK:STDOUT:   %.loc16_16.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc16_16.2: init () = tuple_init () to %x.var.loc16 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc16_17: init () = converted %.loc16_16.1, %.loc16_16.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign %x.var.loc16, %.loc16_17
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
