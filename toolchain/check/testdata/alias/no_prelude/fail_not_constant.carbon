// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  var a: () = ();
  var b: ()* = &a;
  // CHECK:STDERR: fail_not_constant.carbon:[[@LINE+3]]:13: ERROR: Alias initializer must be a name reference.
  // CHECK:STDERR:   alias c = *b;
  // CHECK:STDERR:             ^~
  alias c = *b;
}

// CHECK:STDOUT: --- fail_not_constant.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: F = struct_value () [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT:   %.2: type = ptr_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_11.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc8_11.2: type = converted %.loc8_11.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %.loc8_16.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc8_16.2: init () = tuple_init () to %a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc8_17: init () = converted %.loc8_16.1, %.loc8_16.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign %a.var, %.loc8_17
// CHECK:STDOUT:   %.loc9_11: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_12.1: type = converted %.loc9_11, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc9_12.2: type = ptr_type () [template = constants.%.2]
// CHECK:STDOUT:   %b.var: ref ()* = var b
// CHECK:STDOUT:   %b: ref ()* = bind_name b, %b.var
// CHECK:STDOUT:   %a.ref: ref () = name_ref a, %a
// CHECK:STDOUT:   %.loc9_16: ()* = addr_of %a.ref
// CHECK:STDOUT:   assign %b.var, %.loc9_16
// CHECK:STDOUT:   %b.ref: ref ()* = name_ref b, %b
// CHECK:STDOUT:   %.loc13_14: ()* = bind_value %b.ref
// CHECK:STDOUT:   %.loc13_13: ref () = deref %.loc13_14
// CHECK:STDOUT:   %c: <error> = bind_alias c, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
