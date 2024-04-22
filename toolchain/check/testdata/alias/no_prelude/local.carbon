// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> () {
  var a: () = ();
  alias b = a;
  return b;
}

// CHECK:STDOUT: --- local.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: () = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %.loc7_12.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc7_12.2: type = converted %.loc7_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     @F.%return: ref () = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> () {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_11.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc8_11.2: type = converted %.loc8_11.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %.loc8_16.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc8_16.2: init () = tuple_init () to %a.var [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_17: init () = converted %.loc8_16.1, %.loc8_16.2 [template = constants.%.2]
// CHECK:STDOUT:   assign %a.var, %.loc8_17
// CHECK:STDOUT:   %a.ref: ref () = name_ref a, %a
// CHECK:STDOUT:   %b: ref () = bind_alias b, %a
// CHECK:STDOUT:   %b.ref: ref () = name_ref b, %b
// CHECK:STDOUT:   %.loc10_10: () = tuple_value () [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_11: () = converted %b.ref, %.loc10_10 [template = constants.%.2]
// CHECK:STDOUT:   return %.loc10_11
// CHECK:STDOUT: }
// CHECK:STDOUT:
