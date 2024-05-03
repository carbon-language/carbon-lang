// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

namespace NS;

fn Main() {
  var NS: () = ();
  NS = ();

  var x: () = ();
  if (true) {
    var x: () = ();

    // TODO: We should reject this use of the shadowed variable `x`.
    x = ();
  }
}

// CHECK:STDOUT: --- shadowing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main: type = fn_type @Main [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Main = struct_value () [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT:   %.2: bool = bool_literal true [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Main.decl: Main = fn_decl @Main [template = constants.%struct] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_12.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc10_12.2: type = converted %.loc10_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %NS.var: ref () = var NS
// CHECK:STDOUT:   %NS: ref () = bind_name NS, %NS.var
// CHECK:STDOUT:   %.loc10_17.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc10_17.2: init () = tuple_init () to %NS.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc10_18: init () = converted %.loc10_17.1, %.loc10_17.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign %NS.var, %.loc10_18
// CHECK:STDOUT:   %NS.ref: ref () = name_ref NS, %NS
// CHECK:STDOUT:   %.loc11_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc11_9.2: init () = tuple_init () to %NS.ref [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_6: init () = converted %.loc11_9.1, %.loc11_9.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign %NS.ref, %.loc11_6
// CHECK:STDOUT:   %.loc13_11.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc13_11.2: type = converted %.loc13_11.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %x.var.loc13: ref () = var x
// CHECK:STDOUT:   %x.loc13: ref () = bind_name x, %x.var.loc13
// CHECK:STDOUT:   %.loc13_16.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc13_16.2: init () = tuple_init () to %x.var.loc13 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc13_17: init () = converted %.loc13_16.1, %.loc13_16.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign %x.var.loc13, %.loc13_17
// CHECK:STDOUT:   %.loc14: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   if %.loc14 br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %.loc15_13.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc15_13.2: type = converted %.loc15_13.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %x.var.loc15: ref () = var x
// CHECK:STDOUT:   %x.loc15: ref () = bind_name x, %x.var.loc15
// CHECK:STDOUT:   %.loc15_18.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc15_18.2: init () = tuple_init () to %x.var.loc15 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc15_19: init () = converted %.loc15_18.1, %.loc15_18.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign %x.var.loc15, %.loc15_19
// CHECK:STDOUT:   %x.ref: ref () = name_ref x, %x.loc15
// CHECK:STDOUT:   %.loc18_10.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc18_10.2: init () = tuple_init () to %x.ref [template = constants.%tuple]
// CHECK:STDOUT:   %.loc18_7: init () = converted %.loc18_10.1, %.loc18_10.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign %x.ref, %.loc18_7
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
