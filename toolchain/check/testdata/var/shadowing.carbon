// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

namespace NS;

fn Main() {
  var NS: i32 = 0;
  NS = 1;

  var x: i32 = 0;
  if (true) {
    var x: i32 = 0;

    // TODO: We should reject this use of the shadowed variable `x`.
    x = 1;
  }
}

// CHECK:STDOUT: --- shadowing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: bool = bool_literal true [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .Main = %Main
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %NS.var: ref i32 = var NS
// CHECK:STDOUT:   %NS: ref i32 = bind_name NS, %NS.var
// CHECK:STDOUT:   %.loc10: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   assign %NS.var, %.loc10
// CHECK:STDOUT:   %NS.ref: ref i32 = name_ref NS, %NS
// CHECK:STDOUT:   %.loc11: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   assign %NS.ref, %.loc11
// CHECK:STDOUT:   %x.var.loc13: ref i32 = var x
// CHECK:STDOUT:   %x.loc13: ref i32 = bind_name x, %x.var.loc13
// CHECK:STDOUT:   %.loc13: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   assign %x.var.loc13, %.loc13
// CHECK:STDOUT:   %.loc14: bool = bool_literal true [template = constants.%.3]
// CHECK:STDOUT:   if %.loc14 br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %x.var.loc15: ref i32 = var x
// CHECK:STDOUT:   %x.loc15: ref i32 = bind_name x, %x.var.loc15
// CHECK:STDOUT:   %.loc15: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   assign %x.var.loc15, %.loc15
// CHECK:STDOUT:   %x.ref: ref i32 = name_ref x, %x.loc15
// CHECK:STDOUT:   %.loc18: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   assign %x.ref, %.loc18
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
