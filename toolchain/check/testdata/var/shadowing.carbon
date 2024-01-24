// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() {
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
// CHECK:STDOUT:   %.2: bool = bool_literal true [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Main = %Main} [template]
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var.loc8: ref i32 = var x
// CHECK:STDOUT:   %x.loc8: ref i32 = bind_name x, %x.var.loc8
// CHECK:STDOUT:   %.loc8: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   assign %x.var.loc8, %.loc8
// CHECK:STDOUT:   %.loc9: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   if %.loc9 br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %x.var.loc10: ref i32 = var x
// CHECK:STDOUT:   %x.loc10: ref i32 = bind_name x, %x.var.loc10
// CHECK:STDOUT:   %.loc10: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   assign %x.var.loc10, %.loc10
// CHECK:STDOUT:   %x.ref: ref i32 = name_ref x, %x.loc10
// CHECK:STDOUT:   %.loc13: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   assign %x.ref, %.loc13
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
