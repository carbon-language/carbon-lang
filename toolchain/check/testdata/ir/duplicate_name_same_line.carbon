// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn A() { if (true) { var n: i32 = 1; } if (true) { var n: i32 = 2; } }

// CHECK:STDOUT: --- duplicate_name_same_line.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: bool = bool_literal true [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.A = %A} [template]
// CHECK:STDOUT:   %A: <function> = fn_decl @A [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_14: bool = bool_literal true [template = constants.%.1]
// CHECK:STDOUT:   if %.loc7_14 br !if.then.loc7_18 else br !if.else.loc7_18
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc7_18:
// CHECK:STDOUT:   %n.var.loc7_26: ref i32 = var n
// CHECK:STDOUT:   %n.loc7_26: ref i32 = bind_name n, %n.var.loc7_26
// CHECK:STDOUT:   %.loc7_35: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   assign %n.var.loc7_26, %.loc7_35
// CHECK:STDOUT:   br !if.else.loc7_18
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc7_18:
// CHECK:STDOUT:   %.loc7_44: bool = bool_literal true [template = constants.%.1]
// CHECK:STDOUT:   if %.loc7_44 br !if.then.loc7_48 else br !if.else.loc7_48
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc7_48:
// CHECK:STDOUT:   %n.var.loc7_56: ref i32 = var n
// CHECK:STDOUT:   %n.loc7_56: ref i32 = bind_name n, %n.var.loc7_56
// CHECK:STDOUT:   %.loc7_65: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   assign %n.var.loc7_56, %.loc7_65
// CHECK:STDOUT:   br !if.else.loc7_48
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc7_48:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
