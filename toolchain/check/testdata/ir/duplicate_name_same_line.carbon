// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn A() { var n: i32 = 1; if (true) { var n: i32 = 2; } }

// CHECK:STDOUT: file "duplicate_name_same_line.carbon" {
// CHECK:STDOUT:   %.loc7 = fn_decl @A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.loc7_14: ref i32 = var "n"
// CHECK:STDOUT:   %.loc7_23: i32 = int_literal 1
// CHECK:STDOUT:   assign %n.loc7_14, %.loc7_23
// CHECK:STDOUT:   %.loc7_30: bool = bool_literal true
// CHECK:STDOUT:   if %.loc7_30 br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %n.loc7_42: ref i32 = var "n"
// CHECK:STDOUT:   %.loc7_51: i32 = int_literal 2
// CHECK:STDOUT:   assign %n.loc7_42, %.loc7_51
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
