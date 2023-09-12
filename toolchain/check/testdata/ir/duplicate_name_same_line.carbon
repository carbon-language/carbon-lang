// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "duplicate_name_same_line.carbon" {

fn A() { var n: i32 = 1; if (true) { var n: i32 = 2; } }
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.loc[[@LINE-6]]_14: ref i32 = var "n"
// CHECK:STDOUT:   %.loc[[@LINE-7]]_23: i32 = int_literal 1
// CHECK:STDOUT:   assign %n.loc[[@LINE-8]]_14, %.loc[[@LINE-8]]_23
// CHECK:STDOUT:   %.loc[[@LINE-9]]_30: bool = bool_literal true
// CHECK:STDOUT:   if %.loc[[@LINE-10]]_30 br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %n.loc[[@LINE-13]]_42: ref i32 = var "n"
// CHECK:STDOUT:   %.loc[[@LINE-14]]_51: i32 = int_literal 2
// CHECK:STDOUT:   assign %n.loc[[@LINE-15]]_42, %.loc[[@LINE-15]]_51
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
