// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(b: bool, n: i32, m: i32) -> i32 {
  return if b then n + m else m + n;
}

// CHECK:STDOUT: file "basic.carbon" {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%b: bool, %n: i32, %m: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_reference "b", %b
// CHECK:STDOUT:   if %b.ref br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %n.ref.loc8_20: i32 = name_reference "n", %n
// CHECK:STDOUT:   %m.ref.loc8_24: i32 = name_reference "m", %m
// CHECK:STDOUT:   %.loc8_22: i32 = add %n.ref.loc8_20, %m.ref.loc8_24
// CHECK:STDOUT:   br !if.expr.result(%.loc8_22)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %m.ref.loc8_31: i32 = name_reference "m", %m
// CHECK:STDOUT:   %n.ref.loc8_35: i32 = name_reference "n", %n
// CHECK:STDOUT:   %.loc8_33: i32 = add %m.ref.loc8_31, %n.ref.loc8_35
// CHECK:STDOUT:   br !if.expr.result(%.loc8_33)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc8_10: i32 = block_arg !if.expr.result
// CHECK:STDOUT:   return %.loc8_10
// CHECK:STDOUT: }
