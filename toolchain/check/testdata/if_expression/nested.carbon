// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(a: bool, b: bool, c: bool) -> i32 {
  return if a then if b then 1 else 2 else if c then 3 else 4;
}

// CHECK:STDOUT: file "nested.carbon" {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a: bool, %b: bool, %c: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: bool = name_reference "a", %a
// CHECK:STDOUT:   if %a.ref br !if.expr.then.loc8_10 else br !if.expr.else.loc8_10
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc8_10:
// CHECK:STDOUT:   %b.ref: bool = name_reference "b", %b
// CHECK:STDOUT:   if %b.ref br !if.expr.then.loc8_20 else br !if.expr.else.loc8_20
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc8_20:
// CHECK:STDOUT:   %.loc8_30: i32 = int_literal 1
// CHECK:STDOUT:   br !if.expr.result.loc8_20(%.loc8_30)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc8_20:
// CHECK:STDOUT:   %.loc8_37: i32 = int_literal 2
// CHECK:STDOUT:   br !if.expr.result.loc8_20(%.loc8_37)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc8_20:
// CHECK:STDOUT:   %.loc8_20: i32 = block_arg !if.expr.result.loc8_20
// CHECK:STDOUT:   br !if.expr.result.loc8_10(%.loc8_20)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc8_10:
// CHECK:STDOUT:   %c.ref: bool = name_reference "c", %c
// CHECK:STDOUT:   if %c.ref br !if.expr.then.loc8_44 else br !if.expr.else.loc8_44
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc8_44:
// CHECK:STDOUT:   %.loc8_54: i32 = int_literal 3
// CHECK:STDOUT:   br !if.expr.result.loc8_44(%.loc8_54)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc8_44:
// CHECK:STDOUT:   %.loc8_61: i32 = int_literal 4
// CHECK:STDOUT:   br !if.expr.result.loc8_44(%.loc8_61)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc8_44:
// CHECK:STDOUT:   %.loc8_44: i32 = block_arg !if.expr.result.loc8_44
// CHECK:STDOUT:   br !if.expr.result.loc8_10(%.loc8_44)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc8_10:
// CHECK:STDOUT:   %.loc8_10: i32 = block_arg !if.expr.result.loc8_10
// CHECK:STDOUT:   return %.loc8_10
// CHECK:STDOUT: }
