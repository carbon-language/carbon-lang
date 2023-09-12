// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "nested.carbon" {

fn F(a: bool, b: bool, c: bool) -> i32 {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a: bool, %b: bool, %c: bool) -> i32 {
// CHECK:STDOUT: !entry:
  return if a then if b then 1 else 2 else if c then 3 else 4;
  // CHECK:STDOUT:   if %a br !if.expr.then.loc[[@LINE-1]]_10 else br !if.expr.else.loc[[@LINE-1]]_10
  // CHECK:STDOUT:
  // CHECK:STDOUT: !if.expr.then.loc[[@LINE-3]]_10:
  // CHECK:STDOUT:   if %b br !if.expr.then.loc[[@LINE-4]]_20 else br !if.expr.else.loc[[@LINE-4]]_20
  // CHECK:STDOUT:
  // CHECK:STDOUT: !if.expr.then.loc[[@LINE-6]]_20:
  // CHECK:STDOUT:   %.loc[[@LINE-7]]_30: i32 = int_literal 1
  // CHECK:STDOUT:   br !if.expr.result.loc[[@LINE-8]]_20(%.loc[[@LINE-8]]_30)
  // CHECK:STDOUT:
  // CHECK:STDOUT: !if.expr.else.loc[[@LINE-10]]_20:
  // CHECK:STDOUT:   %.loc[[@LINE-11]]_37: i32 = int_literal 2
  // CHECK:STDOUT:   br !if.expr.result.loc[[@LINE-12]]_20(%.loc[[@LINE-12]]_37)
  // CHECK:STDOUT:
  // CHECK:STDOUT: !if.expr.result.loc[[@LINE-14]]_20:
  // CHECK:STDOUT:   %.loc[[@LINE-15]]_20: i32 = block_arg !if.expr.result.loc[[@LINE-15]]_20
  // CHECK:STDOUT:   br !if.expr.result.loc[[@LINE-16]]_10(%.loc[[@LINE-16]]_20)
  // CHECK:STDOUT:
  // CHECK:STDOUT: !if.expr.else.loc[[@LINE-18]]_10:
  // CHECK:STDOUT:   if %c br !if.expr.then.loc[[@LINE-19]]_44 else br !if.expr.else.loc[[@LINE-19]]_44
  // CHECK:STDOUT:
  // CHECK:STDOUT: !if.expr.then.loc[[@LINE-21]]_44:
  // CHECK:STDOUT:   %.loc[[@LINE-22]]_54: i32 = int_literal 3
  // CHECK:STDOUT:   br !if.expr.result.loc[[@LINE-23]]_44(%.loc[[@LINE-23]]_54)
  // CHECK:STDOUT:
  // CHECK:STDOUT: !if.expr.else.loc[[@LINE-25]]_44:
  // CHECK:STDOUT:   %.loc[[@LINE-26]]_61: i32 = int_literal 4
  // CHECK:STDOUT:   br !if.expr.result.loc[[@LINE-27]]_44(%.loc[[@LINE-27]]_61)
  // CHECK:STDOUT:
  // CHECK:STDOUT: !if.expr.result.loc[[@LINE-29]]_44:
  // CHECK:STDOUT:   %.loc[[@LINE-30]]_44: i32 = block_arg !if.expr.result.loc[[@LINE-30]]_44
  // CHECK:STDOUT:   br !if.expr.result.loc[[@LINE-31]]_10(%.loc[[@LINE-31]]_44)
  // CHECK:STDOUT:
  // CHECK:STDOUT: !if.expr.result.loc[[@LINE-33]]_10:
  // CHECK:STDOUT:   %.loc[[@LINE-34]]_10: i32 = block_arg !if.expr.result.loc[[@LINE-34]]_10
  // CHECK:STDOUT:   return %.loc[[@LINE-35]]_10
  // CHECK:STDOUT: }
}
