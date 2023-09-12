// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "basic.carbon" {

fn F(b: bool, n: i32, m: i32) -> i32 {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%b: bool, %n: i32, %m: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   if %b br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
  return if b then n + m else m + n;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_22: i32 = add %n, %m
  // CHECK:STDOUT:   br !if.expr.result(%.loc[[@LINE-2]]_22)
  // CHECK:STDOUT:
  // CHECK:STDOUT: !if.expr.else:
  // CHECK:STDOUT:   %.loc[[@LINE-5]]_33: i32 = add %m, %n
  // CHECK:STDOUT:   br !if.expr.result(%.loc[[@LINE-6]]_33)
  // CHECK:STDOUT:
  // CHECK:STDOUT: !if.expr.result:
  // CHECK:STDOUT:   %.loc[[@LINE-9]]_10: i32 = block_arg !if.expr.result
  // CHECK:STDOUT:   return %.loc[[@LINE-10]]_10
  // CHECK:STDOUT: }
}
