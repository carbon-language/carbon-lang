// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "control_flow.carbon" {

fn A() -> i32 { return 1; }
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @A
fn B() -> i32 { return 2; }
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @B

fn F(b: bool) -> i32 {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc[[@LINE-11]]: i32 = int_literal 1
// CHECK:STDOUT:   return %.loc[[@LINE-12]]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc[[@LINE-15]]: i32 = int_literal 2
// CHECK:STDOUT:   return %.loc[[@LINE-16]]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%b: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   if %b br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
  return if b then A() else B();
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_21.1: init i32 = call @A()
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_21.2: ref i32 = temporary_storage
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_21.3: ref i32 = temporary %.loc[[@LINE-3]]_21.2, %.loc[[@LINE-3]]_21.1
  // CHECK:STDOUT:   %.loc[[@LINE-4]]_21.4: i32 = bind_value %.loc[[@LINE-4]]_21.3
  // CHECK:STDOUT:   br !if.expr.result(%.loc[[@LINE-5]]_21.4)
  // CHECK:STDOUT:
  // CHECK:STDOUT: !if.expr.else:
  // CHECK:STDOUT:   %.loc[[@LINE-8]]_30.1: init i32 = call @B()
  // CHECK:STDOUT:   %.loc[[@LINE-9]]_30.2: ref i32 = temporary_storage
  // CHECK:STDOUT:   %.loc[[@LINE-10]]_30.3: ref i32 = temporary %.loc[[@LINE-10]]_30.2, %.loc[[@LINE-10]]_30.1
  // CHECK:STDOUT:   %.loc[[@LINE-11]]_30.4: i32 = bind_value %.loc[[@LINE-11]]_30.3
  // CHECK:STDOUT:   br !if.expr.result(%.loc[[@LINE-12]]_30.4)
  // CHECK:STDOUT:
  // CHECK:STDOUT: !if.expr.result:
  // CHECK:STDOUT:   %.loc[[@LINE-15]]_10: i32 = block_arg !if.expr.result
  // CHECK:STDOUT:   return %.loc[[@LINE-16]]_10
  // CHECK:STDOUT: }
}
