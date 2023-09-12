// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "constant_condition.carbon" {

fn A() -> i32 { return 1; }
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @A
fn B() -> i32 { return 2; }
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @B

fn F() -> i32 {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @F
  return if true then A() else B();
}

fn G() -> i32 {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc[[@LINE-16]]: i32 = int_literal 1
// CHECK:STDOUT:   return %.loc[[@LINE-17]]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc[[@LINE-20]]: i32 = int_literal 2
// CHECK:STDOUT:   return %.loc[[@LINE-21]]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc[[@LINE-21]]_13: bool = bool_literal true
// CHECK:STDOUT:   if %.loc[[@LINE-22]]_13 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %.loc[[@LINE-25]]_24.1: init i32 = call @A()
// CHECK:STDOUT:   %.loc[[@LINE-26]]_24.2: ref i32 = temporary_storage
// CHECK:STDOUT:   %.loc[[@LINE-27]]_24.3: ref i32 = temporary %.loc[[@LINE-27]]_24.2, %.loc[[@LINE-27]]_24.1
// CHECK:STDOUT:   %.loc[[@LINE-28]]_24.4: i32 = bind_value %.loc[[@LINE-28]]_24.3
// CHECK:STDOUT:   br !if.expr.result(%.loc[[@LINE-29]]_24.4)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %.loc[[@LINE-32]]_33.1: init i32 = call @B()
// CHECK:STDOUT:   %.loc[[@LINE-33]]_33.2: ref i32 = temporary_storage
// CHECK:STDOUT:   %.loc[[@LINE-34]]_33.3: ref i32 = temporary %.loc[[@LINE-34]]_33.2, %.loc[[@LINE-34]]_33.1
// CHECK:STDOUT:   %.loc[[@LINE-35]]_33.4: i32 = bind_value %.loc[[@LINE-35]]_33.3
// CHECK:STDOUT:   br !if.expr.result(%.loc[[@LINE-36]]_33.4)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc[[@LINE-39]]_10: i32 = block_arg !if.expr.result
// CHECK:STDOUT:   return %.loc[[@LINE-40]]_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> i32 {
// CHECK:STDOUT: !entry:
  return if false then A() else B();
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_13: bool = bool_literal false
  // CHECK:STDOUT:   if %.loc[[@LINE-2]]_13 br !if.expr.then else br !if.expr.else
  // CHECK:STDOUT:
  // CHECK:STDOUT: !if.expr.then:
  // CHECK:STDOUT:   %.loc[[@LINE-5]]_25.1: init i32 = call @A()
  // CHECK:STDOUT:   %.loc[[@LINE-6]]_25.2: ref i32 = temporary_storage
  // CHECK:STDOUT:   %.loc[[@LINE-7]]_25.3: ref i32 = temporary %.loc[[@LINE-7]]_25.2, %.loc[[@LINE-7]]_25.1
  // CHECK:STDOUT:   %.loc[[@LINE-8]]_25.4: i32 = bind_value %.loc[[@LINE-8]]_25.3
  // CHECK:STDOUT:   br !if.expr.result(%.loc[[@LINE-9]]_25.4)
  // CHECK:STDOUT:
  // CHECK:STDOUT: !if.expr.else:
  // CHECK:STDOUT:   %.loc[[@LINE-12]]_34.1: init i32 = call @B()
  // CHECK:STDOUT:   %.loc[[@LINE-13]]_34.2: ref i32 = temporary_storage
  // CHECK:STDOUT:   %.loc[[@LINE-14]]_34.3: ref i32 = temporary %.loc[[@LINE-14]]_34.2, %.loc[[@LINE-14]]_34.1
  // CHECK:STDOUT:   %.loc[[@LINE-15]]_34.4: i32 = bind_value %.loc[[@LINE-15]]_34.3
  // CHECK:STDOUT:   br !if.expr.result(%.loc[[@LINE-16]]_34.4)
  // CHECK:STDOUT:
  // CHECK:STDOUT: !if.expr.result:
  // CHECK:STDOUT:   %.loc[[@LINE-19]]_10: i32 = block_arg !if.expr.result
  // CHECK:STDOUT:   return %.loc[[@LINE-20]]_10
  // CHECK:STDOUT: }
}
