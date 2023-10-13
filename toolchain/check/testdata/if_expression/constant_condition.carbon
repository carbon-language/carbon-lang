// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn A() -> i32 { return 1; }
fn B() -> i32 { return 2; }

fn F() -> i32 {
  return if true then A() else B();
}

fn G() -> i32 {
  return if false then A() else B();
}

// CHECK:STDOUT: file "constant_condition.carbon" {
// CHECK:STDOUT:   %A: <function> = fn_decl @A
// CHECK:STDOUT:   %B: <function> = fn_decl @B
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7: i32 = int_literal 1
// CHECK:STDOUT:   return %.loc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8: i32 = int_literal 2
// CHECK:STDOUT:   return %.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_13: bool = bool_literal true
// CHECK:STDOUT:   if %.loc11_13 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %A.ref: <function> = name_reference "A", package.%A
// CHECK:STDOUT:   %.loc11_24.1: init i32 = call %A.ref()
// CHECK:STDOUT:   %.loc11_24.2: ref i32 = temporary_storage
// CHECK:STDOUT:   %.loc11_24.3: ref i32 = temporary %.loc11_24.2, %.loc11_24.1
// CHECK:STDOUT:   %.loc11_24.4: i32 = bind_value %.loc11_24.3
// CHECK:STDOUT:   br !if.expr.result(%.loc11_24.4)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %B.ref: <function> = name_reference "B", package.%B
// CHECK:STDOUT:   %.loc11_33.1: init i32 = call %B.ref()
// CHECK:STDOUT:   %.loc11_33.2: ref i32 = temporary_storage
// CHECK:STDOUT:   %.loc11_33.3: ref i32 = temporary %.loc11_33.2, %.loc11_33.1
// CHECK:STDOUT:   %.loc11_33.4: i32 = bind_value %.loc11_33.3
// CHECK:STDOUT:   br !if.expr.result(%.loc11_33.4)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc11_10: i32 = block_arg !if.expr.result
// CHECK:STDOUT:   return %.loc11_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_13: bool = bool_literal false
// CHECK:STDOUT:   if %.loc15_13 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %A.ref: <function> = name_reference "A", package.%A
// CHECK:STDOUT:   %.loc15_25.1: init i32 = call %A.ref()
// CHECK:STDOUT:   %.loc15_25.2: ref i32 = temporary_storage
// CHECK:STDOUT:   %.loc15_25.3: ref i32 = temporary %.loc15_25.2, %.loc15_25.1
// CHECK:STDOUT:   %.loc15_25.4: i32 = bind_value %.loc15_25.3
// CHECK:STDOUT:   br !if.expr.result(%.loc15_25.4)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %B.ref: <function> = name_reference "B", package.%B
// CHECK:STDOUT:   %.loc15_34.1: init i32 = call %B.ref()
// CHECK:STDOUT:   %.loc15_34.2: ref i32 = temporary_storage
// CHECK:STDOUT:   %.loc15_34.3: ref i32 = temporary %.loc15_34.2, %.loc15_34.1
// CHECK:STDOUT:   %.loc15_34.4: i32 = bind_value %.loc15_34.3
// CHECK:STDOUT:   br !if.expr.result(%.loc15_34.4)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc15_10: i32 = block_arg !if.expr.result
// CHECK:STDOUT:   return %.loc15_10
// CHECK:STDOUT: }
