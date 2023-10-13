// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn A() -> i32 { return 1; }
fn B() -> i32 { return 2; }

fn F(b: bool) -> i32 {
  return if b then A() else B();
}

// CHECK:STDOUT: file "control_flow.carbon" {
// CHECK:STDOUT:   %A: <function> = fn_decl @A
// CHECK:STDOUT:   %B: <function> = fn_decl @B
// CHECK:STDOUT:   %F: <function> = fn_decl @F
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
// CHECK:STDOUT: fn @F(%b: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_reference "b", %b
// CHECK:STDOUT:   if %b.ref br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %A.ref: <function> = name_reference "A", package.%A
// CHECK:STDOUT:   %.loc11_21.1: init i32 = call %A.ref()
// CHECK:STDOUT:   %.loc11_21.2: ref i32 = temporary_storage
// CHECK:STDOUT:   %.loc11_21.3: ref i32 = temporary %.loc11_21.2, %.loc11_21.1
// CHECK:STDOUT:   %.loc11_21.4: i32 = bind_value %.loc11_21.3
// CHECK:STDOUT:   br !if.expr.result(%.loc11_21.4)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %B.ref: <function> = name_reference "B", package.%B
// CHECK:STDOUT:   %.loc11_30.1: init i32 = call %B.ref()
// CHECK:STDOUT:   %.loc11_30.2: ref i32 = temporary_storage
// CHECK:STDOUT:   %.loc11_30.3: ref i32 = temporary %.loc11_30.2, %.loc11_30.1
// CHECK:STDOUT:   %.loc11_30.4: i32 = bind_value %.loc11_30.3
// CHECK:STDOUT:   br !if.expr.result(%.loc11_30.4)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc11_10: i32 = block_arg !if.expr.result
// CHECK:STDOUT:   return %.loc11_10
// CHECK:STDOUT: }
