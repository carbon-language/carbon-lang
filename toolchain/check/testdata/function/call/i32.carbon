// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Echo(a: i32) -> i32 {
  return a;
}

fn Main() {
  var b: i32 = Echo(1);
}

// CHECK:STDOUT: file "i32.carbon" {
// CHECK:STDOUT:   %Echo: <function> = fn_decl @Echo
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Echo(%a: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: i32 = name_reference "a", %a
// CHECK:STDOUT:   return %a.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b: ref i32 = var "b"
// CHECK:STDOUT:   %Echo.ref: <function> = name_reference "Echo", package.%Echo
// CHECK:STDOUT:   %.loc12_21: i32 = int_literal 1
// CHECK:STDOUT:   %.loc12_20: init i32 = call %Echo.ref(%.loc12_21)
// CHECK:STDOUT:   assign %b, %.loc12_20
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
