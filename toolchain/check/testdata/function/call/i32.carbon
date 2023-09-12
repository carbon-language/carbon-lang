// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "i32.carbon" {

fn Echo(a: i32) -> i32 {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Echo
  return a;
}

fn Main() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Echo(%a: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return %a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b: ref i32 = var "b"
  var b: i32 = Echo(1);
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_21.1: i32 = int_literal 1
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_21.2: i32 = stub_reference %.loc[[@LINE-2]]_21.1
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_20: init i32 = call @Echo(%.loc[[@LINE-3]]_21.2)
  // CHECK:STDOUT:   assign %b, %.loc[[@LINE-4]]_20
  // CHECK:STDOUT:   return
  // CHECK:STDOUT: }
}
