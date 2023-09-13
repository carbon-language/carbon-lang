// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {}
fn G() {}
fn H() {}

fn If(b: bool) {
  if (b) {
    F();
  } else {
    G();
  }
  H();
}

// CHECK:STDOUT: file "else.carbon" {
// CHECK:STDOUT:   %.loc7 = fn_decl @F
// CHECK:STDOUT:   %.loc8 = fn_decl @G
// CHECK:STDOUT:   %.loc9 = fn_decl @H
// CHECK:STDOUT:   %.loc11 = fn_decl @If
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @If(%b: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   if %b br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %.loc13_6.1: type = tuple_type ()
// CHECK:STDOUT:   %.loc13_6.2: init () = call @F()
// CHECK:STDOUT:   br !if.done
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %.loc15: init () = call @G()
// CHECK:STDOUT:   br !if.done
// CHECK:STDOUT:
// CHECK:STDOUT: !if.done:
// CHECK:STDOUT:   %.loc17: init () = call @H()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
