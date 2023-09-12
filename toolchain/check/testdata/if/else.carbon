// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "else.carbon" {

fn F() {}
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @F
fn G() {}
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @G
fn H() {}
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @H

fn If(b: bool) {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @If
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
  if (b) {
    F();
    // CHECK:STDOUT:   %.loc[[@LINE-1]]_6.1: type = tuple_type ()
    // CHECK:STDOUT:   %.loc[[@LINE-2]]_6.2: init () = call @F()
    // CHECK:STDOUT:   br !if.done
    // CHECK:STDOUT:
    // CHECK:STDOUT: !if.else:
  } else {
    G();
    // CHECK:STDOUT:   %.loc[[@LINE-1]]: init () = call @G()
    // CHECK:STDOUT:   br !if.done
    // CHECK:STDOUT:
    // CHECK:STDOUT: !if.done:
  }
  H();
  // CHECK:STDOUT:   %.loc[[@LINE-1]]: init () = call @H()
  // CHECK:STDOUT:   return
  // CHECK:STDOUT: }
}
