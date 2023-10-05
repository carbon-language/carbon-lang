// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {}
fn G() {}

fn If(b: bool) {
  if (b) {
    F();
  }
  G();
}

// CHECK:STDOUT: file "no_else.carbon" {
// CHECK:STDOUT:   %F = fn_decl @F
// CHECK:STDOUT:   %G = fn_decl @G
// CHECK:STDOUT:   %If = fn_decl @If
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
// CHECK:STDOUT: fn @If(%b: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_reference "b", %b
// CHECK:STDOUT:   if %b.ref br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %F.ref = name_reference_untyped "F", package.%F
// CHECK:STDOUT:   %.loc12_6.1: type = tuple_type ()
// CHECK:STDOUT:   %.loc12_6.2: init () = call @F()
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %G.ref = name_reference_untyped "G", package.%G
// CHECK:STDOUT:   %.loc14: init () = call @G()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
