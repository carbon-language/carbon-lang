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

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc12: type = tuple_type ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "no_else.carbon" {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT:   %If: <function> = fn_decl @If
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
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, file.%F
// CHECK:STDOUT:   %.loc12: init () = call %F.ref()
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %G.ref: <function> = name_ref G, file.%G
// CHECK:STDOUT:   %.loc14: init () = call %G.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
