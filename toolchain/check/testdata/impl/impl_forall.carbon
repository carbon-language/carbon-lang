// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface Simple {
  fn F();
}

impl forall [T:! type] T as Simple {
  fn F() {}
}

// CHECK:STDOUT: --- impl_forall.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Simple [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Simple = %Simple.decl} [template]
// CHECK:STDOUT:   %Simple.decl = interface_decl @Simple, ()
// CHECK:STDOUT:   %F: <function> = fn_decl @F.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Simple {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
