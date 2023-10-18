// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn F();
}

// CHECK:STDERR: redefinition.carbon:[[@LINE+6]]:13: ERROR: Redefinition of class Class.
// CHECK:STDERR: class Class {
// CHECK:STDERR:             ^
// CHECK:STDERR: redefinition.carbon:[[@LINE-7]]:1: Previous definition was here.
// CHECK:STDERR: class Class {
// CHECK:STDERR: ^
class Class {
  fn G();
}

fn Class.F() {}

// CHECK:STDOUT: file "redefinition.carbon" {
// CHECK:STDOUT:   %Class.loc7: type = class_declaration @Class, ()
// CHECK:STDOUT:   %Class.loc17: type = class_declaration @Class, ()
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = <unexpected noderef 9>
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G();
