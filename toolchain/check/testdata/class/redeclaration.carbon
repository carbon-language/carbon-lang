// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class;

class Class {
  fn F();
}

fn Class.F() {}

// CHECK:STDOUT: --- redeclaration.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc11: type = struct_type {}, const
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.Class = %Class.decl.loc7}
// CHECK:STDOUT:   %Class.decl.loc7 = class_decl @Class, ()
// CHECK:STDOUT:   %Class: type = class_type @Class, const
// CHECK:STDOUT:   %Class.decl.loc9 = class_decl @Class, ()
// CHECK:STDOUT:   %F: <function> = fn_decl @F, const
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F: <function> = fn_decl @F, const
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
