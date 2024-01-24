// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn F();
  fn H();
}

// CHECK:STDERR: fail_redefinition.carbon:[[@LINE+6]]:1: ERROR: Redefinition of class Class.
// CHECK:STDERR: class Class {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE-8]]:1: Previous definition was here.
// CHECK:STDERR: class Class {
// CHECK:STDERR: ^~~~~~~~~~~~~
class Class {
  fn G();
  fn H();
}

fn Class.F() {}
fn Class.G() {}
fn Class.H() {}

// CHECK:STDOUT: --- fail_redefinition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Class = %Class.decl.loc7} [template]
// CHECK:STDOUT:   %Class.decl.loc7 = class_decl @Class, ()
// CHECK:STDOUT:   %Class.decl.loc18 = class_decl @Class, ()
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template]
// CHECK:STDOUT:   %H: <function> = fn_decl @H [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template]
// CHECK:STDOUT:   %H: <function> = fn_decl @H [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = <unexpected instref inst+3>
// CHECK:STDOUT:   .H = <unexpected instref inst+4>
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
