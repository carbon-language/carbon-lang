// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn F();
  fn H();
  fn I() {}
}

// CHECK:STDERR: fail_redefinition.carbon:[[@LINE+7]]:1: ERROR: Redefinition of `class Class`.
// CHECK:STDERR: class Class {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE-9]]:1: Previously defined here.
// CHECK:STDERR: class Class {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
class Class {
  fn G();
  fn H();
  fn I() {}
}

// CHECK:STDERR: fail_redefinition.carbon:[[@LINE+4]]:10: ERROR: Out-of-line declaration requires a declaration in scoped entity.
// CHECK:STDERR: fn Class.F() {}
// CHECK:STDERR:          ^
// CHECK:STDERR:
fn Class.F() {}
fn Class.G() {}
fn Class.H() {}
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE+6]]:1: ERROR: Redefinition of `fn I`.
// CHECK:STDERR: fn Class.I() {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE-13]]:3: Previously defined here.
// CHECK:STDERR:   fn I() {}
// CHECK:STDERR:   ^~~~~~~~
fn Class.I() {}

// CHECK:STDOUT: --- fail_redefinition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl.loc7
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl.loc7: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %Class.decl.loc20: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %F: <function> = fn_decl @F.2 [template] {}
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template] {}
// CHECK:STDOUT:   %H: <function> = fn_decl @H.2 [template] {}
// CHECK:STDOUT:   %I: <function> = fn_decl @I.2 [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template] {}
// CHECK:STDOUT:   %H: <function> = fn_decl @H.2 [template] {}
// CHECK:STDOUT:   %I: <function> = fn_decl @I.2 [template] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT:   .H = %H
// CHECK:STDOUT:   .I = %I
// CHECK:STDOUT:   .F = file.%F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc39:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
