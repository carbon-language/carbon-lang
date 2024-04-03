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

// CHECK:STDERR: fail_redefinition.carbon:[[@LINE+7]]:1: ERROR: Redefinition of class Class.
// CHECK:STDERR: class Class {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE-9]]:1: Previous definition was here.
// CHECK:STDERR: class Class {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
class Class {
  fn G();
  // CHECK:STDERR: fail_redefinition.carbon:[[@LINE+7]]:3: ERROR: Redundant redeclaration of function H.
  // CHECK:STDERR:   fn H();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_redefinition.carbon:[[@LINE-16]]:3: Previously declared here.
  // CHECK:STDERR:   fn H();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  fn H();
  // CHECK:STDERR: fail_redefinition.carbon:[[@LINE+7]]:3: ERROR: Redefinition of function I.
  // CHECK:STDERR:   fn I() {}
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_redefinition.carbon:[[@LINE-23]]:3: Previously defined here.
  // CHECK:STDERR:   fn I() {}
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  fn I() {}
}

fn Class.F() {}
fn Class.G() {}
fn Class.H() {}
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE+6]]:1: ERROR: Redefinition of function I.
// CHECK:STDERR: fn Class.I() {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE-9]]:3: Previously defined here.
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
// CHECK:STDOUT:     .Class = %Class.decl.loc7
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.decl.loc7: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %Class.decl.loc20: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {}
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template] {}
// CHECK:STDOUT:   %H: <function> = fn_decl @H [template] {}
// CHECK:STDOUT:   %I: <function> = fn_decl @I [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template] {}
// CHECK:STDOUT:   %H: <function> = fn_decl @H [template] {}
// CHECK:STDOUT:   %I: <function> = fn_decl @I [template] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = <unexpected instref inst+3>
// CHECK:STDOUT:   .H = <unexpected instref inst+4>
// CHECK:STDOUT:   .I = <unexpected instref inst+5>
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
// CHECK:STDOUT: fn @I() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc37:
// CHECK:STDOUT:   return
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc49:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
