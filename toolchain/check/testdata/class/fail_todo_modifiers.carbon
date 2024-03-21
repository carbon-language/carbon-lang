// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Access {

  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+4]]:3: ERROR: Semantics TODO: `access modifier`.
  // CHECK:STDERR:   private fn F();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  private fn F();

  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+4]]:3: ERROR: Semantics TODO: `access modifier`.
  // CHECK:STDERR:   protected fn G();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  protected fn G();

  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+4]]:3: ERROR: Semantics TODO: `access modifier`.
  // CHECK:STDERR:   private var k: i32;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  private var k: i32;

  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+4]]:3: ERROR: Semantics TODO: `access modifier`.
  // CHECK:STDERR:   protected var l: i32;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  protected var l: i32;
}

base class Base {

  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+4]]:3: ERROR: Semantics TODO: `method modifier`.
  // CHECK:STDERR:   virtual fn H();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  virtual fn H();

  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+4]]:3: ERROR: Semantics TODO: `method modifier`.
  // CHECK:STDERR:   impl fn I();
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  impl fn I();
}

abstract class Abstract {

  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+4]]:3: ERROR: Semantics TODO: `method modifier`.
  // CHECK:STDERR:   abstract fn J();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  abstract fn J();

  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+4]]:3: ERROR: Semantics TODO: `method modifier`.
  // CHECK:STDERR:   virtual fn K();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  virtual fn K();

  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+3]]:3: ERROR: Semantics TODO: `method modifier`.
  // CHECK:STDERR:   impl fn L();
  // CHECK:STDERR:   ^~~~
  impl fn L();
}

// CHECK:STDOUT: --- fail_todo_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Access: type = class_type @Access [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type Access, i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.k: i32, .l: i32} [template]
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %Abstract: type = class_type @Abstract [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Access = %Access.decl
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Abstract = %Abstract.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Access.decl: type = class_decl @Access [template = constants.%Access] {}
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {}
// CHECK:STDOUT:   %Abstract.decl: type = class_decl @Abstract [template = constants.%Abstract] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Access {
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {}
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template] {}
// CHECK:STDOUT:   %.loc25: <unbound element of class Access> = field_decl k, element0 [template]
// CHECK:STDOUT:   %.loc31: <unbound element of class Access> = field_decl l, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Access
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT:   .k = %.loc25
// CHECK:STDOUT:   .l = %.loc31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %H: <function> = fn_decl @H [template] {}
// CHECK:STDOUT:   %I: <function> = fn_decl @I [template] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .H = %H
// CHECK:STDOUT:   .I = %I
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract {
// CHECK:STDOUT:   %J: <function> = fn_decl @J [template] {}
// CHECK:STDOUT:   %K: <function> = fn_decl @K [template] {}
// CHECK:STDOUT:   %L: <function> = fn_decl @L [template] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Abstract
// CHECK:STDOUT:   .J = %J
// CHECK:STDOUT:   .K = %K
// CHECK:STDOUT:   .L = %L
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @J();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @K();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @L();
// CHECK:STDOUT:
