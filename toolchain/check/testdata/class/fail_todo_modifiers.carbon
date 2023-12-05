// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Access {

  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+3]]:3: ERROR: Semantics TODO: `access modifier`.
  // CHECK:STDERR:   private fn F();
  // CHECK:STDERR:   ^~~~~~~
  private fn F();

  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+3]]:3: ERROR: Semantics TODO: `access modifier`.
  // CHECK:STDERR:   protected fn G();
  // CHECK:STDERR:   ^~~~~~~~~
  protected fn G();

  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+3]]:3: ERROR: Semantics TODO: `access modifier`.
  // CHECK:STDERR:   private var k: i32;
  // CHECK:STDERR:   ^~~~~~~
  private var k: i32;

  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+3]]:3: ERROR: Semantics TODO: `access modifier`.
  // CHECK:STDERR:   protected var l: i32;
  // CHECK:STDERR:   ^~~~~~~~~
  protected var l: i32;
}

base class Base {

  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+3]]:3: ERROR: Semantics TODO: `method modifier`.
  // CHECK:STDERR:   virtual fn H();
  // CHECK:STDERR:   ^~~~~~~
  virtual fn H();

  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+3]]:3: ERROR: Semantics TODO: `method modifier`.
  // CHECK:STDERR:   impl fn I();
  // CHECK:STDERR:   ^~~~
  impl fn I();
}

abstract class Abstract {

  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+3]]:3: ERROR: Semantics TODO: `method modifier`.
  // CHECK:STDERR:   abstract fn J();
  // CHECK:STDERR:   ^~~~~~~~
  abstract fn J();

  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+3]]:3: ERROR: Semantics TODO: `method modifier`.
  // CHECK:STDERR:   virtual fn K();
  // CHECK:STDERR:   ^~~~~~~
  virtual fn K();

  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+3]]:3: ERROR: Semantics TODO: `method modifier`.
  // CHECK:STDERR:   impl fn L();
  // CHECK:STDERR:   ^~~~
  impl fn L();
}

// CHECK:STDOUT: --- fail_todo_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc28: type = struct_type {.k: i32, .l: i32}
// CHECK:STDOUT:   %.loc41: type = struct_type {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Access = %Access.decl, .Base = %Base.decl, .Abstract = %Abstract.decl}
// CHECK:STDOUT:   %Access.decl = class_decl @Access, ()
// CHECK:STDOUT:   %Access: type = class_type @Access
// CHECK:STDOUT:   %Base.decl = class_decl @Base, ()
// CHECK:STDOUT:   %Base: type = class_type @Base
// CHECK:STDOUT:   %Abstract.decl = class_decl @Abstract, ()
// CHECK:STDOUT:   %Abstract: type = class_type @Abstract
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Access {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT:   %.loc22_16.1: type = unbound_element_type Access, i32
// CHECK:STDOUT:   %.loc22_16.2: <unbound element of class Access> = field_decl k, element0
// CHECK:STDOUT:   %k: <unbound element of class Access> = bind_name k, %.loc22_16.2
// CHECK:STDOUT:   %.loc27_18.1: type = unbound_element_type Access, i32
// CHECK:STDOUT:   %.loc27_18.2: <unbound element of class Access> = field_decl l, element1
// CHECK:STDOUT:   %l: <unbound element of class Access> = bind_name l, %.loc27_18.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT:   .k = %k
// CHECK:STDOUT:   .l = %l
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %H: <function> = fn_decl @H
// CHECK:STDOUT:   %I: <function> = fn_decl @I
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .H = %H
// CHECK:STDOUT:   .I = %I
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract {
// CHECK:STDOUT:   %J: <function> = fn_decl @J
// CHECK:STDOUT:   %K: <function> = fn_decl @K
// CHECK:STDOUT:   %L: <function> = fn_decl @L
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
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
