// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_todo_generic_method.carbon:[[@LINE+3]]:1: ERROR: Semantics TODO: `generic class`.
// CHECK:STDERR: class Class(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
class Class(T:! type) {
  var a: T;
  fn F[self: Self](n: T);
}

// CHECK:STDERR: fail_todo_generic_method.carbon:[[@LINE+9]]:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn Class(T:! type).F[self: Self](n: T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_generic_method.carbon:[[@LINE-8]]:1: Name is previously declared here.
// CHECK:STDERR: class Class(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_generic_method.carbon:[[@LINE+3]]:19: ERROR: `fn` declarations must either end with a `;` or have a `{ ... }` block for a definition.
// CHECK:STDERR: fn Class(T:! type).F[self: Self](n: T) {}
// CHECK:STDERR:                   ^
fn Class(T:! type).F[self: Self](n: T) {}

// CHECK:STDOUT: --- fail_todo_generic_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type Class, T [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {.a: T} [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Class = %Class.decl} [template]
// CHECK:STDOUT:   %Class.decl = class_decl @Class, (<unexpected instref inst+1>, <unexpected instref inst+2>)
// CHECK:STDOUT:   %.loc24: <function> = fn_decl @.1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %T.ref: type = name_ref T, <unexpected instref inst+2> [symbolic = <unexpected instref inst+2>]
// CHECK:STDOUT:   %.loc11: <unbound element of class Class> = field_decl a, element0 [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .a = %.loc11
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self: Class](%n: T);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1(%T: type);
// CHECK:STDOUT:
