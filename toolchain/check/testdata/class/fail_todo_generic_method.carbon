// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_todo_generic_method.carbon:[[@LINE+4]]:1: ERROR: Semantics TODO: `generic class`.
// CHECK:STDERR: class Class(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
class Class(T:! type) {
  var a: T;
  fn F[self: Self](n: T);
}

// CHECK:STDERR: fail_todo_generic_method.carbon:[[@LINE+10]]:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn Class(T:! type).F[self: Self](n: T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_generic_method.carbon:[[@LINE-8]]:1: Name is previously declared here.
// CHECK:STDERR: class Class(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
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
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {
// CHECK:STDOUT:     %T.loc11_13.1: type = param T
// CHECK:STDOUT:     %T.loc11_13.2: type = bind_symbolic_name T, %T.loc11_13.1 [symbolic]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc26: <function> = fn_decl @.1 [template] {
// CHECK:STDOUT:     %T.loc26_10.1: type = param T
// CHECK:STDOUT:     @.1.%T: type = bind_symbolic_name T, %T.loc26_10.1 [symbolic]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %T.ref.loc12: type = name_ref T, file.%T.loc11_13.2 [symbolic = file.%T.loc11_13.2]
// CHECK:STDOUT:   %.loc12: <unbound element of class Class> = field_decl a, element0 [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.loc13_8.1: Class = param self
// CHECK:STDOUT:     %self.loc13_8.2: Class = bind_name self, %self.loc13_8.1
// CHECK:STDOUT:     %T.ref.loc13: type = name_ref T, file.%T.loc11_13.2 [symbolic = file.%T.loc11_13.2]
// CHECK:STDOUT:     %n.loc13_20.1: T = param n
// CHECK:STDOUT:     %n.loc13_20.2: T = bind_name n, %n.loc13_20.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .a = %.loc12
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[@Class.%self.loc13_8.2: Class](@Class.%n.loc13_20.2: T);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1(%T: type);
// CHECK:STDOUT:
