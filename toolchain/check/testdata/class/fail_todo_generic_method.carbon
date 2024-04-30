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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type Class, T [symbolic]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: T} [symbolic]
// CHECK:STDOUT:   %.4: type = fn_type @.1 [template]
// CHECK:STDOUT:   %struct.2: <invalid> = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {
// CHECK:STDOUT:     %T.loc11_13.1: type = param T
// CHECK:STDOUT:     %T.loc11_13.2: type = bind_symbolic_name T 0, %T.loc11_13.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: <invalid> = fn_decl @.1 [template = constants.%struct.2] {
// CHECK:STDOUT:     %T.loc26_10.1: type = param T
// CHECK:STDOUT:     @.1.%T: type = bind_symbolic_name T 0, %T.loc26_10.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %T.ref.loc12: type = name_ref T, file.%T.loc11_13.2 [symbolic = constants.%T]
// CHECK:STDOUT:   %.loc12: <unbound element of class Class> = field_decl a, element0 [template]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.1] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.loc13_8.1: Class = param self
// CHECK:STDOUT:     %self.loc13_8.2: Class = bind_name self, %self.loc13_8.1
// CHECK:STDOUT:     %T.ref.loc13: type = name_ref T, file.%T.loc11_13.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %n.loc13_20.1: T = param n
// CHECK:STDOUT:     %n.loc13_20.2: T = bind_name n, %n.loc13_20.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .a = %.loc12
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[@Class.%self.loc13_8.2: Class](@Class.%n.loc13_20.2: T);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1(%T: type);
// CHECK:STDOUT:
