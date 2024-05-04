// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class(T:! type) {
  fn F(n: T) -> T;
}

// CHECK:STDERR: fail_todo_member_out_of_line.carbon:[[@LINE+10]]:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn Class(T:! type).F(n: T) -> T {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_member_out_of_line.carbon:[[@LINE-7]]:1: Name is previously declared here.
// CHECK:STDERR: class Class(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_member_out_of_line.carbon:[[@LINE+3]]:19: ERROR: `fn` declarations must either end with a `;` or have a `{ ... }` block for a definition.
// CHECK:STDERR: fn Class(T:! type).F(n: T) -> T {
// CHECK:STDERR:                   ^
fn Class(T:! type).F(n: T) -> T {
  return n;
}

// CHECK:STDOUT: --- fail_todo_member_out_of_line.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Class.1: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Class = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.2: F = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: type = fn_type @.1 [template]
// CHECK:STDOUT:   %struct.3: <invalid> = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl: Class = class_decl @Class [template = constants.%struct.1] {
// CHECK:STDOUT:     %T.loc7_13.1: type = param T
// CHECK:STDOUT:     %T.loc7_13.2: type = bind_symbolic_name T 0, %T.loc7_13.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: <invalid> = fn_decl @.1 [template = constants.%struct.3] {
// CHECK:STDOUT:     %T.loc21_10.1: type = param T
// CHECK:STDOUT:     @.1.%T: type = bind_symbolic_name T 0, %T.loc21_10.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.2] {
// CHECK:STDOUT:     %T.ref.loc8_11: type = name_ref T, file.%T.loc7_13.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %n.loc8_8.1: T = param n
// CHECK:STDOUT:     %n.loc8_8.2: T = bind_name n, %n.loc8_8.1
// CHECK:STDOUT:     %T.ref.loc8_17: type = name_ref T, file.%T.loc7_13.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %return.var: ref T = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class.2
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(@Class.%n.loc8_8.2: T) -> T;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1(%T: type);
// CHECK:STDOUT:
