// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn F() -> i32;
}

// TODO: Use `:!` here once it is available.
let T: type = Class;

// The class name is required to be written in the same way as in the class
// declaration. An expression that evaluates to the class name is not accepted.
// CHECK:STDERR: fail_member_of_let.carbon:[[@LINE+6]]:6: ERROR: Declaration qualifiers are only allowed for entities that provide a scope.
// CHECK:STDERR: fn T.F() {}
// CHECK:STDERR:      ^
// CHECK:STDERR: fail_member_of_let.carbon:[[@LINE+3]]:4: Non-scope entity referenced here.
// CHECK:STDERR: fn T.F() {}
// CHECK:STDERR:    ^
fn T.F() {}

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc9: type = struct_type {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "fail_member_of_let.carbon" {
// CHECK:STDOUT:   class_decl @Class, ()
// CHECK:STDOUT:   %Class: type = class_type @Class
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, %Class
// CHECK:STDOUT:   %T: type = bind_name T, %Class.ref
// CHECK:STDOUT:   %.loc22: <function> = fn_decl @.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
