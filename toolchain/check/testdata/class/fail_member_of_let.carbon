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
// CHECK:STDERR: fail_member_of_let.carbon:[[@LINE+3]]:4: ERROR: Name `T` not found.
// CHECK:STDERR: fn T.F() {}
// CHECK:STDERR:    ^
fn T.F() {}

// CHECK:STDOUT: --- fail_member_of_let.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Class = %Class.decl} [template]
// CHECK:STDOUT:   %Class.decl = class_decl @Class, ()
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:   %T: type = bind_name T, %Class.ref
// CHECK:STDOUT:   %.loc19: <function> = fn_decl @.1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
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
// CHECK:STDOUT:
