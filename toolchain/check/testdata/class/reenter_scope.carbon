// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn F() -> i32;
  fn G() -> i32;
}

fn Class.F() -> i32 {
  return G();
}

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc10: type = struct_type {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "reenter_scope.carbon" {
// CHECK:STDOUT:   class_decl @Class, ()
// CHECK:STDOUT:   %Class: type = class_type @Class
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: <function> = name_ref G, @Class.%G
// CHECK:STDOUT:   %.loc13_11: init i32 = call %G.ref()
// CHECK:STDOUT:   %.loc13_13.1: i32 = value_of_initializer %.loc13_11
// CHECK:STDOUT:   %.loc13_13.2: i32 = converted %.loc13_11, %.loc13_13.1
// CHECK:STDOUT:   return %.loc13_13.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> i32;
