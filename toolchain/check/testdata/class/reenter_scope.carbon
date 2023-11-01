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

// CHECK:STDOUT: file "reenter_scope.carbon" {
// CHECK:STDOUT:   class_declaration @Class, ()
// CHECK:STDOUT:   %Class: type = class_type @Class
// CHECK:STDOUT:   %.loc10: type = struct_type {}
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
// CHECK:STDOUT:   %G.ref: <function> = name_reference "G", @Class.%G
// CHECK:STDOUT:   %.loc13_11.1: init i32 = call %G.ref()
// CHECK:STDOUT:   %.loc13_11.2: ref i32 = temporary_storage
// CHECK:STDOUT:   %.loc13_11.3: ref i32 = temporary %.loc13_11.2, %.loc13_11.1
// CHECK:STDOUT:   %.loc13_11.4: i32 = bind_value %.loc13_11.3
// CHECK:STDOUT:   return %.loc13_11.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> i32;
