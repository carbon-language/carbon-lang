// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn F() -> i32;
}

fn Run() -> i32 {
  var c: Class;
  return c.F();
}

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc9_1.1: type = struct_type {}
// CHECK:STDOUT:   %.loc9_1.2: type = tuple_type ()
// CHECK:STDOUT:   %.loc7: type = ptr_type {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "static_method.carbon" {
// CHECK:STDOUT:   class_decl @Class, ()
// CHECK:STDOUT:   %Class: type = class_type @Class
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run
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
// CHECK:STDOUT: fn @Run() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class
// CHECK:STDOUT:   %c.var: ref Class = var c
// CHECK:STDOUT:   %c: ref Class = bind_name c, %c.var
// CHECK:STDOUT:   %c.ref: ref Class = name_ref c, %c
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, @Class.%F
// CHECK:STDOUT:   %.loc13_13: init i32 = call %F.ref()
// CHECK:STDOUT:   %.loc13_15.1: i32 = value_of_initializer %.loc13_13
// CHECK:STDOUT:   %.loc13_15.2: i32 = converted %.loc13_13, %.loc13_15.1
// CHECK:STDOUT:   return %.loc13_15.2
// CHECK:STDOUT: }
