// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn F() -> i32 {
    return 1;
  }
}

fn F() -> i32 {
  return 2;
}

fn Run() -> i32 {
  return F() + Class.F();
}

// CHECK:STDOUT: file "scope.carbon" {
// CHECK:STDOUT:   class_declaration @Class, ()
// CHECK:STDOUT:   %Class: type = class_type @Class
// CHECK:STDOUT:   %.loc11: type = struct_type {}
// CHECK:STDOUT:   %F: <function> = fn_decl @F.2
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9: i32 = int_literal 1
// CHECK:STDOUT:   return %.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14: i32 = int_literal 2
// CHECK:STDOUT:   return %.loc14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref.loc18_10: <function> = name_reference "F", file.%F
// CHECK:STDOUT:   %.loc18_11: init i32 = call %F.ref.loc18_10()
// CHECK:STDOUT:   %Class.ref: type = name_reference "Class", file.%Class
// CHECK:STDOUT:   %F.ref.loc18_21: <function> = name_reference "F", @Class.%F
// CHECK:STDOUT:   %.loc18_23.1: init i32 = call %F.ref.loc18_21()
// CHECK:STDOUT:   %.loc18_14.1: i32 = value_of_initializer %.loc18_11
// CHECK:STDOUT:   %.loc18_23.2: i32 = value_of_initializer %.loc18_23.1
// CHECK:STDOUT:   %.loc18_14.2: i32 = add %.loc18_14.1, %.loc18_23.2
// CHECK:STDOUT:   return %.loc18_14.2
// CHECK:STDOUT: }
