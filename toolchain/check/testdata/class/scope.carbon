// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn F() -> i32 {
    return 1;
  }

  fn G() -> i32 {
    return F();
  }
}

fn F() -> i32 {
  return 2;
}

fn Run() -> i32 {
  return F() + Class.F();
}

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc15: type = struct_type {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "scope.carbon" {
// CHECK:STDOUT:   class_decl @Class, ()
// CHECK:STDOUT:   %Class: type = class_type @Class
// CHECK:STDOUT:   %F: <function> = fn_decl @F.2
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.1
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9: i32 = int_literal 1
// CHECK:STDOUT:   return %.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, @Class.%F
// CHECK:STDOUT:   %.loc13_13: init i32 = call %F.ref()
// CHECK:STDOUT:   %.loc13_15.1: i32 = value_of_initializer %.loc13_13
// CHECK:STDOUT:   %.loc13_15.2: i32 = converted %.loc13_13, %.loc13_15.1
// CHECK:STDOUT:   return %.loc13_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18: i32 = int_literal 2
// CHECK:STDOUT:   return %.loc18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref.loc22_10: <function> = name_ref F, file.%F
// CHECK:STDOUT:   %.loc22_11: init i32 = call %F.ref.loc22_10()
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class
// CHECK:STDOUT:   %F.ref.loc22_21: <function> = name_ref F, @Class.%F
// CHECK:STDOUT:   %.loc22_23.1: init i32 = call %F.ref.loc22_21()
// CHECK:STDOUT:   %.loc22_14.1: i32 = value_of_initializer %.loc22_11
// CHECK:STDOUT:   %.loc22_14.2: i32 = converted %.loc22_11, %.loc22_14.1
// CHECK:STDOUT:   %.loc22_23.2: i32 = value_of_initializer %.loc22_23.1
// CHECK:STDOUT:   %.loc22_23.3: i32 = converted %.loc22_23.1, %.loc22_23.2
// CHECK:STDOUT:   %.loc22_14.3: i32 = add %.loc22_14.2, %.loc22_23.3
// CHECK:STDOUT:   return %.loc22_14.3
// CHECK:STDOUT: }
