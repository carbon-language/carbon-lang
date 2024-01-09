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

fn Run() {
  var a: i32 = F();
  var b: i32 = Class.F();
}

// CHECK:STDOUT: --- scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc15: type = struct_type {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Class = %Class.decl, .F = %F, .Run = %Run}
// CHECK:STDOUT:   %Class.decl = class_decl @Class, ()
// CHECK:STDOUT:   %Class: type = class_type @Class
// CHECK:STDOUT:   %.loc7: type = reify_constant %Class, %Class
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
// CHECK:STDOUT:   %.loc13_12: <function> = reify_constant %F.ref, @Class.%F
// CHECK:STDOUT:   %.loc13_13: init i32 = call @Class.%F()
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
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref i32 = var a
// CHECK:STDOUT:   %a: ref i32 = bind_name a, %a.var
// CHECK:STDOUT:   %F.ref.loc22: <function> = name_ref F, file.%F
// CHECK:STDOUT:   %.loc22_16: <function> = reify_constant %F.ref.loc22, file.%F
// CHECK:STDOUT:   %.loc22_17: init i32 = call file.%F()
// CHECK:STDOUT:   assign %a.var, %.loc22_17
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class
// CHECK:STDOUT:   %F.ref.loc23: <function> = name_ref F, @Class.%F
// CHECK:STDOUT:   %.loc23_21: <function> = reify_constant %F.ref.loc23, @Class.%F
// CHECK:STDOUT:   %.loc23_23: init i32 = call @Class.%F()
// CHECK:STDOUT:   assign %b.var, %.loc23_23
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
