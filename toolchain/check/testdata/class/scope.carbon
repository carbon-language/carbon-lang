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
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %F.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.2: G = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %F.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %struct.3: F = struct_value () [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %Run: type = fn_type @Run [template]
// CHECK:STDOUT:   %struct.4: Run = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.2 [template = constants.%struct.3] {
// CHECK:STDOUT:     @F.2.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Run.decl: Run = fn_decl @Run [template = constants.%struct.4] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.1 [template = constants.%struct.1] {
// CHECK:STDOUT:     %return.var.loc8: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.2] {
// CHECK:STDOUT:     %return.var.loc12: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   return %.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: F = name_ref F, @Class.%F.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %F.call: init i32 = call %F.ref()
// CHECK:STDOUT:   %.loc13_15.1: i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc13_15.2: i32 = converted %F.call, %.loc13_15.1
// CHECK:STDOUT:   return %.loc13_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18: i32 = int_literal 2 [template = constants.%.4]
// CHECK:STDOUT:   return %.loc18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref i32 = var a
// CHECK:STDOUT:   %a: ref i32 = bind_name a, %a.var
// CHECK:STDOUT:   %F.ref.loc22: F = name_ref F, file.%F.decl [template = constants.%struct.3]
// CHECK:STDOUT:   %F.call.loc22: init i32 = call %F.ref.loc22()
// CHECK:STDOUT:   assign %a.var, %F.call.loc22
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %F.ref.loc23: F = name_ref F, @Class.%F.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %F.call.loc23: init i32 = call %F.ref.loc23()
// CHECK:STDOUT:   assign %b.var, %F.call.loc23
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
