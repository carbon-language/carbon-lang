// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn F(n: i32) -> i32 {
    return n;
  }

  fn G(n: i32) -> i32;

  var k: i32;
}

fn Class.G(n: i32) -> i32 {
  return n;
}

fn Run() -> i32 {
  return Class.F(4);
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type Class, i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.k: i32} [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 4 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .Run = %Run
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template] {
// CHECK:STDOUT:     %n.loc17_12.1: i32 = param n
// CHECK:STDOUT:     @G.%n: i32 = bind_name n, %n.loc17_12.1
// CHECK:STDOUT:     %return.var.loc17: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run [template] {
// CHECK:STDOUT:     %return.var.loc21: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %n.loc8_8.1: i32 = param n
// CHECK:STDOUT:     %n.loc8_8.2: i32 = bind_name n, %n.loc8_8.1
// CHECK:STDOUT:     %return.var.loc8: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template] {
// CHECK:STDOUT:     %n.loc12_8.1: i32 = param n
// CHECK:STDOUT:     %n.loc12_8.2: i32 = bind_name n, %n.loc12_8.1
// CHECK:STDOUT:     %return.var.loc12: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14: <unbound element of class Class> = field_decl k, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT:   .k = %.loc14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(@Class.%n.loc8_8.2: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, @Class.%n.loc8_8.2
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%n: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, @Class.%F [template = @Class.%F]
// CHECK:STDOUT:   %.loc22_18: i32 = int_literal 4 [template = constants.%.3]
// CHECK:STDOUT:   %.loc22_17.1: init i32 = call %F.ref(%.loc22_18)
// CHECK:STDOUT:   %.loc22_20: i32 = value_of_initializer %.loc22_17.1
// CHECK:STDOUT:   %.loc22_17.2: i32 = converted %.loc22_17.1, %.loc22_20
// CHECK:STDOUT:   return %.loc22_17.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
