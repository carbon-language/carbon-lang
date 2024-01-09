// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn F[self: Class]() -> i32;
  fn G[addr self: Class*]() -> i32;

  var n: i32;
}

fn Class.F[self: Class]() -> i32 {
  return self.n;
}

fn Class.G[addr self: Class*]() -> i32 {
  return (*self).n;
}

// CHECK:STDOUT: --- self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc12: type = struct_type {.n: i32}
// CHECK:STDOUT:   %.loc7: type = ptr_type {.n: i32}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Class = %Class.decl}
// CHECK:STDOUT:   %Class.decl = class_decl @Class, ()
// CHECK:STDOUT:   %Class: type = class_type @Class
// CHECK:STDOUT:   %.loc7: type = reify_constant %Class, %Class
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT:   %.loc11_8.1: type = unbound_element_type Class, i32
// CHECK:STDOUT:   %.loc11_8.2: <unbound element of class Class> = field_decl n, element0
// CHECK:STDOUT:   %n: <unbound element of class Class> = bind_name n, %.loc11_8.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT:   .n = %n
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self: Class]() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: Class = name_ref self, %self
// CHECK:STDOUT:   %.loc15_14.1: ref i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc15_14.2: i32 = bind_value %.loc15_14.1
// CHECK:STDOUT:   return %.loc15_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[addr %self: Class*]() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: Class* = name_ref self, %self
// CHECK:STDOUT:   %.loc19_11: ref Class = deref %self.ref
// CHECK:STDOUT:   %.loc19_17.1: ref i32 = class_element_access %.loc19_11, element0
// CHECK:STDOUT:   %.loc19_17.2: i32 = bind_value %.loc19_17.1
// CHECK:STDOUT:   return %.loc19_17.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
