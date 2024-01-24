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
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = ptr_type Class [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type Class, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.n: i32} [template]
// CHECK:STDOUT:   %.4: type = ptr_type {.n: i32} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Class = %Class.decl} [template]
// CHECK:STDOUT:   %Class.decl = class_decl @Class, ()
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template]
// CHECK:STDOUT:   %.loc11: <unbound element of class Class> = field_decl n, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT:   .n = %.loc11
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
