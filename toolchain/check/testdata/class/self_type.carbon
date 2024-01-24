// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn F[self: Self]() -> i32;
  var p: Self*;
}

// TODO: Reenter the class scope when defining methods, so that we can support
// `Self` here too.
// fn Class.F[self: Self]() -> i32 {
fn Class.F[self: Class]() -> i32 {
  return (*self.p).F();
}

// CHECK:STDOUT: --- self_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = ptr_type Class [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type Class, Class* [template]
// CHECK:STDOUT:   %.3: type = struct_type {.p: Class*} [template]
// CHECK:STDOUT:   %.4: type = ptr_type {.p: Class*} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Class = %Class.decl} [template]
// CHECK:STDOUT:   %Class.decl = class_decl @Class, ()
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:   %.loc9_14: type = ptr_type Class [template = constants.%.1]
// CHECK:STDOUT:   %.loc9_8: <unbound element of class Class> = field_decl p, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .p = %.loc9_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self: Class]() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: Class = name_ref self, %self
// CHECK:STDOUT:   %.loc16_16.1: ref Class* = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc16_16.2: Class* = bind_value %.loc16_16.1
// CHECK:STDOUT:   %.loc16_11.1: ref Class = deref %.loc16_16.2
// CHECK:STDOUT:   %.loc16_19: <bound method> = bound_method %.loc16_11.1, @Class.%F
// CHECK:STDOUT:   %.loc16_11.2: Class = bind_value %.loc16_11.1
// CHECK:STDOUT:   %.loc16_21.1: init i32 = call %.loc16_19(%.loc16_11.2)
// CHECK:STDOUT:   %.loc16_23: i32 = value_of_initializer %.loc16_21.1
// CHECK:STDOUT:   %.loc16_21.2: i32 = converted %.loc16_21.1, %.loc16_23
// CHECK:STDOUT:   return %.loc16_21.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
