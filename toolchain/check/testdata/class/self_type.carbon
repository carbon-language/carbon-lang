// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn F[self: Self]() -> i32;
  var p: Self*;
}

fn Class.F[self: Self]() -> i32 {
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
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.loc12_12.1: Class = param self
// CHECK:STDOUT:     @F.%self: Class = bind_name self, %self.loc12_12.1
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %Self.ref.loc8: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.loc8_8.1: Class = param self
// CHECK:STDOUT:     %self.loc8_8.2: Class = bind_name self, %self.loc8_8.1
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Self.ref.loc9: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:   %.loc9_14: type = ptr_type Class [template = constants.%.1]
// CHECK:STDOUT:   %.loc9_8: <unbound element of class Class> = field_decl p, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .p = %.loc9_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self: Class]() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: Class = name_ref self, %self
// CHECK:STDOUT:   %p.ref: <unbound element of class Class> = name_ref p, @Class.%.loc9_8 [template = @Class.%.loc9_8]
// CHECK:STDOUT:   %.loc13_16.1: ref Class* = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc13_16.2: Class* = bind_value %.loc13_16.1
// CHECK:STDOUT:   %.loc13_11.1: ref Class = deref %.loc13_16.2
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, @Class.%F [template = @Class.%F]
// CHECK:STDOUT:   %.loc13_19: <bound method> = bound_method %.loc13_11.1, %F.ref
// CHECK:STDOUT:   %.loc13_11.2: Class = bind_value %.loc13_11.1
// CHECK:STDOUT:   %.loc13_21.1: init i32 = call %.loc13_19(%.loc13_11.2)
// CHECK:STDOUT:   %.loc13_23: i32 = value_of_initializer %.loc13_21.1
// CHECK:STDOUT:   %.loc13_21.2: i32 = converted %.loc13_21.1, %.loc13_23
// CHECK:STDOUT:   return %.loc13_21.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
