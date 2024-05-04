// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn F[self: Self]() -> i32;
  fn Make() -> Self {
    returned var s: Self;
    s = {.p = &s};
    return var;
  }
  var p: Self*;
}

fn Class.F[self: Self]() -> i32 {
  return (*self.p).F();
}

// CHECK:STDOUT: --- self_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %Make: type = fn_type @Make [template]
// CHECK:STDOUT:   %struct.2: Make = struct_value () [template]
// CHECK:STDOUT:   %.2: type = ptr_type Class [template]
// CHECK:STDOUT:   %.3: type = unbound_element_type Class, Class* [template]
// CHECK:STDOUT:   %.4: type = struct_type {.p: Class*} [template]
// CHECK:STDOUT:   %.5: type = ptr_type {.p: Class*} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.1] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.loc17_12.1: Class = param self
// CHECK:STDOUT:     @F.%self: Class = bind_name self, %self.loc17_12.1
// CHECK:STDOUT:     @F.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.1] {
// CHECK:STDOUT:     %Self.ref.loc8: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.loc8_8.1: Class = param self
// CHECK:STDOUT:     %self.loc8_8.2: Class = bind_name self, %self.loc8_8.1
// CHECK:STDOUT:     %return.var.loc8: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Make.decl: Make = fn_decl @Make [template = constants.%struct.2] {
// CHECK:STDOUT:     %Self.ref.loc9: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %return.var.loc9: ref Class = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Self.ref.loc14: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:   %.loc14_14: type = ptr_type Class [template = constants.%.2]
// CHECK:STDOUT:   %.loc14_8: <unbound element of class Class> = field_decl p, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .Make = %Make.decl
// CHECK:STDOUT:   .p = %.loc14_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self: Class]() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: Class = name_ref self, %self
// CHECK:STDOUT:   %p.ref: <unbound element of class Class> = name_ref p, @Class.%.loc14_8 [template = @Class.%.loc14_8]
// CHECK:STDOUT:   %.loc18_16.1: ref Class* = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc18_16.2: Class* = bind_value %.loc18_16.1
// CHECK:STDOUT:   %.loc18_11.1: ref Class = deref %.loc18_16.2
// CHECK:STDOUT:   %F.ref: F = name_ref F, @Class.%F.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc18_19: <bound method> = bound_method %.loc18_11.1, %F.ref
// CHECK:STDOUT:   %.loc18_11.2: Class = bind_value %.loc18_11.1
// CHECK:STDOUT:   %F.call: init i32 = call %.loc18_19(%.loc18_11.2)
// CHECK:STDOUT:   %.loc18_23.1: i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc18_23.2: i32 = converted %F.call, %.loc18_23.1
// CHECK:STDOUT:   return %.loc18_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> @Class.%return.var.loc9: Class {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:   %s: ref Class = bind_name s, @Class.%return.var.loc9
// CHECK:STDOUT:   %s.ref.loc11_5: ref Class = name_ref s, %s
// CHECK:STDOUT:   %s.ref.loc11_16: ref Class = name_ref s, %s
// CHECK:STDOUT:   %.loc11_15: Class* = addr_of %s.ref.loc11_16
// CHECK:STDOUT:   %.loc11_17.1: {.p: Class*} = struct_literal (%.loc11_15)
// CHECK:STDOUT:   %.loc11_17.2: ref Class* = class_element_access %s.ref.loc11_5, element0
// CHECK:STDOUT:   %.loc11_17.3: init Class* = initialize_from %.loc11_15 to %.loc11_17.2
// CHECK:STDOUT:   %.loc11_17.4: init Class = class_init (%.loc11_17.3), %s.ref.loc11_5
// CHECK:STDOUT:   %.loc11_7: init Class = converted %.loc11_17.1, %.loc11_17.4
// CHECK:STDOUT:   assign %s.ref.loc11_5, %.loc11_7
// CHECK:STDOUT:   return %s to @Class.%return.var.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
