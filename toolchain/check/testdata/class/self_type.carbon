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

// CHECK:STDOUT: file "self_type.carbon" {
// CHECK:STDOUT:   class_declaration @Class, ()
// CHECK:STDOUT:   %Class: type = class_type @Class
// CHECK:STDOUT:   %.loc10: type = struct_type {.p: Class*}
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %Self.ref: type = name_reference "Self", file.%Class
// CHECK:STDOUT:   %.loc9_14: type = ptr_type Class
// CHECK:STDOUT:   %.loc9_8.1: type = unbound_field_type Class, Class*
// CHECK:STDOUT:   %.loc9_8.2: <unbound field of class Class> = field "p", member0
// CHECK:STDOUT:   %p: <unbound field of class Class> = bind_name "p", %.loc9_8.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .p = %p
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self: Class]() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7: type = ptr_type {.p: Class*}
// CHECK:STDOUT:   %self.ref: Class = name_reference "self", %self
// CHECK:STDOUT:   %.loc16_16.1: ref Class* = class_field_access %self.ref, member0
// CHECK:STDOUT:   %.loc16_16.2: Class* = bind_value %.loc16_16.1
// CHECK:STDOUT:   %.loc16_11.1: ref Class = dereference %.loc16_16.2
// CHECK:STDOUT:   %.loc16_19: <bound method> = bound_method %.loc16_11.1, @Class.%F
// CHECK:STDOUT:   %.loc16_11.2: Class = bind_value %.loc16_11.1
// CHECK:STDOUT:   %.loc16_21: init i32 = call %.loc16_19(%.loc16_11.2)
// CHECK:STDOUT:   %.loc16_23: i32 = value_of_initializer %.loc16_21
// CHECK:STDOUT:   return %.loc16_23
// CHECK:STDOUT: }
