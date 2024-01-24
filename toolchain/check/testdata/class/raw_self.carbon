// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn F[addr self: Class*](r#self: i32);
  fn G[self: Class](r#self: i32) -> (i32, i32);
  var n: i32;
}

fn Class.F[addr self: Class*](r#self: i32) {
  (*self).n = r#self;
}

fn Class.G[self: Class](r#self: i32) -> (i32, i32) {
  return (self.n, r#self);
}

// CHECK:STDOUT: --- raw_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = ptr_type Class [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.4: type = ptr_type (i32, i32) [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type Class, i32 [template]
// CHECK:STDOUT:   %.6: type = struct_type {.n: i32} [template]
// CHECK:STDOUT:   %.7: type = ptr_type {.n: i32} [template]
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
// CHECK:STDOUT:   %.loc10: <unbound element of class Class> = field_decl n, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT:   .n = %.loc10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[addr %self.loc13_17: Class*](%self.loc13_31: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref.loc14_5: Class* = name_ref self, %self.loc13_17
// CHECK:STDOUT:   %.loc14_4: ref Class = deref %self.ref.loc14_5
// CHECK:STDOUT:   %.loc14_10: ref i32 = class_element_access %.loc14_4, element0
// CHECK:STDOUT:   %self.ref.loc14_15: i32 = name_ref r#self, %self.loc13_31
// CHECK:STDOUT:   assign %.loc14_10, %self.ref.loc14_15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[%self.loc17_12: Class](%self.loc17_25: i32) -> %return: (i32, i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref.loc18_11: Class = name_ref self, %self.loc17_12
// CHECK:STDOUT:   %.loc18_15.1: ref i32 = class_element_access %self.ref.loc18_11, element0
// CHECK:STDOUT:   %.loc18_15.2: i32 = bind_value %.loc18_15.1
// CHECK:STDOUT:   %self.ref.loc18_19: i32 = name_ref r#self, %self.loc17_25
// CHECK:STDOUT:   %.loc18_25.1: (i32, i32) = tuple_literal (%.loc18_15.2, %self.ref.loc18_19)
// CHECK:STDOUT:   %.loc18_25.2: ref i32 = tuple_access %return, element0
// CHECK:STDOUT:   %.loc18_25.3: init i32 = initialize_from %.loc18_15.2 to %.loc18_25.2
// CHECK:STDOUT:   %.loc18_25.4: ref i32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc18_25.5: init i32 = initialize_from %self.ref.loc18_19 to %.loc18_25.4
// CHECK:STDOUT:   %.loc18_25.6: init (i32, i32) = tuple_init (%.loc18_25.3, %.loc18_25.5) to %return
// CHECK:STDOUT:   %.loc18_25.7: init (i32, i32) = converted %.loc18_25.1, %.loc18_25.6
// CHECK:STDOUT:   return %.loc18_25.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
