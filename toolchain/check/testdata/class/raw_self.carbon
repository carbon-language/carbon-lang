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

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc9_46.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc9_46.2: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %.loc9_46.3: type = ptr_type (i32, i32)
// CHECK:STDOUT:   %.loc11: type = struct_type {.n: i32}
// CHECK:STDOUT:   %.loc7: type = ptr_type {.n: i32}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "raw_self.carbon" {
// CHECK:STDOUT:   class_decl @Class, ()
// CHECK:STDOUT:   %Class: type = class_type @Class
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT:   %.loc10_8.1: type = unbound_field_type Class, i32
// CHECK:STDOUT:   %.loc10_8.2: <unbound field of class Class> = field n, member0
// CHECK:STDOUT:   %n: <unbound field of class Class> = bind_name n, %.loc10_8.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT:   .n = %n
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self.addr: Class*](%self: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref.loc14_5: Class* = name_ref self, %self.addr
// CHECK:STDOUT:   %.loc14_4: ref Class = deref %self.ref.loc14_5
// CHECK:STDOUT:   %.loc14_10: ref i32 = class_field_access %.loc14_4, member0
// CHECK:STDOUT:   %self.ref.loc14_15: i32 = name_ref r#self, %self
// CHECK:STDOUT:   assign %.loc14_10, %self.ref.loc14_15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[%self.loc17_12: Class](%self.loc17_25: i32) -> %return: (i32, i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref.loc18_11: Class = name_ref self, %self.loc17_12
// CHECK:STDOUT:   %.loc18_15.1: ref i32 = class_field_access %self.ref.loc18_11, member0
// CHECK:STDOUT:   %.loc18_15.2: i32 = bind_value %.loc18_15.1
// CHECK:STDOUT:   %self.ref.loc18_19: i32 = name_ref r#self, %self.loc17_25
// CHECK:STDOUT:   %.loc18_25.1: (i32, i32) = tuple_literal (%.loc18_15.2, %self.ref.loc18_19)
// CHECK:STDOUT:   %.loc18_25.2: ref i32 = tuple_access %return, member0
// CHECK:STDOUT:   %.loc18_25.3: init i32 = initialize_from %.loc18_15.2 to %.loc18_25.2
// CHECK:STDOUT:   %.loc18_25.4: ref i32 = tuple_access %return, member1
// CHECK:STDOUT:   %.loc18_25.5: init i32 = initialize_from %self.ref.loc18_19 to %.loc18_25.4
// CHECK:STDOUT:   %.loc18_25.6: init (i32, i32) = tuple_init (%.loc18_25.3, %.loc18_25.5) to %return
// CHECK:STDOUT:   %.loc18_25.7: init (i32, i32) = converted %.loc18_25.1, %.loc18_25.6
// CHECK:STDOUT:   return %.loc18_25.7
// CHECK:STDOUT: }
