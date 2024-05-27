// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn F[addr self: Self*](r#self: i32);
  fn G[self: Self](r#self: i32) -> (i32, i32);
  var n: i32;
}

fn Class.F[addr self: Self*](r#self: i32) {
  (*self).n = r#self;
}

fn Class.G[self: Self](r#self: i32) -> (i32, i32) {
  return (self.n, r#self);
}

// CHECK:STDOUT: --- raw_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = ptr_type Class [template]
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.2: F = struct_value () [template]
// CHECK:STDOUT:   %.3: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.4: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.3: G = struct_value () [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type Class, i32 [template]
// CHECK:STDOUT:   %.6: type = struct_type {.n: i32} [template]
// CHECK:STDOUT:   %.7: type = ptr_type {.n: i32} [template]
// CHECK:STDOUT:   %.8: type = ptr_type (i32, i32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.5: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.6: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.2] {
// CHECK:STDOUT:     %Self.ref.loc13: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %.loc13_27: type = ptr_type Class [template = constants.%.1]
// CHECK:STDOUT:     %self.loc13_17.1: Class* = param self
// CHECK:STDOUT:     @F.%self.loc13_17: Class* = bind_name self, %self.loc13_17.1
// CHECK:STDOUT:     @F.%.loc13: Class* = addr_pattern @F.%self.loc13_17
// CHECK:STDOUT:     %int.make_type_32.loc13: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc13_38.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:     %.loc13_38.2: type = converted %int.make_type_32.loc13, %.loc13_38.1 [template = i32]
// CHECK:STDOUT:     %self.loc13_30.1: i32 = param r#self
// CHECK:STDOUT:     @F.%self.loc13_30: i32 = bind_name r#self, %self.loc13_30.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.7: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.8: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.9: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.3] {
// CHECK:STDOUT:     %Self.ref.loc17: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.loc17_12.1: Class = param self
// CHECK:STDOUT:     @G.%self.loc17_12: Class = bind_name self, %self.loc17_12.1
// CHECK:STDOUT:     %int.make_type_32.loc17_32: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc17_32.1: type = value_of_initializer %int.make_type_32.loc17_32 [template = i32]
// CHECK:STDOUT:     %.loc17_32.2: type = converted %int.make_type_32.loc17_32, %.loc17_32.1 [template = i32]
// CHECK:STDOUT:     %self.loc17_24.1: i32 = param r#self
// CHECK:STDOUT:     @G.%self.loc17_24: i32 = bind_name r#self, %self.loc17_24.1
// CHECK:STDOUT:     %int.make_type_32.loc17_41: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc17_46: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc17_49.1: (type, type) = tuple_literal (%int.make_type_32.loc17_41, %int.make_type_32.loc17_46)
// CHECK:STDOUT:     %.loc17_49.2: type = value_of_initializer %int.make_type_32.loc17_41 [template = i32]
// CHECK:STDOUT:     %.loc17_49.3: type = converted %int.make_type_32.loc17_41, %.loc17_49.2 [template = i32]
// CHECK:STDOUT:     %.loc17_49.4: type = value_of_initializer %int.make_type_32.loc17_46 [template = i32]
// CHECK:STDOUT:     %.loc17_49.5: type = converted %int.make_type_32.loc17_46, %.loc17_49.4 [template = i32]
// CHECK:STDOUT:     %.loc17_49.6: type = converted %.loc17_49.1, constants.%.4 [template = constants.%.4]
// CHECK:STDOUT:     @G.%return: ref (i32, i32) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.2] {
// CHECK:STDOUT:     %Self.ref.loc8: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %.loc8_23: type = ptr_type Class [template = constants.%.1]
// CHECK:STDOUT:     %self.loc8_13.1: Class* = param self
// CHECK:STDOUT:     %self.loc8_13.3: Class* = bind_name self, %self.loc8_13.1
// CHECK:STDOUT:     %.loc8_8: Class* = addr_pattern %self.loc8_13.3
// CHECK:STDOUT:     %int.make_type_32.loc8: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc8_34.1: type = value_of_initializer %int.make_type_32.loc8 [template = i32]
// CHECK:STDOUT:     %.loc8_34.2: type = converted %int.make_type_32.loc8, %.loc8_34.1 [template = i32]
// CHECK:STDOUT:     %self.loc8_26.1: i32 = param r#self
// CHECK:STDOUT:     %self.loc8_26.2: i32 = bind_name r#self, %self.loc8_26.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.3] {
// CHECK:STDOUT:     %Self.ref.loc9: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.loc9_8.1: Class = param self
// CHECK:STDOUT:     %self.loc9_8.2: Class = bind_name self, %self.loc9_8.1
// CHECK:STDOUT:     %int.make_type_32.loc9_28: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc9_28.1: type = value_of_initializer %int.make_type_32.loc9_28 [template = i32]
// CHECK:STDOUT:     %.loc9_28.2: type = converted %int.make_type_32.loc9_28, %.loc9_28.1 [template = i32]
// CHECK:STDOUT:     %self.loc9_20.1: i32 = param r#self
// CHECK:STDOUT:     %self.loc9_20.2: i32 = bind_name r#self, %self.loc9_20.1
// CHECK:STDOUT:     %int.make_type_32.loc9_37: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc9_42: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc9_45.1: (type, type) = tuple_literal (%int.make_type_32.loc9_37, %int.make_type_32.loc9_42)
// CHECK:STDOUT:     %.loc9_45.2: type = value_of_initializer %int.make_type_32.loc9_37 [template = i32]
// CHECK:STDOUT:     %.loc9_45.3: type = converted %int.make_type_32.loc9_37, %.loc9_45.2 [template = i32]
// CHECK:STDOUT:     %.loc9_45.4: type = value_of_initializer %int.make_type_32.loc9_42 [template = i32]
// CHECK:STDOUT:     %.loc9_45.5: type = converted %int.make_type_32.loc9_42, %.loc9_45.4 [template = i32]
// CHECK:STDOUT:     %.loc9_45.6: type = converted %.loc9_45.1, constants.%.4 [template = constants.%.4]
// CHECK:STDOUT:     %return.var: ref (i32, i32) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc10: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc10_10.1: type = value_of_initializer %int.make_type_32.loc10 [template = i32]
// CHECK:STDOUT:   %.loc10_10.2: type = converted %int.make_type_32.loc10, %.loc10_10.1 [template = i32]
// CHECK:STDOUT:   %.loc10_8: <unbound element of class Class> = field_decl n, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .n = %.loc10_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[addr %self.loc13_17: Class*](%self.loc13_30: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref.loc14_5: Class* = name_ref self, %self.loc13_17
// CHECK:STDOUT:   %.loc14_4: ref Class = deref %self.ref.loc14_5
// CHECK:STDOUT:   %n.ref: <unbound element of class Class> = name_ref n, @Class.%.loc10_8 [template = @Class.%.loc10_8]
// CHECK:STDOUT:   %.loc14_10: ref i32 = class_element_access %.loc14_4, element0
// CHECK:STDOUT:   %self.ref.loc14_15: i32 = name_ref r#self, %self.loc13_30
// CHECK:STDOUT:   assign %.loc14_10, %self.ref.loc14_15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[%self.loc17_12: Class](%self.loc17_24: i32) -> %return: (i32, i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref.loc18_11: Class = name_ref self, %self.loc17_12
// CHECK:STDOUT:   %n.ref: <unbound element of class Class> = name_ref n, @Class.%.loc10_8 [template = @Class.%.loc10_8]
// CHECK:STDOUT:   %.loc18_15.1: ref i32 = class_element_access %self.ref.loc18_11, element0
// CHECK:STDOUT:   %.loc18_15.2: i32 = bind_value %.loc18_15.1
// CHECK:STDOUT:   %self.ref.loc18_19: i32 = name_ref r#self, %self.loc17_24
// CHECK:STDOUT:   %.loc18_25.1: (i32, i32) = tuple_literal (%.loc18_15.2, %self.ref.loc18_19)
// CHECK:STDOUT:   %.loc18_25.2: ref i32 = tuple_access %return, element0
// CHECK:STDOUT:   %.loc18_25.3: init i32 = initialize_from %.loc18_15.2 to %.loc18_25.2
// CHECK:STDOUT:   %.loc18_25.4: ref i32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc18_25.5: init i32 = initialize_from %self.ref.loc18_19 to %.loc18_25.4
// CHECK:STDOUT:   %.loc18_25.6: init (i32, i32) = tuple_init (%.loc18_25.3, %.loc18_25.5) to %return
// CHECK:STDOUT:   %.loc18_26: init (i32, i32) = converted %.loc18_25.1, %.loc18_25.6
// CHECK:STDOUT:   return %.loc18_26 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
