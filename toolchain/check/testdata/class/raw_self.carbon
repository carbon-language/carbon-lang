// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/raw_self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/raw_self.carbon

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
// CHECK:STDOUT:     %Self.ref.loc17: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %.loc17_27: type = ptr_type Class [template = constants.%.1]
// CHECK:STDOUT:     %self.loc17_17.1: Class* = param self
// CHECK:STDOUT:     @F.%self.loc17_17: Class* = bind_name self, %self.loc17_17.1
// CHECK:STDOUT:     @F.%.loc17: Class* = addr_pattern @F.%self.loc17_17
// CHECK:STDOUT:     %int.make_type_32.loc17: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc17_38.1: type = value_of_initializer %int.make_type_32.loc17 [template = i32]
// CHECK:STDOUT:     %.loc17_38.2: type = converted %int.make_type_32.loc17, %.loc17_38.1 [template = i32]
// CHECK:STDOUT:     %self.loc17_30.1: i32 = param r#self
// CHECK:STDOUT:     @F.%self.loc17_30: i32 = bind_name r#self, %self.loc17_30.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.7: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.8: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.9: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.3] {
// CHECK:STDOUT:     %Self.ref.loc21: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.loc21_12.1: Class = param self
// CHECK:STDOUT:     @G.%self.loc21_12: Class = bind_name self, %self.loc21_12.1
// CHECK:STDOUT:     %int.make_type_32.loc21_32: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc21_32.1: type = value_of_initializer %int.make_type_32.loc21_32 [template = i32]
// CHECK:STDOUT:     %.loc21_32.2: type = converted %int.make_type_32.loc21_32, %.loc21_32.1 [template = i32]
// CHECK:STDOUT:     %self.loc21_24.1: i32 = param r#self
// CHECK:STDOUT:     @G.%self.loc21_24: i32 = bind_name r#self, %self.loc21_24.1
// CHECK:STDOUT:     %int.make_type_32.loc21_41: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc21_46: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc21_49.1: (type, type) = tuple_literal (%int.make_type_32.loc21_41, %int.make_type_32.loc21_46)
// CHECK:STDOUT:     %.loc21_49.2: type = value_of_initializer %int.make_type_32.loc21_41 [template = i32]
// CHECK:STDOUT:     %.loc21_49.3: type = converted %int.make_type_32.loc21_41, %.loc21_49.2 [template = i32]
// CHECK:STDOUT:     %.loc21_49.4: type = value_of_initializer %int.make_type_32.loc21_46 [template = i32]
// CHECK:STDOUT:     %.loc21_49.5: type = converted %int.make_type_32.loc21_46, %.loc21_49.4 [template = i32]
// CHECK:STDOUT:     %.loc21_49.6: type = converted %.loc21_49.1, constants.%.4 [template = constants.%.4]
// CHECK:STDOUT:     @G.%return: ref (i32, i32) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.2] {
// CHECK:STDOUT:     %Self.ref.loc12: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %.loc12_23: type = ptr_type Class [template = constants.%.1]
// CHECK:STDOUT:     %self.loc12_13.1: Class* = param self
// CHECK:STDOUT:     %self.loc12_13.3: Class* = bind_name self, %self.loc12_13.1
// CHECK:STDOUT:     %.loc12_8: Class* = addr_pattern %self.loc12_13.3
// CHECK:STDOUT:     %int.make_type_32.loc12: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc12_34.1: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:     %.loc12_34.2: type = converted %int.make_type_32.loc12, %.loc12_34.1 [template = i32]
// CHECK:STDOUT:     %self.loc12_26.1: i32 = param r#self
// CHECK:STDOUT:     %self.loc12_26.2: i32 = bind_name r#self, %self.loc12_26.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.3] {
// CHECK:STDOUT:     %Self.ref.loc13: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.loc13_8.1: Class = param self
// CHECK:STDOUT:     %self.loc13_8.2: Class = bind_name self, %self.loc13_8.1
// CHECK:STDOUT:     %int.make_type_32.loc13_28: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc13_28.1: type = value_of_initializer %int.make_type_32.loc13_28 [template = i32]
// CHECK:STDOUT:     %.loc13_28.2: type = converted %int.make_type_32.loc13_28, %.loc13_28.1 [template = i32]
// CHECK:STDOUT:     %self.loc13_20.1: i32 = param r#self
// CHECK:STDOUT:     %self.loc13_20.2: i32 = bind_name r#self, %self.loc13_20.1
// CHECK:STDOUT:     %int.make_type_32.loc13_37: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc13_42: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc13_45.1: (type, type) = tuple_literal (%int.make_type_32.loc13_37, %int.make_type_32.loc13_42)
// CHECK:STDOUT:     %.loc13_45.2: type = value_of_initializer %int.make_type_32.loc13_37 [template = i32]
// CHECK:STDOUT:     %.loc13_45.3: type = converted %int.make_type_32.loc13_37, %.loc13_45.2 [template = i32]
// CHECK:STDOUT:     %.loc13_45.4: type = value_of_initializer %int.make_type_32.loc13_42 [template = i32]
// CHECK:STDOUT:     %.loc13_45.5: type = converted %int.make_type_32.loc13_42, %.loc13_45.4 [template = i32]
// CHECK:STDOUT:     %.loc13_45.6: type = converted %.loc13_45.1, constants.%.4 [template = constants.%.4]
// CHECK:STDOUT:     %return.var: ref (i32, i32) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc14: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc14_10.1: type = value_of_initializer %int.make_type_32.loc14 [template = i32]
// CHECK:STDOUT:   %.loc14_10.2: type = converted %int.make_type_32.loc14, %.loc14_10.1 [template = i32]
// CHECK:STDOUT:   %.loc14_8: <unbound element of class Class> = field_decl n, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .n = %.loc14_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[addr %self.loc17_17: Class*](%self.loc17_30: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref.loc18_5: Class* = name_ref self, %self.loc17_17
// CHECK:STDOUT:   %.loc18_4: ref Class = deref %self.ref.loc18_5
// CHECK:STDOUT:   %n.ref: <unbound element of class Class> = name_ref n, @Class.%.loc14_8 [template = @Class.%.loc14_8]
// CHECK:STDOUT:   %.loc18_10: ref i32 = class_element_access %.loc18_4, element0
// CHECK:STDOUT:   %self.ref.loc18_15: i32 = name_ref r#self, %self.loc17_30
// CHECK:STDOUT:   assign %.loc18_10, %self.ref.loc18_15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[%self.loc21_12: Class](%self.loc21_24: i32) -> %return: (i32, i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref.loc22_11: Class = name_ref self, %self.loc21_12
// CHECK:STDOUT:   %n.ref: <unbound element of class Class> = name_ref n, @Class.%.loc14_8 [template = @Class.%.loc14_8]
// CHECK:STDOUT:   %.loc22_15.1: ref i32 = class_element_access %self.ref.loc22_11, element0
// CHECK:STDOUT:   %.loc22_15.2: i32 = bind_value %.loc22_15.1
// CHECK:STDOUT:   %self.ref.loc22_19: i32 = name_ref r#self, %self.loc21_24
// CHECK:STDOUT:   %.loc22_25.1: (i32, i32) = tuple_literal (%.loc22_15.2, %self.ref.loc22_19)
// CHECK:STDOUT:   %.loc22_25.2: ref i32 = tuple_access %return, element0
// CHECK:STDOUT:   %.loc22_25.3: init i32 = initialize_from %.loc22_15.2 to %.loc22_25.2
// CHECK:STDOUT:   %.loc22_25.4: ref i32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc22_25.5: init i32 = initialize_from %self.ref.loc22_19 to %.loc22_25.4
// CHECK:STDOUT:   %.loc22_25.6: init (i32, i32) = tuple_init (%.loc22_25.3, %.loc22_25.5) to %return
// CHECK:STDOUT:   %.loc22_26: init (i32, i32) = converted %.loc22_25.1, %.loc22_25.6
// CHECK:STDOUT:   return %.loc22_26 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
