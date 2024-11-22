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
// CHECK:STDOUT:   %.1: type = ptr_type %Class [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.2 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = unbound_element_type %Class, %i32 [template]
// CHECK:STDOUT:   %.4: type = struct_type {.n: %i32} [template]
// CHECK:STDOUT:   %.5: <witness> = complete_type_witness %.4 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %self.patt.loc17_17: %.1 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt.loc17_21: %.1 = value_param_pattern %self.patt.loc17_17, runtime_param0
// CHECK:STDOUT:     %.loc17_12: auto = addr_pattern %self.param_patt.loc17_21
// CHECK:STDOUT:     %self.patt.loc17_30: %i32 = binding_pattern r#self
// CHECK:STDOUT:     %self.param_patt.loc17_36: %i32 = value_param_pattern %self.patt.loc17_30, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc17: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %.loc17_27: type = ptr_type %Class [template = constants.%.1]
// CHECK:STDOUT:     %.loc17_38.1: Core.IntLiteral = int_value 32 [template = constants.%.2]
// CHECK:STDOUT:     %int.make_type_signed.loc17: init type = call constants.%Int(%.loc17_38.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc17_38.2: type = value_of_initializer %int.make_type_signed.loc17 [template = constants.%i32]
// CHECK:STDOUT:     %.loc17_38.3: type = converted %int.make_type_signed.loc17, %.loc17_38.2 [template = constants.%i32]
// CHECK:STDOUT:     %self.param.loc17_21: %.1 = value_param runtime_param0
// CHECK:STDOUT:     %self.loc17_17: %.1 = bind_name self, %self.param.loc17_21
// CHECK:STDOUT:     %self.param.loc17_36: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %self.loc17_30: %i32 = bind_name r#self, %self.param.loc17_36
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %self.patt.loc21_12: %Class = binding_pattern self
// CHECK:STDOUT:     %self.param_patt.loc21_16: %Class = value_param_pattern %self.patt.loc21_12, runtime_param0
// CHECK:STDOUT:     %self.patt.loc21_24: %i32 = binding_pattern r#self
// CHECK:STDOUT:     %self.param_patt.loc21_30: %i32 = value_param_pattern %self.patt.loc21_24, runtime_param1
// CHECK:STDOUT:     %return.patt: %tuple.type.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type.2 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc21: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %.loc21_32.1: Core.IntLiteral = int_value 32 [template = constants.%.2]
// CHECK:STDOUT:     %int.make_type_signed.loc21_32: init type = call constants.%Int(%.loc21_32.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_32.2: type = value_of_initializer %int.make_type_signed.loc21_32 [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_32.3: type = converted %int.make_type_signed.loc21_32, %.loc21_32.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_41: Core.IntLiteral = int_value 32 [template = constants.%.2]
// CHECK:STDOUT:     %int.make_type_signed.loc21_41: init type = call constants.%Int(%.loc21_41) [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_46: Core.IntLiteral = int_value 32 [template = constants.%.2]
// CHECK:STDOUT:     %int.make_type_signed.loc21_46: init type = call constants.%Int(%.loc21_46) [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_49.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc21_41, %int.make_type_signed.loc21_46)
// CHECK:STDOUT:     %.loc21_49.2: type = value_of_initializer %int.make_type_signed.loc21_41 [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_49.3: type = converted %int.make_type_signed.loc21_41, %.loc21_49.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_49.4: type = value_of_initializer %int.make_type_signed.loc21_46 [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_49.5: type = converted %int.make_type_signed.loc21_46, %.loc21_49.4 [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_49.6: type = converted %.loc21_49.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:     %self.param.loc21_16: %Class = value_param runtime_param0
// CHECK:STDOUT:     %self.loc21_12: %Class = bind_name self, %self.param.loc21_16
// CHECK:STDOUT:     %self.param.loc21_30: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %self.loc21_24: %i32 = bind_name r#self, %self.param.loc21_30
// CHECK:STDOUT:     %return.param.loc21: ref %tuple.type.2 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %tuple.type.2 = return_slot %return.param.loc21
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %self.patt.loc17_17: %.1 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt.loc17_21: %.1 = value_param_pattern %self.patt.loc17_17, runtime_param0
// CHECK:STDOUT:     %.loc17_12: auto = addr_pattern %self.param_patt.loc17_21
// CHECK:STDOUT:     %self.patt.loc17_30: %i32 = binding_pattern r#self
// CHECK:STDOUT:     %self.param_patt.loc17_36: %i32 = value_param_pattern %self.patt.loc17_30, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc12: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %.loc12_23: type = ptr_type %Class [template = constants.%.1]
// CHECK:STDOUT:     %.loc12_34.1: Core.IntLiteral = int_value 32 [template = constants.%.2]
// CHECK:STDOUT:     %int.make_type_signed.loc12: init type = call constants.%Int(%.loc12_34.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_34.2: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_34.3: type = converted %int.make_type_signed.loc12, %.loc12_34.2 [template = constants.%i32]
// CHECK:STDOUT:     %self.param.loc12_17: %.1 = value_param runtime_param0
// CHECK:STDOUT:     %self.loc12_13: %.1 = bind_name self, %self.param.loc12_17
// CHECK:STDOUT:     %self.param.loc12_32: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %self.loc12_26: %i32 = bind_name r#self, %self.param.loc12_32
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %self.patt.loc21_12: %Class = binding_pattern self
// CHECK:STDOUT:     %self.param_patt.loc21_16: %Class = value_param_pattern %self.patt.loc21_12, runtime_param0
// CHECK:STDOUT:     %self.patt.loc21_24: %i32 = binding_pattern r#self
// CHECK:STDOUT:     %self.param_patt.loc21_30: %i32 = value_param_pattern %self.patt.loc21_24, runtime_param1
// CHECK:STDOUT:     %return.patt: %tuple.type.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type.2 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc13: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %.loc13_28.1: Core.IntLiteral = int_value 32 [template = constants.%.2]
// CHECK:STDOUT:     %int.make_type_signed.loc13_28: init type = call constants.%Int(%.loc13_28.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_28.2: type = value_of_initializer %int.make_type_signed.loc13_28 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_28.3: type = converted %int.make_type_signed.loc13_28, %.loc13_28.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_37: Core.IntLiteral = int_value 32 [template = constants.%.2]
// CHECK:STDOUT:     %int.make_type_signed.loc13_37: init type = call constants.%Int(%.loc13_37) [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_42: Core.IntLiteral = int_value 32 [template = constants.%.2]
// CHECK:STDOUT:     %int.make_type_signed.loc13_42: init type = call constants.%Int(%.loc13_42) [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_45.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc13_37, %int.make_type_signed.loc13_42)
// CHECK:STDOUT:     %.loc13_45.2: type = value_of_initializer %int.make_type_signed.loc13_37 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_45.3: type = converted %int.make_type_signed.loc13_37, %.loc13_45.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_45.4: type = value_of_initializer %int.make_type_signed.loc13_42 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_45.5: type = converted %int.make_type_signed.loc13_42, %.loc13_45.4 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_45.6: type = converted %.loc13_45.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:     %self.param.loc13_12: %Class = value_param runtime_param0
// CHECK:STDOUT:     %self.loc13_8: %Class = bind_name self, %self.param.loc13_12
// CHECK:STDOUT:     %self.param.loc13_26: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %self.loc13_20: %i32 = bind_name r#self, %self.param.loc13_26
// CHECK:STDOUT:     %return.param.loc13: ref %tuple.type.2 = out_param runtime_param2
// CHECK:STDOUT:     %.loc13_33: ref %tuple.type.2 = return_slot %return.param.loc13
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14_10.1: Core.IntLiteral = int_value 32 [template = constants.%.2]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc14_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_10.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_10.3: type = converted %int.make_type_signed, %.loc14_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_8: %.3 = field_decl n, element0 [template]
// CHECK:STDOUT:   %.loc15: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .n = %.loc14_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[addr %self.param_patt.loc17_21: %.1](%self.param_patt.loc17_36: %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref.loc18_5: %.1 = name_ref self, %self.loc17_17
// CHECK:STDOUT:   %.loc18_4: ref %Class = deref %self.ref.loc18_5
// CHECK:STDOUT:   %n.ref: %.3 = name_ref n, @Class.%.loc14_8 [template = @Class.%.loc14_8]
// CHECK:STDOUT:   %.loc18_10: ref %i32 = class_element_access %.loc18_4, element0
// CHECK:STDOUT:   %self.ref.loc18_15: %i32 = name_ref r#self, %self.loc17_30
// CHECK:STDOUT:   assign %.loc18_10, %self.ref.loc18_15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[%self.param_patt.loc21_16: %Class](%self.param_patt.loc21_30: %i32) -> %return: %tuple.type.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref.loc22_11: %Class = name_ref self, %self.loc21_12
// CHECK:STDOUT:   %n.ref: %.3 = name_ref n, @Class.%.loc14_8 [template = @Class.%.loc14_8]
// CHECK:STDOUT:   %.loc22_15.1: ref %i32 = class_element_access %self.ref.loc22_11, element0
// CHECK:STDOUT:   %.loc22_15.2: %i32 = bind_value %.loc22_15.1
// CHECK:STDOUT:   %self.ref.loc22_19: %i32 = name_ref r#self, %self.loc21_24
// CHECK:STDOUT:   %.loc22_25.1: %tuple.type.2 = tuple_literal (%.loc22_15.2, %self.ref.loc22_19)
// CHECK:STDOUT:   %.loc22_25.2: ref %i32 = tuple_access %return, element0
// CHECK:STDOUT:   %.loc22_25.3: init %i32 = initialize_from %.loc22_15.2 to %.loc22_25.2
// CHECK:STDOUT:   %.loc22_25.4: ref %i32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc22_25.5: init %i32 = initialize_from %self.ref.loc22_19 to %.loc22_25.4
// CHECK:STDOUT:   %.loc22_25.6: init %tuple.type.2 = tuple_init (%.loc22_25.3, %.loc22_25.5) to %return
// CHECK:STDOUT:   %.loc22_26: init %tuple.type.2 = converted %.loc22_25.1, %.loc22_25.6
// CHECK:STDOUT:   return %.loc22_26 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
