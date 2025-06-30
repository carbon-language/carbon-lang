// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %ptr.e71: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %pattern_type.796: type = pattern_type %ptr.e71 [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.761: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.511: type = pattern_type %tuple.type.d07 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.54b: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %self.patt.loc21_17: %pattern_type.796 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt.loc21_21: %pattern_type.796 = value_param_pattern %self.patt.loc21_17, call_param0 [concrete]
// CHECK:STDOUT:     %.loc21_12: %pattern_type.f6d = addr_pattern %self.param_patt.loc21_21 [concrete]
// CHECK:STDOUT:     %self.patt.loc21_30: %pattern_type.7ce = binding_pattern r#self [concrete]
// CHECK:STDOUT:     %self.param_patt.loc21_36: %pattern_type.7ce = value_param_pattern %self.patt.loc21_30, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param.loc21_21: %ptr.e71 = value_param call_param0
// CHECK:STDOUT:     %.loc21_27: type = splice_block %ptr.loc21 [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:       %Self.ref.loc21: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr.loc21: type = ptr_type %Self.ref.loc21 [concrete = constants.%ptr.e71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self.loc21_17: %ptr.e71 = bind_name self, %self.param.loc21_21
// CHECK:STDOUT:     %self.param.loc21_36: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc21_38: type = splice_block %i32.loc21 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc21: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self.loc21_30: %i32 = bind_name r#self, %self.param.loc21_36
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %self.patt.loc25_12: %pattern_type.761 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt.loc25_16: %pattern_type.761 = value_param_pattern %self.patt.loc25_12, call_param0 [concrete]
// CHECK:STDOUT:     %self.patt.loc25_24: %pattern_type.7ce = binding_pattern r#self [concrete]
// CHECK:STDOUT:     %self.param_patt.loc25_30: %pattern_type.7ce = value_param_pattern %self.patt.loc25_24, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.511 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.511 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc25_41: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc25_41: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc25_46: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc25_46: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc25_49.1: %tuple.type.24b = tuple_literal (%i32.loc25_41, %i32.loc25_46)
// CHECK:STDOUT:     %.loc25_49.2: type = converted %.loc25_49.1, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:     %self.param.loc25_16: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc25: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc25_12: %Class = bind_name self, %self.param.loc25_16
// CHECK:STDOUT:     %self.param.loc25_30: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc25_32: type = splice_block %i32.loc25_32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc25_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc25_32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self.loc25_24: %i32 = bind_name r#self, %self.param.loc25_30
// CHECK:STDOUT:     %return.param.loc25: ref %tuple.type.d07 = out_param call_param2
// CHECK:STDOUT:     %return.loc25: ref %tuple.type.d07 = return_slot %return.param.loc25
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %self.patt.loc21_17: %pattern_type.796 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt.loc21_21: %pattern_type.796 = value_param_pattern %self.patt.loc21_17, call_param0 [concrete]
// CHECK:STDOUT:     %.loc21_12: %pattern_type.f6d = addr_pattern %self.param_patt.loc21_21 [concrete]
// CHECK:STDOUT:     %self.patt.loc21_30: %pattern_type.7ce = binding_pattern r#self [concrete]
// CHECK:STDOUT:     %self.param_patt.loc21_36: %pattern_type.7ce = value_param_pattern %self.patt.loc21_30, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param.loc16_17: %ptr.e71 = value_param call_param0
// CHECK:STDOUT:     %.loc16_23: type = splice_block %ptr.loc16 [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:       %Self.ref.loc16: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr.loc16: type = ptr_type %Self.ref.loc16 [concrete = constants.%ptr.e71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self.loc16_13: %ptr.e71 = bind_name self, %self.param.loc16_17
// CHECK:STDOUT:     %self.param.loc16_32: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc16_34: type = splice_block %i32.loc16 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self.loc16_26: %i32 = bind_name r#self, %self.param.loc16_32
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %self.patt.loc25_12: %pattern_type.761 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt.loc25_16: %pattern_type.761 = value_param_pattern %self.patt.loc25_12, call_param0 [concrete]
// CHECK:STDOUT:     %self.patt.loc25_24: %pattern_type.7ce = binding_pattern r#self [concrete]
// CHECK:STDOUT:     %self.param_patt.loc25_30: %pattern_type.7ce = value_param_pattern %self.patt.loc25_24, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.511 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.511 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc17_37: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc17_37: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc17_42: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc17_42: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc17_45.1: %tuple.type.24b = tuple_literal (%i32.loc17_37, %i32.loc17_42)
// CHECK:STDOUT:     %.loc17_45.2: type = converted %.loc17_45.1, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:     %self.param.loc17_12: %Class = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc17: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self.loc17_8: %Class = bind_name self, %self.param.loc17_12
// CHECK:STDOUT:     %self.param.loc17_26: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc17_28: type = splice_block %i32.loc17_28 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc17_28: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc17_28: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self.loc17_20: %i32 = bind_name r#self, %self.param.loc17_26
// CHECK:STDOUT:     %return.param.loc17: ref %tuple.type.d07 = out_param call_param2
// CHECK:STDOUT:     %return.loc17: ref %tuple.type.d07 = return_slot %return.param.loc17
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc18: %Class.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete = constants.%struct_type.n]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n [concrete = constants.%complete_type.54b]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .n = %.loc18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%self.param.loc21_21: %ptr.e71, %self.param.loc21_36: %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref.loc22_5: %ptr.e71 = name_ref self, %self.loc21_17
// CHECK:STDOUT:   %.loc22_4: ref %Class = deref %self.ref.loc22_5
// CHECK:STDOUT:   %n.ref: %Class.elem = name_ref n, @Class.%.loc18 [concrete = @Class.%.loc18]
// CHECK:STDOUT:   %.loc22_10: ref %i32 = class_element_access %.loc22_4, element0
// CHECK:STDOUT:   %self.ref.loc22_15: %i32 = name_ref r#self, %self.loc21_30
// CHECK:STDOUT:   assign %.loc22_10, %self.ref.loc22_15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%self.param.loc25_16: %Class, %self.param.loc25_30: %i32) -> %return.param.loc25: %tuple.type.d07 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref.loc26_11: %Class = name_ref self, %self.loc25_12
// CHECK:STDOUT:   %n.ref: %Class.elem = name_ref n, @Class.%.loc18 [concrete = @Class.%.loc18]
// CHECK:STDOUT:   %.loc26_15.1: ref %i32 = class_element_access %self.ref.loc26_11, element0
// CHECK:STDOUT:   %.loc26_15.2: %i32 = bind_value %.loc26_15.1
// CHECK:STDOUT:   %self.ref.loc26_19: %i32 = name_ref r#self, %self.loc25_24
// CHECK:STDOUT:   %.loc26_25.1: %tuple.type.d07 = tuple_literal (%.loc26_15.2, %self.ref.loc26_19)
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %return.loc25, element0
// CHECK:STDOUT:   %.loc26_25.2: init %i32 = initialize_from %.loc26_15.2 to %tuple.elem0
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access %return.loc25, element1
// CHECK:STDOUT:   %.loc26_25.3: init %i32 = initialize_from %self.ref.loc26_19 to %tuple.elem1
// CHECK:STDOUT:   %.loc26_25.4: init %tuple.type.d07 = tuple_init (%.loc26_25.2, %.loc26_25.3) to %return.loc25
// CHECK:STDOUT:   %.loc26_26: init %tuple.type.d07 = converted %.loc26_25.1, %.loc26_25.4
// CHECK:STDOUT:   return %.loc26_26 to %return.loc25
// CHECK:STDOUT: }
// CHECK:STDOUT:
