// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/template/member_out_of_line.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/template/member_out_of_line.carbon

// --- fail_todo_basic.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
class Class(template T: type) {
  fn F(n: T) -> T;
  fn G(self) -> T;
  var n: T;
}

// CHECK:STDERR: fail_todo_basic.carbon:[[@LINE+3]]:30-35: error: type `<pattern for <splice of <cannot stringify inst7800007A: {kind: ConvertToValueAction, arg0: inst78000077, arg1: inst(TypeType), type: type(inst(InstType))}>>>` of parameter 1 in redeclaration differs from previous parameter type `<pattern for T>` [RedeclParamDiffersType]
// CHECK:STDERR: fail_todo_basic.carbon:[[@LINE-6]]:8-11: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR:
fn Class(template T: type).F(n: T.U) -> T.U {
  return n;
}

// CHECK:STDERR: fail_todo_basic.carbon:[[@LINE+3]]:1-43: error: function redeclaration differs because return type is `<splice of <cannot stringify inst780000D8: {kind: ConvertToValueAction, arg0: inst780000D5, arg1: inst(TypeType), type: type(inst(InstType))}>>` [FunctionRedeclReturnTypeDiffers]
// CHECK:STDERR: fail_todo_basic.carbon:[[@LINE-12]]:3-18: note: previously declared with return type `T` [FunctionRedeclReturnTypePrevious]
// CHECK:STDERR:
fn Class(template T: type).G(self) -> T.U {
  return self.n;
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- fail_todo_basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [template]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [template]
// CHECK:STDOUT:   %n.param_patt.fb2: %pattern_type.51d = value_param_pattern [template]
// CHECK:STDOUT:   %n.patt.955: %pattern_type.51d = wrapper_binding_pattern n, %n.param_patt.fb2 [template]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [template]
// CHECK:STDOUT:   %return.param_patt.41d: %pattern_type.51d = out_param_pattern [template]
// CHECK:STDOUT:   %return.patt.b39: %pattern_type.51d = return_slot_pattern %return.param_patt.41d, %T [template]
// CHECK:STDOUT:   %Class.F.type.9829b4.1: type = fn_type @Class.F.loc6, @Class(%T) [template]
// CHECK:STDOUT:   %Class.F.c6358f.1: %Class.F.type.9829b4.1 = struct_value () [template]
// CHECK:STDOUT:   %pattern_type.36e: type = pattern_type %Class [template]
// CHECK:STDOUT:   %self.param_patt: %pattern_type.36e = value_param_pattern [template]
// CHECK:STDOUT:   %self.patt: %pattern_type.36e = wrapper_binding_pattern self, %self.param_patt [template]
// CHECK:STDOUT:   %Class.G.type.ea69e8.1: type = fn_type @Class.G.loc7, @Class(%T) [template]
// CHECK:STDOUT:   %Class.G.5aff6a.1: %Class.G.type.ea69e8.1 = struct_value () [template]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [template]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T [template]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %T} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n [template]
// CHECK:STDOUT:   %.eff391.1: type = type_of_inst @Class.F.loc14.%.loc14_34.1 [template]
// CHECK:STDOUT:   %.b66113.1: %.eff391.1 = splice_inst @Class.F.loc14.%.loc14_34.1 [template]
// CHECK:STDOUT:   %.5372ba.1: type = splice_inst @Class.F.loc14.%.loc14_34.4 [template]
// CHECK:STDOUT:   %pattern_type.fcb8f3.1: type = pattern_type %.5372ba.1 [template]
// CHECK:STDOUT:   %n.param_patt.110b44.1: %pattern_type.fcb8f3.1 = value_param_pattern [template]
// CHECK:STDOUT:   %n.patt.9e3e22.1: %pattern_type.fcb8f3.1 = wrapper_binding_pattern n, %n.param_patt.110b44.1 [template]
// CHECK:STDOUT:   %.eff391.2: type = type_of_inst @Class.F.loc14.%.loc14_42.1 [template]
// CHECK:STDOUT:   %.b66113.2: %.eff391.2 = splice_inst @Class.F.loc14.%.loc14_42.1 [template]
// CHECK:STDOUT:   %.5372ba.2: type = splice_inst @Class.F.loc14.%.loc14_42.4 [template]
// CHECK:STDOUT:   %.cd79f1.1: Core.Form = init_form %.5372ba.2 [template]
// CHECK:STDOUT:   %pattern_type.fcb8f3.2: type = pattern_type %.5372ba.2 [template]
// CHECK:STDOUT:   %return.param_patt.2e9ac5.1: %pattern_type.fcb8f3.2 = out_param_pattern [template]
// CHECK:STDOUT:   %return.patt.7fc6a1.1: %pattern_type.fcb8f3.2 = return_slot_pattern %return.param_patt.2e9ac5.1, %.5372ba.2 [template]
// CHECK:STDOUT:   %.9af: <instruction> = access_member_action %T, U [template]
// CHECK:STDOUT:   %.eff391.3: type = type_of_inst %.9af [template]
// CHECK:STDOUT:   %.b66113.3: %.eff391.3 = splice_inst %.9af [template]
// CHECK:STDOUT:   %.c8c: <instruction> = convert_to_value_action %.b66113.3, type [template]
// CHECK:STDOUT:   %.5372ba.3: type = splice_inst %.c8c [template]
// CHECK:STDOUT:   %pattern_type.fcb8f3.3: type = pattern_type %.5372ba.3 [template]
// CHECK:STDOUT:   %n.param_patt.110b44.2: %pattern_type.fcb8f3.3 = value_param_pattern [template]
// CHECK:STDOUT:   %n.patt.9e3e22.2: %pattern_type.fcb8f3.3 = wrapper_binding_pattern n, %n.param_patt.110b44.2 [template]
// CHECK:STDOUT:   %.cd79f1.2: Core.Form = init_form %.5372ba.3 [template]
// CHECK:STDOUT:   %return.param_patt.2e9ac5.2: %pattern_type.fcb8f3.3 = out_param_pattern [template]
// CHECK:STDOUT:   %return.patt.7fc6a1.2: %pattern_type.fcb8f3.3 = return_slot_pattern %return.param_patt.2e9ac5.2, %.5372ba.3 [template]
// CHECK:STDOUT:   %Class.F.type.9829b4.2: type = fn_type @Class.F.loc14, @Class(%T) [template]
// CHECK:STDOUT:   %Class.F.c6358f.2: %Class.F.type.9829b4.2 = struct_value () [template]
// CHECK:STDOUT:   %require_complete.cb0614.1: <witness> = require_complete_type %.5372ba.1 [template]
// CHECK:STDOUT:   %require_complete.cb0614.2: <witness> = require_complete_type %.5372ba.2 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (<instruction>) [concrete]
// CHECK:STDOUT:   %.5d6323.1: %.5372ba.2 = splice_inst @Class.F.loc14.%tuple.elem0 [template]
// CHECK:STDOUT:   %.eff391.4: type = type_of_inst @Class.G.loc21.%.loc21_40.1 [template]
// CHECK:STDOUT:   %.b66113.4: %.eff391.4 = splice_inst @Class.G.loc21.%.loc21_40.1 [template]
// CHECK:STDOUT:   %.5372ba.4: type = splice_inst @Class.G.loc21.%.loc21_40.4 [template]
// CHECK:STDOUT:   %.cd79f1.3: Core.Form = init_form %.5372ba.4 [template]
// CHECK:STDOUT:   %pattern_type.fcb8f3.4: type = pattern_type %.5372ba.4 [template]
// CHECK:STDOUT:   %return.param_patt.2e9ac5.3: %pattern_type.fcb8f3.4 = out_param_pattern [template]
// CHECK:STDOUT:   %return.patt.7fc6a1.3: %pattern_type.fcb8f3.4 = return_slot_pattern %return.param_patt.2e9ac5.3, %.5372ba.4 [template]
// CHECK:STDOUT:   %Class.G.type.ea69e8.2: type = fn_type @Class.G.loc21, @Class(%T) [template]
// CHECK:STDOUT:   %Class.G.5aff6a.2: %Class.G.type.ea69e8.2 = struct_value () [template]
// CHECK:STDOUT:   %require_complete.4d9: <witness> = require_complete_type %Class [template]
// CHECK:STDOUT:   %require_complete.cb0614.3: <witness> = require_complete_type %.5372ba.4 [template]
// CHECK:STDOUT:   %.237: type = type_of_inst @Class.G.loc21.%.loc22_14.3 [template]
// CHECK:STDOUT:   %.f96: %.237 = splice_inst @Class.G.loc21.%.loc22_14.3 [template]
// CHECK:STDOUT:   %.5d6323.2: %.5372ba.4 = splice_inst @Class.G.loc21.%tuple.elem0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt.loc5_23.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_23.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_25.1: type = splice_block %.loc5_25.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc5_25.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_23.2: type = symbolic_binding T, 0, template [template = %T.loc5_23.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type.9829b4.2 = fn_decl @Class.F.loc14 [template = constants.%Class.F.c6358f.2] {
// CHECK:STDOUT:     %n.param_patt.loc14_31.1: @Class.F.loc14.%pattern_type.loc14_31 (%pattern_type.fcb8f3.1) = value_param_pattern [template = %n.param_patt.loc14_31.2 (constants.%n.param_patt.110b44.1)]
// CHECK:STDOUT:     %n.patt.loc14_31.1: @Class.F.loc14.%pattern_type.loc14_31 (%pattern_type.fcb8f3.1) = wrapper_binding_pattern n, %n.param_patt.loc14_31.1 [template = %n.patt.loc14_31.2 (constants.%n.patt.9e3e22.1)]
// CHECK:STDOUT:     %return.param_patt.loc14_42.1: @Class.F.loc14.%pattern_type.loc14_42 (%pattern_type.fcb8f3.2) = out_param_pattern [template = %return.param_patt.loc14_42.2 (constants.%return.param_patt.2e9ac5.1)]
// CHECK:STDOUT:     %return.patt.loc14_38.1: @Class.F.loc14.%pattern_type.loc14_42 (%pattern_type.fcb8f3.2) = return_slot_pattern %return.param_patt.loc14_42.1, %.loc14_42.9 [template = %return.patt.loc14_38.2 (constants.%return.patt.7fc6a1.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_22.1: type = splice_block %.loc14_22.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc14_22.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc14_20: type = symbolic_binding T, 0, template [template = @Class.%T.loc5_23.1 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc14_41: type = name_ref T, %T.loc14_20 [template = %T.loc14_33 (constants.%T)]
// CHECK:STDOUT:     %.loc14_42.7: type = type_of_inst %.loc14_42.1 [template = %.loc14_42.2 (constants.%.eff391.2)]
// CHECK:STDOUT:     %.loc14_42.8: @Class.F.loc14.%.loc14_42.2 (%.eff391.2) = splice_inst %.loc14_42.1 [template = %.loc14_42.3 (constants.%.b66113.2)]
// CHECK:STDOUT:     %.loc14_42.9: type = splice_inst %.loc14_42.4 [template = %.loc14_42.5 (constants.%.5372ba.2)]
// CHECK:STDOUT:     %.loc14_42.10: Core.Form = init_form %.loc14_42.9 [template = %.loc14_42.6 (constants.%.cd79f1.1)]
// CHECK:STDOUT:     %n.param: @Class.F.loc14.%.loc14_34.5 (%.5372ba.1) = value_param call_param0
// CHECK:STDOUT:     %.loc14_34.6: type = splice_block %.loc14_34.9 [template = %.loc14_34.5 (constants.%.5372ba.1)] {
// CHECK:STDOUT:       %T.ref.loc14_33: type = name_ref T, %T.loc14_20 [template = %T.loc14_33 (constants.%T)]
// CHECK:STDOUT:       %.loc14_34.7: type = type_of_inst %.loc14_34.1 [template = %.loc14_34.2 (constants.%.eff391.1)]
// CHECK:STDOUT:       %.loc14_34.8: @Class.F.loc14.%.loc14_34.2 (%.eff391.1) = splice_inst %.loc14_34.1 [template = %.loc14_34.3 (constants.%.b66113.1)]
// CHECK:STDOUT:       %.loc14_34.9: type = splice_inst %.loc14_34.4 [template = %.loc14_34.5 (constants.%.5372ba.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: @Class.F.loc14.%.loc14_34.5 (%.5372ba.1) = wrapper_binding n, %n.param
// CHECK:STDOUT:     %return.param: ref @Class.F.loc14.%.loc14_42.5 (%.5372ba.2) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Class.F.loc14.%.loc14_42.5 (%.5372ba.2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.G.decl: %Class.G.type.ea69e8.2 = fn_decl @Class.G.loc21 [template = constants.%Class.G.5aff6a.2] {
// CHECK:STDOUT:     %self.param_patt.loc21_30.1: @Class.G.loc21.%pattern_type.loc21_30 (%pattern_type.36e) = value_param_pattern [template = %self.param_patt.loc21_30.2 (constants.%self.param_patt)]
// CHECK:STDOUT:     %self.patt.loc21_30.1: @Class.G.loc21.%pattern_type.loc21_30 (%pattern_type.36e) = wrapper_binding_pattern self, %self.param_patt.loc21_30.1 [template = %self.patt.loc21_30.2 (constants.%self.patt)]
// CHECK:STDOUT:     %return.param_patt.loc21_40.1: @Class.G.loc21.%pattern_type.loc21_40 (%pattern_type.fcb8f3.4) = out_param_pattern [template = %return.param_patt.loc21_40.2 (constants.%return.param_patt.2e9ac5.3)]
// CHECK:STDOUT:     %return.patt.loc21_36.1: @Class.G.loc21.%pattern_type.loc21_40 (%pattern_type.fcb8f3.4) = return_slot_pattern %return.param_patt.loc21_40.1, %.loc21_40.9 [template = %return.patt.loc21_36.2 (constants.%return.patt.7fc6a1.3)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc21_22.1: type = splice_block %.loc21_22.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc21_22.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc21_20: type = symbolic_binding T, 0, template [template = @Class.%T.loc5_23.1 (constants.%T)]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc21_20 [template = %T.loc21_30 (constants.%T)]
// CHECK:STDOUT:     %.loc21_40.7: type = type_of_inst %.loc21_40.1 [template = %.loc21_40.2 (constants.%.eff391.4)]
// CHECK:STDOUT:     %.loc21_40.8: @Class.G.loc21.%.loc21_40.2 (%.eff391.4) = splice_inst %.loc21_40.1 [template = %.loc21_40.3 (constants.%.b66113.4)]
// CHECK:STDOUT:     %.loc21_40.9: type = splice_inst %.loc21_40.4 [template = %.loc21_40.5 (constants.%.5372ba.4)]
// CHECK:STDOUT:     %.loc21_40.10: Core.Form = init_form %.loc21_40.9 [template = %.loc21_40.6 (constants.%.cd79f1.3)]
// CHECK:STDOUT:     %self.param: @Class.G.loc21.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:     %.loc21_30.1: type = splice_block %Self.ref [template = %Class (constants.%Class)] {
// CHECK:STDOUT:       %.loc21_30.2: type = specific_constant constants.%Class, @Class(constants.%T) [template = %Class (constants.%Class)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc21_30.2 [template = %Class (constants.%Class)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Class.G.loc21.%Class (%Class) = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref @Class.G.loc21.%.loc21_40.5 (%.5372ba.4) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Class.G.loc21.%.loc21_40.5 (%.5372ba.4) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc5_23.2: type) {
// CHECK:STDOUT:   %T.patt.loc5_23.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_23.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc5_23.1: type = symbolic_binding T, 0, template [template = %T.loc5_23.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F.loc6, @Class(%T.loc5_23.1) [template = %Class.F.type (constants.%Class.F.type.9829b4.1)]
// CHECK:STDOUT:   %Class.F: @Class.%Class.F.type (%Class.F.type.9829b4.1) = struct_value () [template = %Class.F (constants.%Class.F.c6358f.1)]
// CHECK:STDOUT:   %Class.G.type: type = fn_type @Class.G.loc7, @Class(%T.loc5_23.1) [template = %Class.G.type (constants.%Class.G.type.ea69e8.1)]
// CHECK:STDOUT:   %Class.G: @Class.%Class.G.type (%Class.G.type.ea69e8.1) = struct_value () [template = %Class.G (constants.%Class.G.5aff6a.1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc5_23.1 [template = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc5_23.1) [template = %Class (constants.%Class)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.loc5_23.1 [template = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: @Class.%T.loc5_23.1 (%T)} [template = %struct_type.n (constants.%struct_type.n)]
// CHECK:STDOUT:   %complete_type.loc9_1.2: <witness> = complete_type_witness %struct_type.n [template = %complete_type.loc9_1.2 (constants.%complete_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Class.F.decl: @Class.%Class.F.type (%Class.F.type.9829b4.1) = fn_decl @Class.F.loc6 [template = @Class.%Class.F (constants.%Class.F.c6358f.1)] {
// CHECK:STDOUT:       %n.param_patt.loc6_9.1: @Class.F.loc6.%pattern_type (%pattern_type.51d) = value_param_pattern [template = %n.param_patt.loc6_9.2 (constants.%n.param_patt.fb2)]
// CHECK:STDOUT:       %n.patt.loc6_9.1: @Class.F.loc6.%pattern_type (%pattern_type.51d) = wrapper_binding_pattern n, %n.param_patt.loc6_9.1 [template = %n.patt.loc6_9.2 (constants.%n.patt.955)]
// CHECK:STDOUT:       %return.param_patt.loc6_17.1: @Class.F.loc6.%pattern_type (%pattern_type.51d) = out_param_pattern [template = %return.param_patt.loc6_17.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:       %return.patt.loc6_14.1: @Class.F.loc6.%pattern_type (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc6_17.1, %T.ref.loc6_17 [template = %return.patt.loc6_14.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref.loc6_17: type = name_ref T, @Class.%T.loc5_23.2 [template = %T (constants.%T)]
// CHECK:STDOUT:       %.loc6_17.2: Core.Form = init_form %T.ref.loc6_17 [template = %.loc6_17.1 (constants.%.184)]
// CHECK:STDOUT:       %n.param: @Class.F.loc6.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %T.ref.loc6_11: type = name_ref T, @Class.%T.loc5_23.2 [template = %T (constants.%T)]
// CHECK:STDOUT:       %n: @Class.F.loc6.%T (%T) = wrapper_binding n, %n.param
// CHECK:STDOUT:       %return.param: ref @Class.F.loc6.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Class.F.loc6.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Class.G.decl: @Class.%Class.G.type (%Class.G.type.ea69e8.1) = fn_decl @Class.G.loc7 [template = @Class.%Class.G (constants.%Class.G.5aff6a.1)] {
// CHECK:STDOUT:       %self.param_patt.loc7_8.1: @Class.G.loc7.%pattern_type.loc7_8 (%pattern_type.36e) = value_param_pattern [template = %self.param_patt.loc7_8.2 (constants.%self.param_patt)]
// CHECK:STDOUT:       %self.patt.loc7_8.1: @Class.G.loc7.%pattern_type.loc7_8 (%pattern_type.36e) = wrapper_binding_pattern self, %self.param_patt.loc7_8.1 [template = %self.patt.loc7_8.2 (constants.%self.patt)]
// CHECK:STDOUT:       %return.param_patt.loc7_17.1: @Class.G.loc7.%pattern_type.loc7_17 (%pattern_type.51d) = out_param_pattern [template = %return.param_patt.loc7_17.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:       %return.patt.loc7_14.1: @Class.G.loc7.%pattern_type.loc7_17 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc7_17.1, %T.ref [template = %return.patt.loc7_14.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc5_23.2 [template = %T (constants.%T)]
// CHECK:STDOUT:       %.loc7_17.2: Core.Form = init_form %T.ref [template = %.loc7_17.1 (constants.%.184)]
// CHECK:STDOUT:       %self.param: @Class.G.loc7.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:       %.loc7_8.1: type = splice_block %Self.ref [template = %Class (constants.%Class)] {
// CHECK:STDOUT:         %.loc7_8.2: type = specific_constant constants.%Class, @Class(constants.%T) [template = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc7_8.2 [template = %Class (constants.%Class)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Class.G.loc7.%Class (%Class) = wrapper_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @Class.G.loc7.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Class.G.loc7.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %field_decl: @Class.%Class.elem (%Class.elem) = field_decl n, element0, %T.ref in [concrete] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc5_23.2 [template = %T.loc5_23.1 (constants.%T)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type.loc9_1.1: <witness> = complete_type_witness constants.%struct_type.n [template = %complete_type.loc9_1.2 (constants.%complete_type)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc9_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %Class.F.decl
// CHECK:STDOUT:     .G = %Class.G.decl
// CHECK:STDOUT:     .n = %field_decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.F.loc6(@Class.%T.loc5_23.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0, template [template = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [template = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %n.param_patt.loc6_9.2: @Class.F.loc6.%pattern_type (%pattern_type.51d) = value_param_pattern [template = %n.param_patt.loc6_9.2 (constants.%n.param_patt.fb2)]
// CHECK:STDOUT:   %n.patt.loc6_9.2: @Class.F.loc6.%pattern_type (%pattern_type.51d) = wrapper_binding_pattern n, %n.param_patt.loc6_9.2 [template = %n.patt.loc6_9.2 (constants.%n.patt.955)]
// CHECK:STDOUT:   %.loc6_17.1: Core.Form = init_form %T [template = %.loc6_17.1 (constants.%.184)]
// CHECK:STDOUT:   %return.param_patt.loc6_17.2: @Class.F.loc6.%pattern_type (%pattern_type.51d) = out_param_pattern [template = %return.param_patt.loc6_17.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt.loc6_14.2: @Class.F.loc6.%pattern_type (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc6_17.2, %T [template = %return.patt.loc6_14.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param: @Class.F.loc6.%T (%T)) -> out %return.param: @Class.F.loc6.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.G.loc7(@Class.%T.loc5_23.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0, template [template = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [template = %Class (constants.%Class)]
// CHECK:STDOUT:   %pattern_type.loc7_8: type = pattern_type %Class [template = %pattern_type.loc7_8 (constants.%pattern_type.36e)]
// CHECK:STDOUT:   %self.param_patt.loc7_8.2: @Class.G.loc7.%pattern_type.loc7_8 (%pattern_type.36e) = value_param_pattern [template = %self.param_patt.loc7_8.2 (constants.%self.param_patt)]
// CHECK:STDOUT:   %self.patt.loc7_8.2: @Class.G.loc7.%pattern_type.loc7_8 (%pattern_type.36e) = wrapper_binding_pattern self, %self.param_patt.loc7_8.2 [template = %self.patt.loc7_8.2 (constants.%self.patt)]
// CHECK:STDOUT:   %.loc7_17.1: Core.Form = init_form %T [template = %.loc7_17.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc7_17: type = pattern_type %T [template = %pattern_type.loc7_17 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt.loc7_17.2: @Class.G.loc7.%pattern_type.loc7_17 (%pattern_type.51d) = out_param_pattern [template = %return.param_patt.loc7_17.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt.loc7_14.2: @Class.G.loc7.%pattern_type.loc7_17 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc7_17.2, %T [template = %return.patt.loc7_14.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.G.loc7.%Class (%Class)) -> out %return.param: @Class.G.loc7.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.F.loc14(%T.loc14_20: type) {
// CHECK:STDOUT:   %T.loc14_33: type = symbolic_binding T, 0, template [template = %T.loc14_33 (constants.%T)]
// CHECK:STDOUT:   %.loc14_34.1: <instruction> = access_member_action %T.ref.loc14_33, U [template]
// CHECK:STDOUT:   %.loc14_34.2: type = type_of_inst %.loc14_34.1 [template = %.loc14_34.2 (constants.%.eff391.1)]
// CHECK:STDOUT:   %.loc14_34.3: @Class.F.loc14.%.loc14_34.2 (%.eff391.1) = splice_inst %.loc14_34.1 [template = %.loc14_34.3 (constants.%.b66113.1)]
// CHECK:STDOUT:   %.loc14_34.4: <instruction> = convert_to_value_action %.loc14_34.8, type [template]
// CHECK:STDOUT:   %.loc14_34.5: type = splice_inst %.loc14_34.4 [template = %.loc14_34.5 (constants.%.5372ba.1)]
// CHECK:STDOUT:   %pattern_type.loc14_31: type = pattern_type %.loc14_34.5 [template = %pattern_type.loc14_31 (constants.%pattern_type.fcb8f3.1)]
// CHECK:STDOUT:   %n.param_patt.loc14_31.2: @Class.F.loc14.%pattern_type.loc14_31 (%pattern_type.fcb8f3.1) = value_param_pattern [template = %n.param_patt.loc14_31.2 (constants.%n.param_patt.110b44.1)]
// CHECK:STDOUT:   %n.patt.loc14_31.2: @Class.F.loc14.%pattern_type.loc14_31 (%pattern_type.fcb8f3.1) = wrapper_binding_pattern n, %n.param_patt.loc14_31.2 [template = %n.patt.loc14_31.2 (constants.%n.patt.9e3e22.1)]
// CHECK:STDOUT:   %.loc14_42.1: <instruction> = access_member_action %T.ref.loc14_41, U [template]
// CHECK:STDOUT:   %.loc14_42.2: type = type_of_inst %.loc14_42.1 [template = %.loc14_42.2 (constants.%.eff391.2)]
// CHECK:STDOUT:   %.loc14_42.3: @Class.F.loc14.%.loc14_42.2 (%.eff391.2) = splice_inst %.loc14_42.1 [template = %.loc14_42.3 (constants.%.b66113.2)]
// CHECK:STDOUT:   %.loc14_42.4: <instruction> = convert_to_value_action %.loc14_42.8, type [template]
// CHECK:STDOUT:   %.loc14_42.5: type = splice_inst %.loc14_42.4 [template = %.loc14_42.5 (constants.%.5372ba.2)]
// CHECK:STDOUT:   %.loc14_42.6: Core.Form = init_form %.loc14_42.5 [template = %.loc14_42.6 (constants.%.cd79f1.1)]
// CHECK:STDOUT:   %pattern_type.loc14_42: type = pattern_type %.loc14_42.5 [template = %pattern_type.loc14_42 (constants.%pattern_type.fcb8f3.2)]
// CHECK:STDOUT:   %return.param_patt.loc14_42.2: @Class.F.loc14.%pattern_type.loc14_42 (%pattern_type.fcb8f3.2) = out_param_pattern [template = %return.param_patt.loc14_42.2 (constants.%return.param_patt.2e9ac5.1)]
// CHECK:STDOUT:   %return.patt.loc14_38.2: @Class.F.loc14.%pattern_type.loc14_42 (%pattern_type.fcb8f3.2) = return_slot_pattern %return.param_patt.loc14_42.2, %.loc14_42.5 [template = %return.patt.loc14_38.2 (constants.%return.patt.7fc6a1.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc14_31: <witness> = require_complete_type %.loc14_34.5 [template = %require_complete.loc14_31 (constants.%require_complete.cb0614.1)]
// CHECK:STDOUT:   %require_complete.loc14_38: <witness> = require_complete_type %.loc14_42.5 [template = %require_complete.loc14_38 (constants.%require_complete.cb0614.2)]
// CHECK:STDOUT:   %.loc15_11.2: %tuple.type = initialize_action %n.ref, %.loc14_42.5, %return.param, false [template]
// CHECK:STDOUT:   %tuple.elem0: <instruction> = tuple_access %.loc15_11.2, element0 [template]
// CHECK:STDOUT:   %.loc15_11.3: @Class.F.loc14.%.loc14_42.5 (%.5372ba.2) = splice_inst %tuple.elem0 [template = %.loc15_11.3 (constants.%.5d6323.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param: @Class.F.loc14.%.loc14_34.5 (%.5372ba.1)) -> out %return.param: @Class.F.loc14.%.loc14_42.5 (%.5372ba.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %n.ref: @Class.F.loc14.%.loc14_34.5 (%.5372ba.1) = name_ref n, %n
// CHECK:STDOUT:     %.loc15_11.1: @Class.F.loc14.%.loc14_42.5 (%.5372ba.2) = splice_inst %tuple.elem0 [template = %.loc15_11.3 (constants.%.5d6323.1)]
// CHECK:STDOUT:     return %.loc15_11.1 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.G.loc21(%T.loc21_20: type) {
// CHECK:STDOUT:   %T.loc21_30: type = symbolic_binding T, 0, template [template = %T.loc21_30 (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc21_30) [template = %Class (constants.%Class)]
// CHECK:STDOUT:   %pattern_type.loc21_30: type = pattern_type %Class [template = %pattern_type.loc21_30 (constants.%pattern_type.36e)]
// CHECK:STDOUT:   %self.param_patt.loc21_30.2: @Class.G.loc21.%pattern_type.loc21_30 (%pattern_type.36e) = value_param_pattern [template = %self.param_patt.loc21_30.2 (constants.%self.param_patt)]
// CHECK:STDOUT:   %self.patt.loc21_30.2: @Class.G.loc21.%pattern_type.loc21_30 (%pattern_type.36e) = wrapper_binding_pattern self, %self.param_patt.loc21_30.2 [template = %self.patt.loc21_30.2 (constants.%self.patt)]
// CHECK:STDOUT:   %.loc21_40.1: <instruction> = access_member_action %T.ref, U [template]
// CHECK:STDOUT:   %.loc21_40.2: type = type_of_inst %.loc21_40.1 [template = %.loc21_40.2 (constants.%.eff391.4)]
// CHECK:STDOUT:   %.loc21_40.3: @Class.G.loc21.%.loc21_40.2 (%.eff391.4) = splice_inst %.loc21_40.1 [template = %.loc21_40.3 (constants.%.b66113.4)]
// CHECK:STDOUT:   %.loc21_40.4: <instruction> = convert_to_value_action %.loc21_40.8, type [template]
// CHECK:STDOUT:   %.loc21_40.5: type = splice_inst %.loc21_40.4 [template = %.loc21_40.5 (constants.%.5372ba.4)]
// CHECK:STDOUT:   %.loc21_40.6: Core.Form = init_form %.loc21_40.5 [template = %.loc21_40.6 (constants.%.cd79f1.3)]
// CHECK:STDOUT:   %pattern_type.loc21_40: type = pattern_type %.loc21_40.5 [template = %pattern_type.loc21_40 (constants.%pattern_type.fcb8f3.4)]
// CHECK:STDOUT:   %return.param_patt.loc21_40.2: @Class.G.loc21.%pattern_type.loc21_40 (%pattern_type.fcb8f3.4) = out_param_pattern [template = %return.param_patt.loc21_40.2 (constants.%return.param_patt.2e9ac5.3)]
// CHECK:STDOUT:   %return.patt.loc21_36.2: @Class.G.loc21.%pattern_type.loc21_40 (%pattern_type.fcb8f3.4) = return_slot_pattern %return.param_patt.loc21_40.2, %.loc21_40.5 [template = %return.patt.loc21_36.2 (constants.%return.patt.7fc6a1.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc21_30: <witness> = require_complete_type %Class [template = %require_complete.loc21_30 (constants.%require_complete.4d9)]
// CHECK:STDOUT:   %require_complete.loc21_36: <witness> = require_complete_type %.loc21_40.5 [template = %require_complete.loc21_36 (constants.%require_complete.cb0614.3)]
// CHECK:STDOUT:   %.loc22_14.3: <instruction> = access_member_action %self.ref, n [template]
// CHECK:STDOUT:   %.loc22_14.4: type = type_of_inst %.loc22_14.3 [template = %.loc22_14.4 (constants.%.237)]
// CHECK:STDOUT:   %.loc22_14.5: @Class.G.loc21.%.loc22_14.4 (%.237) = splice_inst %.loc22_14.3 [template = %.loc22_14.5 (constants.%.f96)]
// CHECK:STDOUT:   %.loc22_16.2: %tuple.type = initialize_action %.loc22_14.2, %.loc21_40.5, %return.param, false [template]
// CHECK:STDOUT:   %tuple.elem0: <instruction> = tuple_access %.loc22_16.2, element0 [template]
// CHECK:STDOUT:   %.loc22_16.3: @Class.G.loc21.%.loc21_40.5 (%.5372ba.4) = splice_inst %tuple.elem0 [template = %.loc22_16.3 (constants.%.5d6323.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.G.loc21.%Class (%Class)) -> out %return.param: @Class.G.loc21.%.loc21_40.5 (%.5372ba.4) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.G.loc21.%Class (%Class) = name_ref self, %self
// CHECK:STDOUT:     %.loc22_14.1: type = type_of_inst %.loc22_14.3 [template = %.loc22_14.4 (constants.%.237)]
// CHECK:STDOUT:     %.loc22_14.2: @Class.G.loc21.%.loc22_14.4 (%.237) = splice_inst %.loc22_14.3 [template = %.loc22_14.5 (constants.%.f96)]
// CHECK:STDOUT:     %.loc22_16.1: @Class.G.loc21.%.loc21_40.5 (%.5372ba.4) = splice_inst %tuple.elem0 [template = %.loc22_16.3 (constants.%.5d6323.2)]
// CHECK:STDOUT:     return %.loc22_16.1 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc5_23.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc5_23.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.F.type => constants.%Class.F.type.9829b4.1
// CHECK:STDOUT:   %Class.F => constants.%Class.F.c6358f.1
// CHECK:STDOUT:   %Class.G.type => constants.%Class.G.type.ea69e8.1
// CHECK:STDOUT:   %Class.G => constants.%Class.G.5aff6a.1
// CHECK:STDOUT:   %require_complete => constants.%require_complete.944
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n
// CHECK:STDOUT:   %complete_type.loc9_1.2 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.F.loc6(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT:   %n.param_patt.loc6_9.2 => constants.%n.param_patt.fb2
// CHECK:STDOUT:   %n.patt.loc6_9.2 => constants.%n.patt.955
// CHECK:STDOUT:   %.loc6_17.1 => constants.%.184
// CHECK:STDOUT:   %return.param_patt.loc6_17.2 => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt.loc6_14.2 => constants.%return.patt.b39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.G.loc7(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %pattern_type.loc7_8 => constants.%pattern_type.36e
// CHECK:STDOUT:   %self.param_patt.loc7_8.2 => constants.%self.param_patt
// CHECK:STDOUT:   %self.patt.loc7_8.2 => constants.%self.patt
// CHECK:STDOUT:   %.loc7_17.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc7_17 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt.loc7_17.2 => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt.loc7_14.2 => constants.%return.patt.b39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.F.loc14(constants.%T) {
// CHECK:STDOUT:   %T.loc14_33 => constants.%T
// CHECK:STDOUT:   %.loc14_34.1 => constants.%.9af
// CHECK:STDOUT:   %.loc14_34.2 => constants.%.eff391.3
// CHECK:STDOUT:   %.loc14_34.3 => constants.%.b66113.3
// CHECK:STDOUT:   %.loc14_34.4 => constants.%.c8c
// CHECK:STDOUT:   %.loc14_34.5 => constants.%.5372ba.3
// CHECK:STDOUT:   %pattern_type.loc14_31 => constants.%pattern_type.fcb8f3.3
// CHECK:STDOUT:   %n.param_patt.loc14_31.2 => constants.%n.param_patt.110b44.2
// CHECK:STDOUT:   %n.patt.loc14_31.2 => constants.%n.patt.9e3e22.2
// CHECK:STDOUT:   %.loc14_42.1 => constants.%.9af
// CHECK:STDOUT:   %.loc14_42.2 => constants.%.eff391.3
// CHECK:STDOUT:   %.loc14_42.3 => constants.%.b66113.3
// CHECK:STDOUT:   %.loc14_42.4 => constants.%.c8c
// CHECK:STDOUT:   %.loc14_42.5 => constants.%.5372ba.3
// CHECK:STDOUT:   %.loc14_42.6 => constants.%.cd79f1.2
// CHECK:STDOUT:   %pattern_type.loc14_42 => constants.%pattern_type.fcb8f3.3
// CHECK:STDOUT:   %return.param_patt.loc14_42.2 => constants.%return.param_patt.2e9ac5.2
// CHECK:STDOUT:   %return.patt.loc14_38.2 => constants.%return.patt.7fc6a1.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.G.loc21(constants.%T) {
// CHECK:STDOUT:   %T.loc21_30 => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %pattern_type.loc21_30 => constants.%pattern_type.36e
// CHECK:STDOUT:   %self.param_patt.loc21_30.2 => constants.%self.param_patt
// CHECK:STDOUT:   %self.patt.loc21_30.2 => constants.%self.patt
// CHECK:STDOUT:   %.loc21_40.1 => constants.%.9af
// CHECK:STDOUT:   %.loc21_40.2 => constants.%.eff391.3
// CHECK:STDOUT:   %.loc21_40.3 => constants.%.b66113.3
// CHECK:STDOUT:   %.loc21_40.4 => constants.%.c8c
// CHECK:STDOUT:   %.loc21_40.5 => constants.%.5372ba.3
// CHECK:STDOUT:   %.loc21_40.6 => constants.%.cd79f1.2
// CHECK:STDOUT:   %pattern_type.loc21_40 => constants.%pattern_type.fcb8f3.3
// CHECK:STDOUT:   %return.param_patt.loc21_40.2 => constants.%return.param_patt.2e9ac5.2
// CHECK:STDOUT:   %return.patt.loc21_36.2 => constants.%return.patt.7fc6a1.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
