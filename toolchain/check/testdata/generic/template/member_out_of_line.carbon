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

// CHECK:STDERR:        ╭─┤ fail_todo_basic.carbon:[[@LINE+16]]:30
// CHECK:STDERR:        │   fn F(n: T) -> T;
// CHECK:STDERR:        ·        ──┬─
// CHECK:STDERR:        ·          ╰──┤ previous declaration's corresponding parameter here
// CHECK:STDERR:        ·             │ [RedeclParamPrevious]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ fn Class(template T: type).F(n: T.U) -> T.U {
// CHECK:STDERR:        ·                              ━━━┳━━
// CHECK:STDERR:        · ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
// CHECK:STDERR:        · ┃ error: type
// CHECK:STDERR:        · ┃        `<pattern for <splice of <cannot stringify inst7800007F: {kind: ConvertToValueAction, arg0: inst7800007C, arg1: inst(TypeType), type: type(inst(InstType))}>>>`
// CHECK:STDERR:        · ┃        of parameter 1 in redeclaration differs from previous parameter type
// CHECK:STDERR:        · ┃        `<pattern for T>`
// CHECK:STDERR:        · ┃        [RedeclParamDiffersType]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn Class(template T: type).F(n: T.U) -> T.U {
  return n;
}

// CHECK:STDERR:        ╭─┤ fail_todo_basic.carbon:[[@LINE+14]]:1
// CHECK:STDERR:        │   fn G(self) -> T;
// CHECK:STDERR:        ·   ────────┬───────
// CHECK:STDERR:        ·           ╰──┤ previously declared with return type `T`
// CHECK:STDERR:        ·              │ [FunctionRedeclReturnTypePrevious]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ fn Class(template T: type).G(self) -> T.U {
// CHECK:STDERR:        · ━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        · ┏━━━━━━━━━━━━━━━━━━━━┛
// CHECK:STDERR:        · ┃ error: function redeclaration differs because return type is
// CHECK:STDERR:        · ┃        `<splice of <cannot stringify inst780000DF: {kind: ConvertToValueAction, arg0: inst780000DC, arg1: inst(TypeType), type: type(inst(InstType))}>>`
// CHECK:STDERR:        · ┃        [FunctionRedeclReturnTypeDiffers]
// CHECK:STDERR:        ╰────
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
// CHECK:STDOUT:   %Class.type: type = parameterized_class_name_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [template]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [template]
// CHECK:STDOUT:   %n.param_patt.fb2: %pattern_type.51d = value_param_pattern [template]
// CHECK:STDOUT:   %n.patt.955: %pattern_type.51d = wrapper_binding_pattern n, %n.param_patt.fb2 [template]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [template]
// CHECK:STDOUT:   %return.param_patt.41d: %pattern_type.51d = out_param_pattern [template]
// CHECK:STDOUT:   %return.patt.b39: %pattern_type.51d = return_slot_pattern %return.param_patt.41d, %T [template]
// CHECK:STDOUT:   %Class.F.type.30a249.1: type = fn_type @Class.F.loc6, @Class(%T) [template]
// CHECK:STDOUT:   %Class.F.4883b3.1: %Class.F.type.30a249.1 = struct_value () [template]
// CHECK:STDOUT:   %pattern_type.28f: type = pattern_type %Class [template]
// CHECK:STDOUT:   %self.param_patt: %pattern_type.28f = value_param_pattern [template]
// CHECK:STDOUT:   %self.patt: %pattern_type.28f = wrapper_binding_pattern self, %self.param_patt [template]
// CHECK:STDOUT:   %Class.G.type.f7c942.1: type = fn_type @Class.G.loc7, @Class(%T) [template]
// CHECK:STDOUT:   %Class.G.05f5b3.1: %Class.G.type.f7c942.1 = struct_value () [template]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [template]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T [template]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %T} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n [template]
// CHECK:STDOUT:   %.eff391.1: type = type_of_inst @Class.F.loc27.%.loc27_34.1 [template]
// CHECK:STDOUT:   %.b66113.1: %.eff391.1 = splice_inst @Class.F.loc27.%.loc27_34.1 [template]
// CHECK:STDOUT:   %.5372ba.1: type = splice_inst @Class.F.loc27.%.loc27_34.4 [template]
// CHECK:STDOUT:   %pattern_type.fcb8f3.1: type = pattern_type %.5372ba.1 [template]
// CHECK:STDOUT:   %n.param_patt.110b44.1: %pattern_type.fcb8f3.1 = value_param_pattern [template]
// CHECK:STDOUT:   %n.patt.9e3e22.1: %pattern_type.fcb8f3.1 = wrapper_binding_pattern n, %n.param_patt.110b44.1 [template]
// CHECK:STDOUT:   %.eff391.2: type = type_of_inst @Class.F.loc27.%.loc27_42.1 [template]
// CHECK:STDOUT:   %.b66113.2: %.eff391.2 = splice_inst @Class.F.loc27.%.loc27_42.1 [template]
// CHECK:STDOUT:   %.5372ba.2: type = splice_inst @Class.F.loc27.%.loc27_42.4 [template]
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
// CHECK:STDOUT:   %Class.F.type.30a249.2: type = fn_type @Class.F.loc27, @Class(%T) [template]
// CHECK:STDOUT:   %Class.F.4883b3.2: %Class.F.type.30a249.2 = struct_value () [template]
// CHECK:STDOUT:   %require_complete.cb0614.1: <witness> = require_complete_type %.5372ba.1 [template]
// CHECK:STDOUT:   %require_complete.cb0614.2: <witness> = require_complete_type %.5372ba.2 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (<instruction>) [concrete]
// CHECK:STDOUT:   %.5d6323.1: %.5372ba.2 = splice_inst @Class.F.loc27.%tuple.elem0 [template]
// CHECK:STDOUT:   %.eff391.4: type = type_of_inst @Class.G.loc45.%.loc45_40.1 [template]
// CHECK:STDOUT:   %.b66113.4: %.eff391.4 = splice_inst @Class.G.loc45.%.loc45_40.1 [template]
// CHECK:STDOUT:   %.5372ba.4: type = splice_inst @Class.G.loc45.%.loc45_40.4 [template]
// CHECK:STDOUT:   %.cd79f1.3: Core.Form = init_form %.5372ba.4 [template]
// CHECK:STDOUT:   %pattern_type.fcb8f3.4: type = pattern_type %.5372ba.4 [template]
// CHECK:STDOUT:   %return.param_patt.2e9ac5.3: %pattern_type.fcb8f3.4 = out_param_pattern [template]
// CHECK:STDOUT:   %return.patt.7fc6a1.3: %pattern_type.fcb8f3.4 = return_slot_pattern %return.param_patt.2e9ac5.3, %.5372ba.4 [template]
// CHECK:STDOUT:   %Class.G.type.f7c942.2: type = fn_type @Class.G.loc45, @Class(%T) [template]
// CHECK:STDOUT:   %Class.G.05f5b3.2: %Class.G.type.f7c942.2 = struct_value () [template]
// CHECK:STDOUT:   %require_complete.ac0: <witness> = require_complete_type %Class [template]
// CHECK:STDOUT:   %require_complete.cb0614.3: <witness> = require_complete_type %.5372ba.4 [template]
// CHECK:STDOUT:   %.237: type = type_of_inst @Class.G.loc45.%.loc46_14.3 [template]
// CHECK:STDOUT:   %.f96: %.237 = splice_inst @Class.G.loc45.%.loc46_14.3 [template]
// CHECK:STDOUT:   %.5d6323.2: %.5372ba.4 = splice_inst @Class.G.loc45.%tuple.elem0 [template]
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
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type.30a249.2 = fn_decl @Class.F.loc27 [template = constants.%Class.F.4883b3.2] {
// CHECK:STDOUT:     %n.param_patt.loc27_31.1: @Class.F.loc27.%pattern_type.loc27_31 (%pattern_type.fcb8f3.1) = value_param_pattern [template = %n.param_patt.loc27_31.2 (constants.%n.param_patt.110b44.1)]
// CHECK:STDOUT:     %n.patt.loc27_31.1: @Class.F.loc27.%pattern_type.loc27_31 (%pattern_type.fcb8f3.1) = wrapper_binding_pattern n, %n.param_patt.loc27_31.1 [template = %n.patt.loc27_31.2 (constants.%n.patt.9e3e22.1)]
// CHECK:STDOUT:     %return.param_patt.loc27_42.1: @Class.F.loc27.%pattern_type.loc27_42 (%pattern_type.fcb8f3.2) = out_param_pattern [template = %return.param_patt.loc27_42.2 (constants.%return.param_patt.2e9ac5.1)]
// CHECK:STDOUT:     %return.patt.loc27_38.1: @Class.F.loc27.%pattern_type.loc27_42 (%pattern_type.fcb8f3.2) = return_slot_pattern %return.param_patt.loc27_42.1, %.loc27_42.9 [template = %return.patt.loc27_38.2 (constants.%return.patt.7fc6a1.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc27_22.1: type = splice_block %.loc27_22.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc27_22.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc27_20: type = symbolic_binding T, 0, template [template = @Class.%T.loc5_23.1 (constants.%T)]
// CHECK:STDOUT:     %.loc27_41: type = specific_constant %T.loc27_20, @Class(constants.%T) [template = %T.loc27_33 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc27_41: type = name_ref T, %.loc27_41 [template = %T.loc27_33 (constants.%T)]
// CHECK:STDOUT:     %.loc27_42.7: type = type_of_inst %.loc27_42.1 [template = %.loc27_42.2 (constants.%.eff391.2)]
// CHECK:STDOUT:     %.loc27_42.8: @Class.F.loc27.%.loc27_42.2 (%.eff391.2) = splice_inst %.loc27_42.1 [template = %.loc27_42.3 (constants.%.b66113.2)]
// CHECK:STDOUT:     %.loc27_42.9: type = splice_inst %.loc27_42.4 [template = %.loc27_42.5 (constants.%.5372ba.2)]
// CHECK:STDOUT:     %.loc27_42.10: Core.Form = init_form %.loc27_42.9 [template = %.loc27_42.6 (constants.%.cd79f1.1)]
// CHECK:STDOUT:     %n.param: @Class.F.loc27.%.loc27_34.5 (%.5372ba.1) = value_param call_param0
// CHECK:STDOUT:     %.loc27_34.6: type = splice_block %.loc27_34.9 [template = %.loc27_34.5 (constants.%.5372ba.1)] {
// CHECK:STDOUT:       %.loc27_33: type = specific_constant %T.loc27_20, @Class(constants.%T) [template = %T.loc27_33 (constants.%T)]
// CHECK:STDOUT:       %T.ref.loc27_33: type = name_ref T, %.loc27_33 [template = %T.loc27_33 (constants.%T)]
// CHECK:STDOUT:       %.loc27_34.7: type = type_of_inst %.loc27_34.1 [template = %.loc27_34.2 (constants.%.eff391.1)]
// CHECK:STDOUT:       %.loc27_34.8: @Class.F.loc27.%.loc27_34.2 (%.eff391.1) = splice_inst %.loc27_34.1 [template = %.loc27_34.3 (constants.%.b66113.1)]
// CHECK:STDOUT:       %.loc27_34.9: type = splice_inst %.loc27_34.4 [template = %.loc27_34.5 (constants.%.5372ba.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: @Class.F.loc27.%.loc27_34.5 (%.5372ba.1) = wrapper_binding n, %n.param
// CHECK:STDOUT:     %return.param: ref @Class.F.loc27.%.loc27_42.5 (%.5372ba.2) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Class.F.loc27.%.loc27_42.5 (%.5372ba.2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.G.decl: %Class.G.type.f7c942.2 = fn_decl @Class.G.loc45 [template = constants.%Class.G.05f5b3.2] {
// CHECK:STDOUT:     %self.param_patt.loc45_30.1: @Class.G.loc45.%pattern_type.loc45_30 (%pattern_type.28f) = value_param_pattern [template = %self.param_patt.loc45_30.2 (constants.%self.param_patt)]
// CHECK:STDOUT:     %self.patt.loc45_30.1: @Class.G.loc45.%pattern_type.loc45_30 (%pattern_type.28f) = wrapper_binding_pattern self, %self.param_patt.loc45_30.1 [template = %self.patt.loc45_30.2 (constants.%self.patt)]
// CHECK:STDOUT:     %return.param_patt.loc45_40.1: @Class.G.loc45.%pattern_type.loc45_40 (%pattern_type.fcb8f3.4) = out_param_pattern [template = %return.param_patt.loc45_40.2 (constants.%return.param_patt.2e9ac5.3)]
// CHECK:STDOUT:     %return.patt.loc45_36.1: @Class.G.loc45.%pattern_type.loc45_40 (%pattern_type.fcb8f3.4) = return_slot_pattern %return.param_patt.loc45_40.1, %.loc45_40.9 [template = %return.patt.loc45_36.2 (constants.%return.patt.7fc6a1.3)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc45_22.1: type = splice_block %.loc45_22.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc45_22.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc45_20: type = symbolic_binding T, 0, template [template = @Class.%T.loc5_23.1 (constants.%T)]
// CHECK:STDOUT:     %.loc45_39: type = specific_constant %T.loc45_20, @Class(constants.%T) [template = %T.loc45_30 (constants.%T)]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %.loc45_39 [template = %T.loc45_30 (constants.%T)]
// CHECK:STDOUT:     %.loc45_40.7: type = type_of_inst %.loc45_40.1 [template = %.loc45_40.2 (constants.%.eff391.4)]
// CHECK:STDOUT:     %.loc45_40.8: @Class.G.loc45.%.loc45_40.2 (%.eff391.4) = splice_inst %.loc45_40.1 [template = %.loc45_40.3 (constants.%.b66113.4)]
// CHECK:STDOUT:     %.loc45_40.9: type = splice_inst %.loc45_40.4 [template = %.loc45_40.5 (constants.%.5372ba.4)]
// CHECK:STDOUT:     %.loc45_40.10: Core.Form = init_form %.loc45_40.9 [template = %.loc45_40.6 (constants.%.cd79f1.3)]
// CHECK:STDOUT:     %self.param: @Class.G.loc45.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:     %.loc45_30.1: type = splice_block %Self.ref [template = %Class (constants.%Class)] {
// CHECK:STDOUT:       %.loc45_30.2: type = specific_constant constants.%Class, @Class(constants.%T) [template = %Class (constants.%Class)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc45_30.2 [template = %Class (constants.%Class)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Class.G.loc45.%Class (%Class) = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref @Class.G.loc45.%.loc45_40.5 (%.5372ba.4) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Class.G.loc45.%.loc45_40.5 (%.5372ba.4) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc5_23.2: type) {
// CHECK:STDOUT:   %T.patt.loc5_23.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_23.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc5_23.1: type = symbolic_binding T, 0, template [template = %T.loc5_23.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F.loc6, @Class(%T.loc5_23.1) [template = %Class.F.type (constants.%Class.F.type.30a249.1)]
// CHECK:STDOUT:   %Class.F: @Class.%Class.F.type (%Class.F.type.30a249.1) = struct_value () [template = %Class.F (constants.%Class.F.4883b3.1)]
// CHECK:STDOUT:   %Class.G.type: type = fn_type @Class.G.loc7, @Class(%T.loc5_23.1) [template = %Class.G.type (constants.%Class.G.type.f7c942.1)]
// CHECK:STDOUT:   %Class.G: @Class.%Class.G.type (%Class.G.type.f7c942.1) = struct_value () [template = %Class.G (constants.%Class.G.05f5b3.1)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc5_23.1 [template = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc5_23.1) [template = %Class (constants.%Class)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.loc5_23.1 [template = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: @Class.%T.loc5_23.1 (%T)} [template = %struct_type.n (constants.%struct_type.n)]
// CHECK:STDOUT:   %complete_type.loc9_1.2: <witness> = complete_type_witness %struct_type.n [template = %complete_type.loc9_1.2 (constants.%complete_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Class.F.decl: @Class.%Class.F.type (%Class.F.type.30a249.1) = fn_decl @Class.F.loc6 [template = @Class.%Class.F (constants.%Class.F.4883b3.1)] {
// CHECK:STDOUT:       %n.param_patt.loc6_9.1: @Class.F.loc6.%pattern_type (%pattern_type.51d) = value_param_pattern [template = %n.param_patt.loc6_9.2 (constants.%n.param_patt.fb2)]
// CHECK:STDOUT:       %n.patt.loc6_9.1: @Class.F.loc6.%pattern_type (%pattern_type.51d) = wrapper_binding_pattern n, %n.param_patt.loc6_9.1 [template = %n.patt.loc6_9.2 (constants.%n.patt.955)]
// CHECK:STDOUT:       %return.param_patt.loc6_17.1: @Class.F.loc6.%pattern_type (%pattern_type.51d) = out_param_pattern [template = %return.param_patt.loc6_17.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:       %return.patt.loc6_14.1: @Class.F.loc6.%pattern_type (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc6_17.1, %T.ref.loc6_17 [template = %return.patt.loc6_14.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc6_17.2: type = specific_constant @Class.%T.loc5_23.2, @Class(constants.%T) [template = %T (constants.%T)]
// CHECK:STDOUT:       %T.ref.loc6_17: type = name_ref T, %.loc6_17.2 [template = %T (constants.%T)]
// CHECK:STDOUT:       %.loc6_17.3: Core.Form = init_form %T.ref.loc6_17 [template = %.loc6_17.1 (constants.%.184)]
// CHECK:STDOUT:       %n.param: @Class.F.loc6.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %.loc6_11.1: type = splice_block %T.ref.loc6_11 [template = %T (constants.%T)] {
// CHECK:STDOUT:         %.loc6_11.2: type = specific_constant @Class.%T.loc5_23.2, @Class(constants.%T) [template = %T (constants.%T)]
// CHECK:STDOUT:         %T.ref.loc6_11: type = name_ref T, %.loc6_11.2 [template = %T (constants.%T)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %n: @Class.F.loc6.%T (%T) = wrapper_binding n, %n.param
// CHECK:STDOUT:       %return.param: ref @Class.F.loc6.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Class.F.loc6.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Class.G.decl: @Class.%Class.G.type (%Class.G.type.f7c942.1) = fn_decl @Class.G.loc7 [template = @Class.%Class.G (constants.%Class.G.05f5b3.1)] {
// CHECK:STDOUT:       %self.param_patt.loc7_8.1: @Class.G.loc7.%pattern_type.loc7_8 (%pattern_type.28f) = value_param_pattern [template = %self.param_patt.loc7_8.2 (constants.%self.param_patt)]
// CHECK:STDOUT:       %self.patt.loc7_8.1: @Class.G.loc7.%pattern_type.loc7_8 (%pattern_type.28f) = wrapper_binding_pattern self, %self.param_patt.loc7_8.1 [template = %self.patt.loc7_8.2 (constants.%self.patt)]
// CHECK:STDOUT:       %return.param_patt.loc7_17.1: @Class.G.loc7.%pattern_type.loc7_17 (%pattern_type.51d) = out_param_pattern [template = %return.param_patt.loc7_17.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:       %return.patt.loc7_14.1: @Class.G.loc7.%pattern_type.loc7_17 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc7_17.1, %T.ref [template = %return.patt.loc7_14.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc7_17.2: type = specific_constant @Class.%T.loc5_23.2, @Class(constants.%T) [template = %T (constants.%T)]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %.loc7_17.2 [template = %T (constants.%T)]
// CHECK:STDOUT:       %.loc7_17.3: Core.Form = init_form %T.ref [template = %.loc7_17.1 (constants.%.184)]
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
// CHECK:STDOUT:   %pattern_type.loc7_8: type = pattern_type %Class [template = %pattern_type.loc7_8 (constants.%pattern_type.28f)]
// CHECK:STDOUT:   %self.param_patt.loc7_8.2: @Class.G.loc7.%pattern_type.loc7_8 (%pattern_type.28f) = value_param_pattern [template = %self.param_patt.loc7_8.2 (constants.%self.param_patt)]
// CHECK:STDOUT:   %self.patt.loc7_8.2: @Class.G.loc7.%pattern_type.loc7_8 (%pattern_type.28f) = wrapper_binding_pattern self, %self.param_patt.loc7_8.2 [template = %self.patt.loc7_8.2 (constants.%self.patt)]
// CHECK:STDOUT:   %.loc7_17.1: Core.Form = init_form %T [template = %.loc7_17.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc7_17: type = pattern_type %T [template = %pattern_type.loc7_17 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt.loc7_17.2: @Class.G.loc7.%pattern_type.loc7_17 (%pattern_type.51d) = out_param_pattern [template = %return.param_patt.loc7_17.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt.loc7_14.2: @Class.G.loc7.%pattern_type.loc7_17 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc7_17.2, %T [template = %return.patt.loc7_14.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.G.loc7.%Class (%Class)) -> out %return.param: @Class.G.loc7.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.F.loc27(%T.loc27_20: type) {
// CHECK:STDOUT:   %T.loc27_33: type = symbolic_binding T, 0, template [template = %T.loc27_33 (constants.%T)]
// CHECK:STDOUT:   %.loc27_34.1: <instruction> = access_member_action %T.ref.loc27_33, U [template]
// CHECK:STDOUT:   %.loc27_34.2: type = type_of_inst %.loc27_34.1 [template = %.loc27_34.2 (constants.%.eff391.1)]
// CHECK:STDOUT:   %.loc27_34.3: @Class.F.loc27.%.loc27_34.2 (%.eff391.1) = splice_inst %.loc27_34.1 [template = %.loc27_34.3 (constants.%.b66113.1)]
// CHECK:STDOUT:   %.loc27_34.4: <instruction> = convert_to_value_action %.loc27_34.8, type [template]
// CHECK:STDOUT:   %.loc27_34.5: type = splice_inst %.loc27_34.4 [template = %.loc27_34.5 (constants.%.5372ba.1)]
// CHECK:STDOUT:   %pattern_type.loc27_31: type = pattern_type %.loc27_34.5 [template = %pattern_type.loc27_31 (constants.%pattern_type.fcb8f3.1)]
// CHECK:STDOUT:   %n.param_patt.loc27_31.2: @Class.F.loc27.%pattern_type.loc27_31 (%pattern_type.fcb8f3.1) = value_param_pattern [template = %n.param_patt.loc27_31.2 (constants.%n.param_patt.110b44.1)]
// CHECK:STDOUT:   %n.patt.loc27_31.2: @Class.F.loc27.%pattern_type.loc27_31 (%pattern_type.fcb8f3.1) = wrapper_binding_pattern n, %n.param_patt.loc27_31.2 [template = %n.patt.loc27_31.2 (constants.%n.patt.9e3e22.1)]
// CHECK:STDOUT:   %.loc27_42.1: <instruction> = access_member_action %T.ref.loc27_41, U [template]
// CHECK:STDOUT:   %.loc27_42.2: type = type_of_inst %.loc27_42.1 [template = %.loc27_42.2 (constants.%.eff391.2)]
// CHECK:STDOUT:   %.loc27_42.3: @Class.F.loc27.%.loc27_42.2 (%.eff391.2) = splice_inst %.loc27_42.1 [template = %.loc27_42.3 (constants.%.b66113.2)]
// CHECK:STDOUT:   %.loc27_42.4: <instruction> = convert_to_value_action %.loc27_42.8, type [template]
// CHECK:STDOUT:   %.loc27_42.5: type = splice_inst %.loc27_42.4 [template = %.loc27_42.5 (constants.%.5372ba.2)]
// CHECK:STDOUT:   %.loc27_42.6: Core.Form = init_form %.loc27_42.5 [template = %.loc27_42.6 (constants.%.cd79f1.1)]
// CHECK:STDOUT:   %pattern_type.loc27_42: type = pattern_type %.loc27_42.5 [template = %pattern_type.loc27_42 (constants.%pattern_type.fcb8f3.2)]
// CHECK:STDOUT:   %return.param_patt.loc27_42.2: @Class.F.loc27.%pattern_type.loc27_42 (%pattern_type.fcb8f3.2) = out_param_pattern [template = %return.param_patt.loc27_42.2 (constants.%return.param_patt.2e9ac5.1)]
// CHECK:STDOUT:   %return.patt.loc27_38.2: @Class.F.loc27.%pattern_type.loc27_42 (%pattern_type.fcb8f3.2) = return_slot_pattern %return.param_patt.loc27_42.2, %.loc27_42.5 [template = %return.patt.loc27_38.2 (constants.%return.patt.7fc6a1.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc27_31: <witness> = require_complete_type %.loc27_34.5 [template = %require_complete.loc27_31 (constants.%require_complete.cb0614.1)]
// CHECK:STDOUT:   %require_complete.loc27_38: <witness> = require_complete_type %.loc27_42.5 [template = %require_complete.loc27_38 (constants.%require_complete.cb0614.2)]
// CHECK:STDOUT:   %.loc28_11.2: %tuple.type = initialize_action %n.ref, %.loc27_42.5, %return.param, none, false [template]
// CHECK:STDOUT:   %tuple.elem0: <instruction> = tuple_access %.loc28_11.2, element0 [template]
// CHECK:STDOUT:   %.loc28_11.3: @Class.F.loc27.%.loc27_42.5 (%.5372ba.2) = splice_inst %tuple.elem0 [template = %.loc28_11.3 (constants.%.5d6323.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param: @Class.F.loc27.%.loc27_34.5 (%.5372ba.1)) -> out %return.param: @Class.F.loc27.%.loc27_42.5 (%.5372ba.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %n.ref: @Class.F.loc27.%.loc27_34.5 (%.5372ba.1) = name_ref n, %n
// CHECK:STDOUT:     %.loc28_11.1: @Class.F.loc27.%.loc27_42.5 (%.5372ba.2) = splice_inst %tuple.elem0 [template = %.loc28_11.3 (constants.%.5d6323.1)]
// CHECK:STDOUT:     return %.loc28_11.1 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.G.loc45(%T.loc45_20: type) {
// CHECK:STDOUT:   %T.loc45_30: type = symbolic_binding T, 0, template [template = %T.loc45_30 (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc45_30) [template = %Class (constants.%Class)]
// CHECK:STDOUT:   %pattern_type.loc45_30: type = pattern_type %Class [template = %pattern_type.loc45_30 (constants.%pattern_type.28f)]
// CHECK:STDOUT:   %self.param_patt.loc45_30.2: @Class.G.loc45.%pattern_type.loc45_30 (%pattern_type.28f) = value_param_pattern [template = %self.param_patt.loc45_30.2 (constants.%self.param_patt)]
// CHECK:STDOUT:   %self.patt.loc45_30.2: @Class.G.loc45.%pattern_type.loc45_30 (%pattern_type.28f) = wrapper_binding_pattern self, %self.param_patt.loc45_30.2 [template = %self.patt.loc45_30.2 (constants.%self.patt)]
// CHECK:STDOUT:   %.loc45_40.1: <instruction> = access_member_action %T.ref, U [template]
// CHECK:STDOUT:   %.loc45_40.2: type = type_of_inst %.loc45_40.1 [template = %.loc45_40.2 (constants.%.eff391.4)]
// CHECK:STDOUT:   %.loc45_40.3: @Class.G.loc45.%.loc45_40.2 (%.eff391.4) = splice_inst %.loc45_40.1 [template = %.loc45_40.3 (constants.%.b66113.4)]
// CHECK:STDOUT:   %.loc45_40.4: <instruction> = convert_to_value_action %.loc45_40.8, type [template]
// CHECK:STDOUT:   %.loc45_40.5: type = splice_inst %.loc45_40.4 [template = %.loc45_40.5 (constants.%.5372ba.4)]
// CHECK:STDOUT:   %.loc45_40.6: Core.Form = init_form %.loc45_40.5 [template = %.loc45_40.6 (constants.%.cd79f1.3)]
// CHECK:STDOUT:   %pattern_type.loc45_40: type = pattern_type %.loc45_40.5 [template = %pattern_type.loc45_40 (constants.%pattern_type.fcb8f3.4)]
// CHECK:STDOUT:   %return.param_patt.loc45_40.2: @Class.G.loc45.%pattern_type.loc45_40 (%pattern_type.fcb8f3.4) = out_param_pattern [template = %return.param_patt.loc45_40.2 (constants.%return.param_patt.2e9ac5.3)]
// CHECK:STDOUT:   %return.patt.loc45_36.2: @Class.G.loc45.%pattern_type.loc45_40 (%pattern_type.fcb8f3.4) = return_slot_pattern %return.param_patt.loc45_40.2, %.loc45_40.5 [template = %return.patt.loc45_36.2 (constants.%return.patt.7fc6a1.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc45_30: <witness> = require_complete_type %Class [template = %require_complete.loc45_30 (constants.%require_complete.ac0)]
// CHECK:STDOUT:   %require_complete.loc45_36: <witness> = require_complete_type %.loc45_40.5 [template = %require_complete.loc45_36 (constants.%require_complete.cb0614.3)]
// CHECK:STDOUT:   %.loc46_14.3: <instruction> = access_member_action %self.ref, n [template]
// CHECK:STDOUT:   %.loc46_14.4: type = type_of_inst %.loc46_14.3 [template = %.loc46_14.4 (constants.%.237)]
// CHECK:STDOUT:   %.loc46_14.5: @Class.G.loc45.%.loc46_14.4 (%.237) = splice_inst %.loc46_14.3 [template = %.loc46_14.5 (constants.%.f96)]
// CHECK:STDOUT:   %.loc46_16.2: %tuple.type = initialize_action %.loc46_14.2, %.loc45_40.5, %return.param, none, false [template]
// CHECK:STDOUT:   %tuple.elem0: <instruction> = tuple_access %.loc46_16.2, element0 [template]
// CHECK:STDOUT:   %.loc46_16.3: @Class.G.loc45.%.loc45_40.5 (%.5372ba.4) = splice_inst %tuple.elem0 [template = %.loc46_16.3 (constants.%.5d6323.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.G.loc45.%Class (%Class)) -> out %return.param: @Class.G.loc45.%.loc45_40.5 (%.5372ba.4) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.G.loc45.%Class (%Class) = name_ref self, %self
// CHECK:STDOUT:     %.loc46_14.1: type = type_of_inst %.loc46_14.3 [template = %.loc46_14.4 (constants.%.237)]
// CHECK:STDOUT:     %.loc46_14.2: @Class.G.loc45.%.loc46_14.4 (%.237) = splice_inst %.loc46_14.3 [template = %.loc46_14.5 (constants.%.f96)]
// CHECK:STDOUT:     %.loc46_16.1: @Class.G.loc45.%.loc45_40.5 (%.5372ba.4) = splice_inst %tuple.elem0 [template = %.loc46_16.3 (constants.%.5d6323.2)]
// CHECK:STDOUT:     return %.loc46_16.1 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc5_23.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc5_23.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.F.type => constants.%Class.F.type.30a249.1
// CHECK:STDOUT:   %Class.F => constants.%Class.F.4883b3.1
// CHECK:STDOUT:   %Class.G.type => constants.%Class.G.type.f7c942.1
// CHECK:STDOUT:   %Class.G => constants.%Class.G.05f5b3.1
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
// CHECK:STDOUT:   %pattern_type.loc7_8 => constants.%pattern_type.28f
// CHECK:STDOUT:   %self.param_patt.loc7_8.2 => constants.%self.param_patt
// CHECK:STDOUT:   %self.patt.loc7_8.2 => constants.%self.patt
// CHECK:STDOUT:   %.loc7_17.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc7_17 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt.loc7_17.2 => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt.loc7_14.2 => constants.%return.patt.b39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.F.loc27(constants.%T) {
// CHECK:STDOUT:   %T.loc27_33 => constants.%T
// CHECK:STDOUT:   %.loc27_34.1 => constants.%.9af
// CHECK:STDOUT:   %.loc27_34.2 => constants.%.eff391.3
// CHECK:STDOUT:   %.loc27_34.3 => constants.%.b66113.3
// CHECK:STDOUT:   %.loc27_34.4 => constants.%.c8c
// CHECK:STDOUT:   %.loc27_34.5 => constants.%.5372ba.3
// CHECK:STDOUT:   %pattern_type.loc27_31 => constants.%pattern_type.fcb8f3.3
// CHECK:STDOUT:   %n.param_patt.loc27_31.2 => constants.%n.param_patt.110b44.2
// CHECK:STDOUT:   %n.patt.loc27_31.2 => constants.%n.patt.9e3e22.2
// CHECK:STDOUT:   %.loc27_42.1 => constants.%.9af
// CHECK:STDOUT:   %.loc27_42.2 => constants.%.eff391.3
// CHECK:STDOUT:   %.loc27_42.3 => constants.%.b66113.3
// CHECK:STDOUT:   %.loc27_42.4 => constants.%.c8c
// CHECK:STDOUT:   %.loc27_42.5 => constants.%.5372ba.3
// CHECK:STDOUT:   %.loc27_42.6 => constants.%.cd79f1.2
// CHECK:STDOUT:   %pattern_type.loc27_42 => constants.%pattern_type.fcb8f3.3
// CHECK:STDOUT:   %return.param_patt.loc27_42.2 => constants.%return.param_patt.2e9ac5.2
// CHECK:STDOUT:   %return.patt.loc27_38.2 => constants.%return.patt.7fc6a1.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.G.loc45(constants.%T) {
// CHECK:STDOUT:   %T.loc45_30 => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %pattern_type.loc45_30 => constants.%pattern_type.28f
// CHECK:STDOUT:   %self.param_patt.loc45_30.2 => constants.%self.param_patt
// CHECK:STDOUT:   %self.patt.loc45_30.2 => constants.%self.patt
// CHECK:STDOUT:   %.loc45_40.1 => constants.%.9af
// CHECK:STDOUT:   %.loc45_40.2 => constants.%.eff391.3
// CHECK:STDOUT:   %.loc45_40.3 => constants.%.b66113.3
// CHECK:STDOUT:   %.loc45_40.4 => constants.%.c8c
// CHECK:STDOUT:   %.loc45_40.5 => constants.%.5372ba.3
// CHECK:STDOUT:   %.loc45_40.6 => constants.%.cd79f1.2
// CHECK:STDOUT:   %pattern_type.loc45_40 => constants.%pattern_type.fcb8f3.3
// CHECK:STDOUT:   %return.param_patt.loc45_40.2 => constants.%return.param_patt.2e9ac5.2
// CHECK:STDOUT:   %return.patt.loc45_36.2 => constants.%return.patt.7fc6a1.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
