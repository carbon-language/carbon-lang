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

// CHECK:STDERR: fail_todo_basic.carbon:[[@LINE+7]]:30: error: type `<pattern for <splice of <cannot stringify inst7800007C: {kind: ConvertToValueAction, arg0: inst78000079, arg1: inst(TypeType), type: type(inst(InstType))}>>>` of parameter 1 in redeclaration differs from previous parameter type `<pattern for T>` [RedeclParamDiffersType]
// CHECK:STDERR: fn Class(template T: type).F(n: T.U) -> T.U {
// CHECK:STDERR:                              ^~~~~~
// CHECK:STDERR: fail_todo_basic.carbon:[[@LINE-8]]:8: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR:   fn F(n: T) -> T;
// CHECK:STDERR:        ^~~~
// CHECK:STDERR:
fn Class(template T: type).F(n: T.U) -> T.U {
  return n;
}

// CHECK:STDERR: fail_todo_basic.carbon:[[@LINE+7]]:1: error: function redeclaration differs because return type is `<splice of <cannot stringify inst780000DA: {kind: ConvertToValueAction, arg0: inst780000D7, arg1: inst(TypeType), type: type(inst(InstType))}>>` [FunctionRedeclReturnTypeDiffers]
// CHECK:STDERR: fn Class(template T: type).G(self) -> T.U {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_basic.carbon:[[@LINE-18]]:3: note: previously declared with return type `T` [FunctionRedeclReturnTypePrevious]
// CHECK:STDERR:   fn G(self) -> T;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Class(template T: type).G(self) -> T.U {
  return self.n;
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- fail_todo_basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template]
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
// CHECK:STDOUT:   %.96c438.1: type = type_of_inst @Class.F.loc18.%.loc18_34.1 [template]
// CHECK:STDOUT:   %.8220de.1: %.96c438.1 = splice_inst @Class.F.loc18.%.loc18_34.1 [template]
// CHECK:STDOUT:   %.3cdd63.1: type = splice_inst @Class.F.loc18.%.loc18_34.4 [template]
// CHECK:STDOUT:   %pattern_type.8c644d.1: type = pattern_type %.3cdd63.1 [template]
// CHECK:STDOUT:   %n.param_patt.3a5: %pattern_type.8c644d.1 = value_param_pattern [template]
// CHECK:STDOUT:   %n.patt.10a: %pattern_type.8c644d.1 = wrapper_binding_pattern n, %n.param_patt.3a5 [template]
// CHECK:STDOUT:   %.96c438.2: type = type_of_inst @Class.F.loc18.%.loc18_42.1 [template]
// CHECK:STDOUT:   %.8220de.2: %.96c438.2 = splice_inst @Class.F.loc18.%.loc18_42.1 [template]
// CHECK:STDOUT:   %.3cdd63.2: type = splice_inst @Class.F.loc18.%.loc18_42.4 [template]
// CHECK:STDOUT:   %.a428e6.1: Core.Form = init_form %.3cdd63.2 [template]
// CHECK:STDOUT:   %pattern_type.8c644d.2: type = pattern_type %.3cdd63.2 [template]
// CHECK:STDOUT:   %return.param_patt.eed287.1: %pattern_type.8c644d.2 = out_param_pattern [template]
// CHECK:STDOUT:   %return.patt.7881ad.1: %pattern_type.8c644d.2 = return_slot_pattern %return.param_patt.eed287.1, %.3cdd63.2 [template]
// CHECK:STDOUT:   %.3d9: <instruction> = access_member_action %T, U [template]
// CHECK:STDOUT:   %.cf6: type = type_of_inst %.3d9 [template]
// CHECK:STDOUT:   %.2db: %.cf6 = splice_inst %.3d9 [template]
// CHECK:STDOUT:   %.dd0: <instruction> = convert_to_value_action %.2db, type [template]
// CHECK:STDOUT:   %.3e1: type = splice_inst %.dd0 [template]
// CHECK:STDOUT:   %pattern_type.e0b: type = pattern_type %.3e1 [template]
// CHECK:STDOUT:   %n.param_patt.396: %pattern_type.e0b = value_param_pattern [template]
// CHECK:STDOUT:   %n.patt.f9c: %pattern_type.e0b = wrapper_binding_pattern n, %n.param_patt.396 [template]
// CHECK:STDOUT:   %.764: Core.Form = init_form %.3e1 [template]
// CHECK:STDOUT:   %return.param_patt.fc1: %pattern_type.e0b = out_param_pattern [template]
// CHECK:STDOUT:   %return.patt.fcc: %pattern_type.e0b = return_slot_pattern %return.param_patt.fc1, %.3e1 [template]
// CHECK:STDOUT:   %Class.F.type.9829b4.2: type = fn_type @Class.F.loc18, @Class(%T) [template]
// CHECK:STDOUT:   %Class.F.c6358f.2: %Class.F.type.9829b4.2 = struct_value () [template]
// CHECK:STDOUT:   %require_complete.b6b0e1.1: <witness> = require_complete_type %.3cdd63.1 [template]
// CHECK:STDOUT:   %require_complete.b6b0e1.2: <witness> = require_complete_type %.3cdd63.2 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (<instruction>) [concrete]
// CHECK:STDOUT:   %.c17: %.3cdd63.2 = splice_inst @Class.F.loc18.%tuple.elem0 [template]
// CHECK:STDOUT:   %.96c438.3: type = type_of_inst @Class.G.loc28.%.loc28_40.1 [template]
// CHECK:STDOUT:   %.8220de.3: %.96c438.3 = splice_inst @Class.G.loc28.%.loc28_40.1 [template]
// CHECK:STDOUT:   %.3cdd63.3: type = splice_inst @Class.G.loc28.%.loc28_40.4 [template]
// CHECK:STDOUT:   %.a428e6.2: Core.Form = init_form %.3cdd63.3 [template]
// CHECK:STDOUT:   %pattern_type.8c644d.3: type = pattern_type %.3cdd63.3 [template]
// CHECK:STDOUT:   %return.param_patt.eed287.2: %pattern_type.8c644d.3 = out_param_pattern [template]
// CHECK:STDOUT:   %return.patt.7881ad.2: %pattern_type.8c644d.3 = return_slot_pattern %return.param_patt.eed287.2, %.3cdd63.3 [template]
// CHECK:STDOUT:   %Class.G.type.ea69e8.2: type = fn_type @Class.G.loc28, @Class(%T) [template]
// CHECK:STDOUT:   %Class.G.5aff6a.2: %Class.G.type.ea69e8.2 = struct_value () [template]
// CHECK:STDOUT:   %require_complete.4d9: <witness> = require_complete_type %Class [template]
// CHECK:STDOUT:   %require_complete.b6b0e1.3: <witness> = require_complete_type %.3cdd63.3 [template]
// CHECK:STDOUT:   %.dd9: type = type_of_inst @Class.G.loc28.%.loc29_14.3 [template]
// CHECK:STDOUT:   %.732: %.dd9 = splice_inst @Class.G.loc28.%.loc29_14.3 [template]
// CHECK:STDOUT:   %.b9e: %.3cdd63.3 = splice_inst @Class.G.loc28.%tuple.elem0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt.loc5_23.1: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_23.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_25.1: type = splice_block %.loc5_25.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc5_25.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_23.2: type = symbolic_binding T, 0, template [template = %T.loc5_23.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type.9829b4.2 = fn_decl @Class.F.loc18 [template = constants.%Class.F.c6358f.2] {
// CHECK:STDOUT:     %n.param_patt.loc18_31.1: @Class.F.loc18.%pattern_type.loc18_31 (%pattern_type.8c644d.1) = value_param_pattern [template = %n.param_patt.loc18_31.2 (constants.%n.param_patt.3a5)]
// CHECK:STDOUT:     %n.patt.loc18_31.1: @Class.F.loc18.%pattern_type.loc18_31 (%pattern_type.8c644d.1) = wrapper_binding_pattern n, %n.param_patt.loc18_31.1 [template = %n.patt.loc18_31.2 (constants.%n.patt.10a)]
// CHECK:STDOUT:     %return.param_patt.loc18_42.1: @Class.F.loc18.%pattern_type.loc18_42 (%pattern_type.8c644d.2) = out_param_pattern [template = %return.param_patt.loc18_42.2 (constants.%return.param_patt.eed287.1)]
// CHECK:STDOUT:     %return.patt.loc18_38.1: @Class.F.loc18.%pattern_type.loc18_42 (%pattern_type.8c644d.2) = return_slot_pattern %return.param_patt.loc18_42.1, %.loc18_42.9 [template = %return.patt.loc18_38.2 (constants.%return.patt.7881ad.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18_22.1: type = splice_block %.loc18_22.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc18_22.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc18_20: type = symbolic_binding T, 0, template [template = @Class.%T.loc5_23.1 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc18_41: type = name_ref T, %T.loc18_20 [template = %T.loc18_33 (constants.%T)]
// CHECK:STDOUT:     %.loc18_42.7: type = type_of_inst %.loc18_42.1 [template = %.loc18_42.2 (constants.%.96c438.2)]
// CHECK:STDOUT:     %.loc18_42.8: @Class.F.loc18.%.loc18_42.2 (%.96c438.2) = splice_inst %.loc18_42.1 [template = %.loc18_42.3 (constants.%.8220de.2)]
// CHECK:STDOUT:     %.loc18_42.9: type = splice_inst %.loc18_42.4 [template = %.loc18_42.5 (constants.%.3cdd63.2)]
// CHECK:STDOUT:     %.loc18_42.10: Core.Form = init_form %.loc18_42.9 [template = %.loc18_42.6 (constants.%.a428e6.1)]
// CHECK:STDOUT:     %n.param: @Class.F.loc18.%.loc18_34.5 (%.3cdd63.1) = value_param call_param0
// CHECK:STDOUT:     %.loc18_34.6: type = splice_block %.loc18_34.9 [template = %.loc18_34.5 (constants.%.3cdd63.1)] {
// CHECK:STDOUT:       %T.ref.loc18_33: type = name_ref T, %T.loc18_20 [template = %T.loc18_33 (constants.%T)]
// CHECK:STDOUT:       %.loc18_34.7: type = type_of_inst %.loc18_34.1 [template = %.loc18_34.2 (constants.%.96c438.1)]
// CHECK:STDOUT:       %.loc18_34.8: @Class.F.loc18.%.loc18_34.2 (%.96c438.1) = splice_inst %.loc18_34.1 [template = %.loc18_34.3 (constants.%.8220de.1)]
// CHECK:STDOUT:       %.loc18_34.9: type = splice_inst %.loc18_34.4 [template = %.loc18_34.5 (constants.%.3cdd63.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: @Class.F.loc18.%.loc18_34.5 (%.3cdd63.1) = wrapper_binding n, %n.param
// CHECK:STDOUT:     %return.param: ref @Class.F.loc18.%.loc18_42.5 (%.3cdd63.2) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Class.F.loc18.%.loc18_42.5 (%.3cdd63.2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.G.decl: %Class.G.type.ea69e8.2 = fn_decl @Class.G.loc28 [template = constants.%Class.G.5aff6a.2] {
// CHECK:STDOUT:     %self.param_patt.loc28_30.1: @Class.G.loc28.%pattern_type.loc28_30 (%pattern_type.36e) = value_param_pattern [template = %self.param_patt.loc28_30.2 (constants.%self.param_patt)]
// CHECK:STDOUT:     %self.patt.loc28_30.1: @Class.G.loc28.%pattern_type.loc28_30 (%pattern_type.36e) = wrapper_binding_pattern self, %self.param_patt.loc28_30.1 [template = %self.patt.loc28_30.2 (constants.%self.patt)]
// CHECK:STDOUT:     %return.param_patt.loc28_40.1: @Class.G.loc28.%pattern_type.loc28_40 (%pattern_type.8c644d.3) = out_param_pattern [template = %return.param_patt.loc28_40.2 (constants.%return.param_patt.eed287.2)]
// CHECK:STDOUT:     %return.patt.loc28_36.1: @Class.G.loc28.%pattern_type.loc28_40 (%pattern_type.8c644d.3) = return_slot_pattern %return.param_patt.loc28_40.1, %.loc28_40.9 [template = %return.patt.loc28_36.2 (constants.%return.patt.7881ad.2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc28_22.1: type = splice_block %.loc28_22.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc28_22.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc28_20: type = symbolic_binding T, 0, template [template = @Class.%T.loc5_23.1 (constants.%T)]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc28_20 [template = %T.loc28_30 (constants.%T)]
// CHECK:STDOUT:     %.loc28_40.7: type = type_of_inst %.loc28_40.1 [template = %.loc28_40.2 (constants.%.96c438.3)]
// CHECK:STDOUT:     %.loc28_40.8: @Class.G.loc28.%.loc28_40.2 (%.96c438.3) = splice_inst %.loc28_40.1 [template = %.loc28_40.3 (constants.%.8220de.3)]
// CHECK:STDOUT:     %.loc28_40.9: type = splice_inst %.loc28_40.4 [template = %.loc28_40.5 (constants.%.3cdd63.3)]
// CHECK:STDOUT:     %.loc28_40.10: Core.Form = init_form %.loc28_40.9 [template = %.loc28_40.6 (constants.%.a428e6.2)]
// CHECK:STDOUT:     %self.param: @Class.G.loc28.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:     %.loc28_30.1: type = splice_block %Self.ref [template = %Class (constants.%Class)] {
// CHECK:STDOUT:       %.loc28_30.2: type = specific_constant constants.%Class, @Class(constants.%T) [template = %Class (constants.%Class)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc28_30.2 [template = %Class (constants.%Class)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Class.G.loc28.%Class (%Class) = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref @Class.G.loc28.%.loc28_40.5 (%.3cdd63.3) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Class.G.loc28.%.loc28_40.5 (%.3cdd63.3) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc5_23.2: type) {
// CHECK:STDOUT:   %T.patt.loc5_23.2: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_23.2 (constants.%T.patt)]
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
// CHECK:STDOUT: generic fn @Class.F.loc18(%T.loc18_20: type) {
// CHECK:STDOUT:   %T.loc18_33: type = symbolic_binding T, 0, template [template = %T.loc18_33 (constants.%T)]
// CHECK:STDOUT:   %.loc18_34.1: <instruction> = access_member_action %T.ref.loc18_33, U [template]
// CHECK:STDOUT:   %.loc18_34.2: type = type_of_inst %.loc18_34.1 [template = %.loc18_34.2 (constants.%.96c438.1)]
// CHECK:STDOUT:   %.loc18_34.3: @Class.F.loc18.%.loc18_34.2 (%.96c438.1) = splice_inst %.loc18_34.1 [template = %.loc18_34.3 (constants.%.8220de.1)]
// CHECK:STDOUT:   %.loc18_34.4: <instruction> = convert_to_value_action %.loc18_34.8, type [template]
// CHECK:STDOUT:   %.loc18_34.5: type = splice_inst %.loc18_34.4 [template = %.loc18_34.5 (constants.%.3cdd63.1)]
// CHECK:STDOUT:   %pattern_type.loc18_31: type = pattern_type %.loc18_34.5 [template = %pattern_type.loc18_31 (constants.%pattern_type.8c644d.1)]
// CHECK:STDOUT:   %n.param_patt.loc18_31.2: @Class.F.loc18.%pattern_type.loc18_31 (%pattern_type.8c644d.1) = value_param_pattern [template = %n.param_patt.loc18_31.2 (constants.%n.param_patt.3a5)]
// CHECK:STDOUT:   %n.patt.loc18_31.2: @Class.F.loc18.%pattern_type.loc18_31 (%pattern_type.8c644d.1) = wrapper_binding_pattern n, %n.param_patt.loc18_31.2 [template = %n.patt.loc18_31.2 (constants.%n.patt.10a)]
// CHECK:STDOUT:   %.loc18_42.1: <instruction> = access_member_action %T.ref.loc18_41, U [template]
// CHECK:STDOUT:   %.loc18_42.2: type = type_of_inst %.loc18_42.1 [template = %.loc18_42.2 (constants.%.96c438.2)]
// CHECK:STDOUT:   %.loc18_42.3: @Class.F.loc18.%.loc18_42.2 (%.96c438.2) = splice_inst %.loc18_42.1 [template = %.loc18_42.3 (constants.%.8220de.2)]
// CHECK:STDOUT:   %.loc18_42.4: <instruction> = convert_to_value_action %.loc18_42.8, type [template]
// CHECK:STDOUT:   %.loc18_42.5: type = splice_inst %.loc18_42.4 [template = %.loc18_42.5 (constants.%.3cdd63.2)]
// CHECK:STDOUT:   %.loc18_42.6: Core.Form = init_form %.loc18_42.5 [template = %.loc18_42.6 (constants.%.a428e6.1)]
// CHECK:STDOUT:   %pattern_type.loc18_42: type = pattern_type %.loc18_42.5 [template = %pattern_type.loc18_42 (constants.%pattern_type.8c644d.2)]
// CHECK:STDOUT:   %return.param_patt.loc18_42.2: @Class.F.loc18.%pattern_type.loc18_42 (%pattern_type.8c644d.2) = out_param_pattern [template = %return.param_patt.loc18_42.2 (constants.%return.param_patt.eed287.1)]
// CHECK:STDOUT:   %return.patt.loc18_38.2: @Class.F.loc18.%pattern_type.loc18_42 (%pattern_type.8c644d.2) = return_slot_pattern %return.param_patt.loc18_42.2, %.loc18_42.5 [template = %return.patt.loc18_38.2 (constants.%return.patt.7881ad.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc18_31: <witness> = require_complete_type %.loc18_34.5 [template = %require_complete.loc18_31 (constants.%require_complete.b6b0e1.1)]
// CHECK:STDOUT:   %require_complete.loc18_38: <witness> = require_complete_type %.loc18_42.5 [template = %require_complete.loc18_38 (constants.%require_complete.b6b0e1.2)]
// CHECK:STDOUT:   %.loc19_11.2: %tuple.type = initialize_action %n.ref, %.loc18_42.5, %return.param, false [template]
// CHECK:STDOUT:   %tuple.elem0: <instruction> = tuple_access %.loc19_11.2, element0 [template]
// CHECK:STDOUT:   %.loc19_11.3: @Class.F.loc18.%.loc18_42.5 (%.3cdd63.2) = splice_inst %tuple.elem0 [template = %.loc19_11.3 (constants.%.c17)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param: @Class.F.loc18.%.loc18_34.5 (%.3cdd63.1)) -> out %return.param: @Class.F.loc18.%.loc18_42.5 (%.3cdd63.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %n.ref: @Class.F.loc18.%.loc18_34.5 (%.3cdd63.1) = name_ref n, %n
// CHECK:STDOUT:     %.loc19_11.1: @Class.F.loc18.%.loc18_42.5 (%.3cdd63.2) = splice_inst %tuple.elem0 [template = %.loc19_11.3 (constants.%.c17)]
// CHECK:STDOUT:     return %.loc19_11.1 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.G.loc28(%T.loc28_20: type) {
// CHECK:STDOUT:   %T.loc28_30: type = symbolic_binding T, 0, template [template = %T.loc28_30 (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc28_30) [template = %Class (constants.%Class)]
// CHECK:STDOUT:   %pattern_type.loc28_30: type = pattern_type %Class [template = %pattern_type.loc28_30 (constants.%pattern_type.36e)]
// CHECK:STDOUT:   %self.param_patt.loc28_30.2: @Class.G.loc28.%pattern_type.loc28_30 (%pattern_type.36e) = value_param_pattern [template = %self.param_patt.loc28_30.2 (constants.%self.param_patt)]
// CHECK:STDOUT:   %self.patt.loc28_30.2: @Class.G.loc28.%pattern_type.loc28_30 (%pattern_type.36e) = wrapper_binding_pattern self, %self.param_patt.loc28_30.2 [template = %self.patt.loc28_30.2 (constants.%self.patt)]
// CHECK:STDOUT:   %.loc28_40.1: <instruction> = access_member_action %T.ref, U [template]
// CHECK:STDOUT:   %.loc28_40.2: type = type_of_inst %.loc28_40.1 [template = %.loc28_40.2 (constants.%.96c438.3)]
// CHECK:STDOUT:   %.loc28_40.3: @Class.G.loc28.%.loc28_40.2 (%.96c438.3) = splice_inst %.loc28_40.1 [template = %.loc28_40.3 (constants.%.8220de.3)]
// CHECK:STDOUT:   %.loc28_40.4: <instruction> = convert_to_value_action %.loc28_40.8, type [template]
// CHECK:STDOUT:   %.loc28_40.5: type = splice_inst %.loc28_40.4 [template = %.loc28_40.5 (constants.%.3cdd63.3)]
// CHECK:STDOUT:   %.loc28_40.6: Core.Form = init_form %.loc28_40.5 [template = %.loc28_40.6 (constants.%.a428e6.2)]
// CHECK:STDOUT:   %pattern_type.loc28_40: type = pattern_type %.loc28_40.5 [template = %pattern_type.loc28_40 (constants.%pattern_type.8c644d.3)]
// CHECK:STDOUT:   %return.param_patt.loc28_40.2: @Class.G.loc28.%pattern_type.loc28_40 (%pattern_type.8c644d.3) = out_param_pattern [template = %return.param_patt.loc28_40.2 (constants.%return.param_patt.eed287.2)]
// CHECK:STDOUT:   %return.patt.loc28_36.2: @Class.G.loc28.%pattern_type.loc28_40 (%pattern_type.8c644d.3) = return_slot_pattern %return.param_patt.loc28_40.2, %.loc28_40.5 [template = %return.patt.loc28_36.2 (constants.%return.patt.7881ad.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc28_30: <witness> = require_complete_type %Class [template = %require_complete.loc28_30 (constants.%require_complete.4d9)]
// CHECK:STDOUT:   %require_complete.loc28_36: <witness> = require_complete_type %.loc28_40.5 [template = %require_complete.loc28_36 (constants.%require_complete.b6b0e1.3)]
// CHECK:STDOUT:   %.loc29_14.3: <instruction> = access_member_action %self.ref, n [template]
// CHECK:STDOUT:   %.loc29_14.4: type = type_of_inst %.loc29_14.3 [template = %.loc29_14.4 (constants.%.dd9)]
// CHECK:STDOUT:   %.loc29_14.5: @Class.G.loc28.%.loc29_14.4 (%.dd9) = splice_inst %.loc29_14.3 [template = %.loc29_14.5 (constants.%.732)]
// CHECK:STDOUT:   %.loc29_16.2: %tuple.type = initialize_action %.loc29_14.2, %.loc28_40.5, %return.param, false [template]
// CHECK:STDOUT:   %tuple.elem0: <instruction> = tuple_access %.loc29_16.2, element0 [template]
// CHECK:STDOUT:   %.loc29_16.3: @Class.G.loc28.%.loc28_40.5 (%.3cdd63.3) = splice_inst %tuple.elem0 [template = %.loc29_16.3 (constants.%.b9e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.G.loc28.%Class (%Class)) -> out %return.param: @Class.G.loc28.%.loc28_40.5 (%.3cdd63.3) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.G.loc28.%Class (%Class) = name_ref self, %self
// CHECK:STDOUT:     %.loc29_14.1: type = type_of_inst %.loc29_14.3 [template = %.loc29_14.4 (constants.%.dd9)]
// CHECK:STDOUT:     %.loc29_14.2: @Class.G.loc28.%.loc29_14.4 (%.dd9) = splice_inst %.loc29_14.3 [template = %.loc29_14.5 (constants.%.732)]
// CHECK:STDOUT:     %.loc29_16.1: @Class.G.loc28.%.loc28_40.5 (%.3cdd63.3) = splice_inst %tuple.elem0 [template = %.loc29_16.3 (constants.%.b9e)]
// CHECK:STDOUT:     return %.loc29_16.1 to %return.param
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
// CHECK:STDOUT: specific @Class.F.loc18(constants.%T) {
// CHECK:STDOUT:   %T.loc18_33 => constants.%T
// CHECK:STDOUT:   %.loc18_34.1 => constants.%.3d9
// CHECK:STDOUT:   %.loc18_34.2 => constants.%.cf6
// CHECK:STDOUT:   %.loc18_34.3 => constants.%.2db
// CHECK:STDOUT:   %.loc18_34.4 => constants.%.dd0
// CHECK:STDOUT:   %.loc18_34.5 => constants.%.3e1
// CHECK:STDOUT:   %pattern_type.loc18_31 => constants.%pattern_type.e0b
// CHECK:STDOUT:   %n.param_patt.loc18_31.2 => constants.%n.param_patt.396
// CHECK:STDOUT:   %n.patt.loc18_31.2 => constants.%n.patt.f9c
// CHECK:STDOUT:   %.loc18_42.1 => constants.%.3d9
// CHECK:STDOUT:   %.loc18_42.2 => constants.%.cf6
// CHECK:STDOUT:   %.loc18_42.3 => constants.%.2db
// CHECK:STDOUT:   %.loc18_42.4 => constants.%.dd0
// CHECK:STDOUT:   %.loc18_42.5 => constants.%.3e1
// CHECK:STDOUT:   %.loc18_42.6 => constants.%.764
// CHECK:STDOUT:   %pattern_type.loc18_42 => constants.%pattern_type.e0b
// CHECK:STDOUT:   %return.param_patt.loc18_42.2 => constants.%return.param_patt.fc1
// CHECK:STDOUT:   %return.patt.loc18_38.2 => constants.%return.patt.fcc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.G.loc28(constants.%T) {
// CHECK:STDOUT:   %T.loc28_30 => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %pattern_type.loc28_30 => constants.%pattern_type.36e
// CHECK:STDOUT:   %self.param_patt.loc28_30.2 => constants.%self.param_patt
// CHECK:STDOUT:   %self.patt.loc28_30.2 => constants.%self.patt
// CHECK:STDOUT:   %.loc28_40.1 => constants.%.3d9
// CHECK:STDOUT:   %.loc28_40.2 => constants.%.cf6
// CHECK:STDOUT:   %.loc28_40.3 => constants.%.2db
// CHECK:STDOUT:   %.loc28_40.4 => constants.%.dd0
// CHECK:STDOUT:   %.loc28_40.5 => constants.%.3e1
// CHECK:STDOUT:   %.loc28_40.6 => constants.%.764
// CHECK:STDOUT:   %pattern_type.loc28_40 => constants.%pattern_type.e0b
// CHECK:STDOUT:   %return.param_patt.loc28_40.2 => constants.%return.param_patt.fc1
// CHECK:STDOUT:   %return.patt.loc28_36.2 => constants.%return.patt.fcc
// CHECK:STDOUT: }
// CHECK:STDOUT:
