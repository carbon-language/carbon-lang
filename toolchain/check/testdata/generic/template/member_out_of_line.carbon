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

// CHECK:STDERR: fail_todo_basic.carbon:[[@LINE+3]]:30-35: error: type `<pattern for <splice of <cannot stringify inst7800007C: {kind: ConvertToValueAction, arg0: inst78000079, arg1: inst(TypeType), type: type(inst(InstType))}>>>` of parameter 1 in redeclaration differs from previous parameter type `<pattern for T>` [RedeclParamDiffersType]
// CHECK:STDERR: fail_todo_basic.carbon:[[@LINE-6]]:8-11: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR:
fn Class(template T: type).F(n: T.U) -> T.U {
  return n;
}

// CHECK:STDERR: fail_todo_basic.carbon:[[@LINE+3]]:1-43: error: function redeclaration differs because return type is `<splice of <cannot stringify inst78000138: {kind: ConvertToValueAction, arg0: inst78000135, arg1: inst(TypeType), type: type(inst(InstType))}>>` [FunctionRedeclReturnTypeDiffers]
// CHECK:STDERR: fail_todo_basic.carbon:[[@LINE-12]]:3-18: note: previously declared with return type `T` [FunctionRedeclReturnTypePrevious]
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
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T [template]
// CHECK:STDOUT:   %n.param_patt.fb2: %pattern_type.51d1c4.1 = value_param_pattern [template]
// CHECK:STDOUT:   %n.patt.955: %pattern_type.51d1c4.1 = wrapper_binding_pattern n, %n.param_patt.fb2 [template]
// CHECK:STDOUT:   %.184347.1: Core.Form = init_form %T [template]
// CHECK:STDOUT:   %return.param_patt.41d2d9.1: %pattern_type.51d1c4.1 = out_param_pattern [template]
// CHECK:STDOUT:   %return.patt.b39: %pattern_type.51d1c4.1 = return_slot_pattern %return.param_patt.41d2d9.1, %T [template]
// CHECK:STDOUT:   %Class.F.type.9829b4.1: type = fn_type @Class.F.loc6, @Class(%T) [template]
// CHECK:STDOUT:   %Class.F.c6358f.1: %Class.F.type.9829b4.1 = struct_value () [template]
// CHECK:STDOUT:   %pattern_type.36e: type = pattern_type %Class [template]
// CHECK:STDOUT:   %self.param_patt.f93: %pattern_type.36e = value_param_pattern [template]
// CHECK:STDOUT:   %self.patt.1dd: %pattern_type.36e = wrapper_binding_pattern self, %self.param_patt.f93 [template]
// CHECK:STDOUT:   %Class.G.type.ea69e8.1: type = fn_type @Class.G.loc7, @Class(%T) [template]
// CHECK:STDOUT:   %Class.G.5aff6a.1: %Class.G.type.ea69e8.1 = struct_value () [template]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [template]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T [template]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %T} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n [template]
// CHECK:STDOUT:   %.96c438.1: type = type_of_inst @Class.F.loc14.%.loc14_34.1 [template]
// CHECK:STDOUT:   %.8220de.1: %.96c438.1 = splice_inst @Class.F.loc14.%.loc14_34.1 [template]
// CHECK:STDOUT:   %.3cdd63.1: type = splice_inst @Class.F.loc14.%.loc14_34.4 [template]
// CHECK:STDOUT:   %pattern_type.8c644d.1: type = pattern_type %.3cdd63.1 [template]
// CHECK:STDOUT:   %n.param_patt.3a5: %pattern_type.8c644d.1 = value_param_pattern [template]
// CHECK:STDOUT:   %n.patt.10a: %pattern_type.8c644d.1 = wrapper_binding_pattern n, %n.param_patt.3a5 [template]
// CHECK:STDOUT:   %.96c438.2: type = type_of_inst @Class.F.loc14.%.loc14_42.1 [template]
// CHECK:STDOUT:   %.8220de.2: %.96c438.2 = splice_inst @Class.F.loc14.%.loc14_42.1 [template]
// CHECK:STDOUT:   %.3cdd63.2: type = splice_inst @Class.F.loc14.%.loc14_42.4 [template]
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
// CHECK:STDOUT:   %Class.F.type.9829b4.2: type = fn_type @Class.F.loc14, @Class(%T) [template]
// CHECK:STDOUT:   %Class.F.c6358f.2: %Class.F.type.9829b4.2 = struct_value () [template]
// CHECK:STDOUT:   %require_complete.b6b0e1.1: <witness> = require_complete_type %.3cdd63.1 [template]
// CHECK:STDOUT:   %require_complete.b6b0e1.2: <witness> = require_complete_type %.3cdd63.2 [template]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %.3a748a.1: type = type_of_inst @Class.F.loc14.%.loc15_11.11 [template]
// CHECK:STDOUT:   %.c3d21a.1: %.3a748a.1 = splice_inst @Class.F.loc14.%.loc15_11.11 [template]
// CHECK:STDOUT:   %.e294f6.1: type = type_of_inst @Class.F.loc14.%.loc15_11.14 [template]
// CHECK:STDOUT:   %.491b27.1: %.e294f6.1 = splice_inst @Class.F.loc14.%.loc15_11.14 [template]
// CHECK:STDOUT:   %.516: type = type_of_inst @Class.F.loc14.%.loc15_11.17 [template]
// CHECK:STDOUT:   %.3f2: %.516 = splice_inst @Class.F.loc14.%.loc15_11.17 [template]
// CHECK:STDOUT:   %.8db: type = type_of_inst @Class.F.loc14.%.loc15_11.20 [template]
// CHECK:STDOUT:   %.d67: %.8db = splice_inst @Class.F.loc14.%.loc15_11.20 [template]
// CHECK:STDOUT:   %.a56: %.3cdd63.2 = splice_inst @Class.F.loc14.%.loc15_11.23 [template]
// CHECK:STDOUT:   %.96c438.3: type = type_of_inst @Class.G.loc21.%.loc21_40.1 [template]
// CHECK:STDOUT:   %.8220de.3: %.96c438.3 = splice_inst @Class.G.loc21.%.loc21_40.1 [template]
// CHECK:STDOUT:   %.3cdd63.3: type = splice_inst @Class.G.loc21.%.loc21_40.4 [template]
// CHECK:STDOUT:   %.a428e6.2: Core.Form = init_form %.3cdd63.3 [template]
// CHECK:STDOUT:   %pattern_type.8c644d.3: type = pattern_type %.3cdd63.3 [template]
// CHECK:STDOUT:   %return.param_patt.eed287.2: %pattern_type.8c644d.3 = out_param_pattern [template]
// CHECK:STDOUT:   %return.patt.7881ad.2: %pattern_type.8c644d.3 = return_slot_pattern %return.param_patt.eed287.2, %.3cdd63.3 [template]
// CHECK:STDOUT:   %Class.G.type.ea69e8.2: type = fn_type @Class.G.loc21, @Class(%T) [template]
// CHECK:STDOUT:   %Class.G.5aff6a.2: %Class.G.type.ea69e8.2 = struct_value () [template]
// CHECK:STDOUT:   %require_complete.4d9: <witness> = require_complete_type %Class [template]
// CHECK:STDOUT:   %require_complete.b6b0e1.3: <witness> = require_complete_type %.3cdd63.3 [template]
// CHECK:STDOUT:   %.dd9: type = type_of_inst @Class.G.loc21.%.loc22_14.3 [template]
// CHECK:STDOUT:   %.732: %.dd9 = splice_inst @Class.G.loc21.%.loc22_14.3 [template]
// CHECK:STDOUT:   %.3a748a.2: type = type_of_inst @Class.G.loc21.%.loc22_16.11 [template]
// CHECK:STDOUT:   %.c3d21a.2: %.3a748a.2 = splice_inst @Class.G.loc21.%.loc22_16.11 [template]
// CHECK:STDOUT:   %.e294f6.2: type = type_of_inst @Class.G.loc21.%.loc22_16.14 [template]
// CHECK:STDOUT:   %.491b27.2: %.e294f6.2 = splice_inst @Class.G.loc21.%.loc22_16.14 [template]
// CHECK:STDOUT:   %.b0f: type = type_of_inst @Class.G.loc21.%.loc22_16.17 [template]
// CHECK:STDOUT:   %.adc: %.b0f = splice_inst @Class.G.loc21.%.loc22_16.17 [template]
// CHECK:STDOUT:   %.09a: type = type_of_inst @Class.G.loc21.%.loc22_16.20 [template]
// CHECK:STDOUT:   %.cb0: %.09a = splice_inst @Class.G.loc21.%.loc22_16.20 [template]
// CHECK:STDOUT:   %.957: %.3cdd63.3 = splice_inst @Class.G.loc21.%.loc22_16.23 [template]
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
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type.9829b4.2 = fn_decl @Class.F.loc14 [template = constants.%Class.F.c6358f.2] {
// CHECK:STDOUT:     %n.param_patt.loc14_31.1: @Class.F.loc14.%pattern_type.loc14_31 (%pattern_type.8c644d.1) = value_param_pattern [template = %n.param_patt.loc14_31.2 (constants.%n.param_patt.3a5)]
// CHECK:STDOUT:     %n.patt.loc14_31.1: @Class.F.loc14.%pattern_type.loc14_31 (%pattern_type.8c644d.1) = wrapper_binding_pattern n, %n.param_patt.loc14_31.1 [template = %n.patt.loc14_31.2 (constants.%n.patt.10a)]
// CHECK:STDOUT:     %return.param_patt.loc14_42.1: @Class.F.loc14.%pattern_type.loc14_42 (%pattern_type.8c644d.2) = out_param_pattern [template = %return.param_patt.loc14_42.2 (constants.%return.param_patt.eed287.1)]
// CHECK:STDOUT:     %return.patt.loc14_38.1: @Class.F.loc14.%pattern_type.loc14_42 (%pattern_type.8c644d.2) = return_slot_pattern %return.param_patt.loc14_42.1, %.loc14_42.9 [template = %return.patt.loc14_38.2 (constants.%return.patt.7881ad.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_22.1: type = splice_block %.loc14_22.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc14_22.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc14_20: type = symbolic_binding T, 0, template [template = @Class.%T.loc5_23.1 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc14_41: type = name_ref T, %T.loc14_20 [template = %T.loc14_33 (constants.%T)]
// CHECK:STDOUT:     %.loc14_42.7: type = type_of_inst %.loc14_42.1 [template = %.loc14_42.2 (constants.%.96c438.2)]
// CHECK:STDOUT:     %.loc14_42.8: @Class.F.loc14.%.loc14_42.2 (%.96c438.2) = splice_inst %.loc14_42.1 [template = %.loc14_42.3 (constants.%.8220de.2)]
// CHECK:STDOUT:     %.loc14_42.9: type = splice_inst %.loc14_42.4 [template = %.loc14_42.5 (constants.%.3cdd63.2)]
// CHECK:STDOUT:     %.loc14_42.10: Core.Form = init_form %.loc14_42.9 [template = %.loc14_42.6 (constants.%.a428e6.1)]
// CHECK:STDOUT:     %n.param: @Class.F.loc14.%.loc14_34.5 (%.3cdd63.1) = value_param call_param0
// CHECK:STDOUT:     %.loc14_34.6: type = splice_block %.loc14_34.9 [template = %.loc14_34.5 (constants.%.3cdd63.1)] {
// CHECK:STDOUT:       %T.ref.loc14_33: type = name_ref T, %T.loc14_20 [template = %T.loc14_33 (constants.%T)]
// CHECK:STDOUT:       %.loc14_34.7: type = type_of_inst %.loc14_34.1 [template = %.loc14_34.2 (constants.%.96c438.1)]
// CHECK:STDOUT:       %.loc14_34.8: @Class.F.loc14.%.loc14_34.2 (%.96c438.1) = splice_inst %.loc14_34.1 [template = %.loc14_34.3 (constants.%.8220de.1)]
// CHECK:STDOUT:       %.loc14_34.9: type = splice_inst %.loc14_34.4 [template = %.loc14_34.5 (constants.%.3cdd63.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: @Class.F.loc14.%.loc14_34.5 (%.3cdd63.1) = wrapper_binding n, %n.param
// CHECK:STDOUT:     %return.param: ref @Class.F.loc14.%.loc14_42.5 (%.3cdd63.2) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Class.F.loc14.%.loc14_42.5 (%.3cdd63.2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.G.decl: %Class.G.type.ea69e8.2 = fn_decl @Class.G.loc21 [template = constants.%Class.G.5aff6a.2] {
// CHECK:STDOUT:     %self.param_patt.loc21_30.1: @Class.G.loc21.%pattern_type.loc21_30 (%pattern_type.36e) = value_param_pattern [template = %self.param_patt.loc21_30.2 (constants.%self.param_patt.f93)]
// CHECK:STDOUT:     %self.patt.loc21_30.1: @Class.G.loc21.%pattern_type.loc21_30 (%pattern_type.36e) = wrapper_binding_pattern self, %self.param_patt.loc21_30.1 [template = %self.patt.loc21_30.2 (constants.%self.patt.1dd)]
// CHECK:STDOUT:     %return.param_patt.loc21_40.1: @Class.G.loc21.%pattern_type.loc21_40 (%pattern_type.8c644d.3) = out_param_pattern [template = %return.param_patt.loc21_40.2 (constants.%return.param_patt.eed287.2)]
// CHECK:STDOUT:     %return.patt.loc21_36.1: @Class.G.loc21.%pattern_type.loc21_40 (%pattern_type.8c644d.3) = return_slot_pattern %return.param_patt.loc21_40.1, %.loc21_40.9 [template = %return.patt.loc21_36.2 (constants.%return.patt.7881ad.2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc21_22.1: type = splice_block %.loc21_22.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc21_22.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc21_20: type = symbolic_binding T, 0, template [template = @Class.%T.loc5_23.1 (constants.%T)]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc21_20 [template = %T.loc21_30 (constants.%T)]
// CHECK:STDOUT:     %.loc21_40.7: type = type_of_inst %.loc21_40.1 [template = %.loc21_40.2 (constants.%.96c438.3)]
// CHECK:STDOUT:     %.loc21_40.8: @Class.G.loc21.%.loc21_40.2 (%.96c438.3) = splice_inst %.loc21_40.1 [template = %.loc21_40.3 (constants.%.8220de.3)]
// CHECK:STDOUT:     %.loc21_40.9: type = splice_inst %.loc21_40.4 [template = %.loc21_40.5 (constants.%.3cdd63.3)]
// CHECK:STDOUT:     %.loc21_40.10: Core.Form = init_form %.loc21_40.9 [template = %.loc21_40.6 (constants.%.a428e6.2)]
// CHECK:STDOUT:     %self.param: @Class.G.loc21.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:     %.loc21_30.1: type = splice_block %Self.ref [template = %Class (constants.%Class)] {
// CHECK:STDOUT:       %.loc21_30.2: type = specific_constant constants.%Class, @Class(constants.%T) [template = %Class (constants.%Class)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc21_30.2 [template = %Class (constants.%Class)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Class.G.loc21.%Class (%Class) = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref @Class.G.loc21.%.loc21_40.5 (%.3cdd63.3) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Class.G.loc21.%.loc21_40.5 (%.3cdd63.3) = return_slot %return.param
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
// CHECK:STDOUT:       %n.param_patt.loc6_9.1: @Class.F.loc6.%pattern_type (%pattern_type.51d1c4.1) = value_param_pattern [template = %n.param_patt.loc6_9.2 (constants.%n.param_patt.fb2)]
// CHECK:STDOUT:       %n.patt.loc6_9.1: @Class.F.loc6.%pattern_type (%pattern_type.51d1c4.1) = wrapper_binding_pattern n, %n.param_patt.loc6_9.1 [template = %n.patt.loc6_9.2 (constants.%n.patt.955)]
// CHECK:STDOUT:       %return.param_patt.loc6_17.1: @Class.F.loc6.%pattern_type (%pattern_type.51d1c4.1) = out_param_pattern [template = %return.param_patt.loc6_17.2 (constants.%return.param_patt.41d2d9.1)]
// CHECK:STDOUT:       %return.patt.loc6_14.1: @Class.F.loc6.%pattern_type (%pattern_type.51d1c4.1) = return_slot_pattern %return.param_patt.loc6_17.1, %T.ref.loc6_17 [template = %return.patt.loc6_14.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref.loc6_17: type = name_ref T, @Class.%T.loc5_23.2 [template = %T (constants.%T)]
// CHECK:STDOUT:       %.loc6_17.2: Core.Form = init_form %T.ref.loc6_17 [template = %.loc6_17.1 (constants.%.184347.1)]
// CHECK:STDOUT:       %n.param: @Class.F.loc6.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %T.ref.loc6_11: type = name_ref T, @Class.%T.loc5_23.2 [template = %T (constants.%T)]
// CHECK:STDOUT:       %n: @Class.F.loc6.%T (%T) = wrapper_binding n, %n.param
// CHECK:STDOUT:       %return.param: ref @Class.F.loc6.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Class.F.loc6.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Class.G.decl: @Class.%Class.G.type (%Class.G.type.ea69e8.1) = fn_decl @Class.G.loc7 [template = @Class.%Class.G (constants.%Class.G.5aff6a.1)] {
// CHECK:STDOUT:       %self.param_patt.loc7_8.1: @Class.G.loc7.%pattern_type.loc7_8 (%pattern_type.36e) = value_param_pattern [template = %self.param_patt.loc7_8.2 (constants.%self.param_patt.f93)]
// CHECK:STDOUT:       %self.patt.loc7_8.1: @Class.G.loc7.%pattern_type.loc7_8 (%pattern_type.36e) = wrapper_binding_pattern self, %self.param_patt.loc7_8.1 [template = %self.patt.loc7_8.2 (constants.%self.patt.1dd)]
// CHECK:STDOUT:       %return.param_patt.loc7_17.1: @Class.G.loc7.%pattern_type.loc7_17 (%pattern_type.51d1c4.1) = out_param_pattern [template = %return.param_patt.loc7_17.2 (constants.%return.param_patt.41d2d9.1)]
// CHECK:STDOUT:       %return.patt.loc7_14.1: @Class.G.loc7.%pattern_type.loc7_17 (%pattern_type.51d1c4.1) = return_slot_pattern %return.param_patt.loc7_17.1, %T.ref [template = %return.patt.loc7_14.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc5_23.2 [template = %T (constants.%T)]
// CHECK:STDOUT:       %.loc7_17.2: Core.Form = init_form %T.ref [template = %.loc7_17.1 (constants.%.184347.1)]
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
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [template = %pattern_type (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:   %n.param_patt.loc6_9.2: @Class.F.loc6.%pattern_type (%pattern_type.51d1c4.1) = value_param_pattern [template = %n.param_patt.loc6_9.2 (constants.%n.param_patt.fb2)]
// CHECK:STDOUT:   %n.patt.loc6_9.2: @Class.F.loc6.%pattern_type (%pattern_type.51d1c4.1) = wrapper_binding_pattern n, %n.param_patt.loc6_9.2 [template = %n.patt.loc6_9.2 (constants.%n.patt.955)]
// CHECK:STDOUT:   %.loc6_17.1: Core.Form = init_form %T [template = %.loc6_17.1 (constants.%.184347.1)]
// CHECK:STDOUT:   %return.param_patt.loc6_17.2: @Class.F.loc6.%pattern_type (%pattern_type.51d1c4.1) = out_param_pattern [template = %return.param_patt.loc6_17.2 (constants.%return.param_patt.41d2d9.1)]
// CHECK:STDOUT:   %return.patt.loc6_14.2: @Class.F.loc6.%pattern_type (%pattern_type.51d1c4.1) = return_slot_pattern %return.param_patt.loc6_17.2, %T [template = %return.patt.loc6_14.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param: @Class.F.loc6.%T (%T)) -> out %return.param: @Class.F.loc6.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.G.loc7(@Class.%T.loc5_23.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0, template [template = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [template = %Class (constants.%Class)]
// CHECK:STDOUT:   %pattern_type.loc7_8: type = pattern_type %Class [template = %pattern_type.loc7_8 (constants.%pattern_type.36e)]
// CHECK:STDOUT:   %self.param_patt.loc7_8.2: @Class.G.loc7.%pattern_type.loc7_8 (%pattern_type.36e) = value_param_pattern [template = %self.param_patt.loc7_8.2 (constants.%self.param_patt.f93)]
// CHECK:STDOUT:   %self.patt.loc7_8.2: @Class.G.loc7.%pattern_type.loc7_8 (%pattern_type.36e) = wrapper_binding_pattern self, %self.param_patt.loc7_8.2 [template = %self.patt.loc7_8.2 (constants.%self.patt.1dd)]
// CHECK:STDOUT:   %.loc7_17.1: Core.Form = init_form %T [template = %.loc7_17.1 (constants.%.184347.1)]
// CHECK:STDOUT:   %pattern_type.loc7_17: type = pattern_type %T [template = %pattern_type.loc7_17 (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:   %return.param_patt.loc7_17.2: @Class.G.loc7.%pattern_type.loc7_17 (%pattern_type.51d1c4.1) = out_param_pattern [template = %return.param_patt.loc7_17.2 (constants.%return.param_patt.41d2d9.1)]
// CHECK:STDOUT:   %return.patt.loc7_14.2: @Class.G.loc7.%pattern_type.loc7_17 (%pattern_type.51d1c4.1) = return_slot_pattern %return.param_patt.loc7_17.2, %T [template = %return.patt.loc7_14.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.G.loc7.%Class (%Class)) -> out %return.param: @Class.G.loc7.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.F.loc14(%T.loc14_20: type) {
// CHECK:STDOUT:   %T.loc14_33: type = symbolic_binding T, 0, template [template = %T.loc14_33 (constants.%T)]
// CHECK:STDOUT:   %.loc14_34.1: <instruction> = access_member_action %T.ref.loc14_33, U [template]
// CHECK:STDOUT:   %.loc14_34.2: type = type_of_inst %.loc14_34.1 [template = %.loc14_34.2 (constants.%.96c438.1)]
// CHECK:STDOUT:   %.loc14_34.3: @Class.F.loc14.%.loc14_34.2 (%.96c438.1) = splice_inst %.loc14_34.1 [template = %.loc14_34.3 (constants.%.8220de.1)]
// CHECK:STDOUT:   %.loc14_34.4: <instruction> = convert_to_value_action %.loc14_34.8, type [template]
// CHECK:STDOUT:   %.loc14_34.5: type = splice_inst %.loc14_34.4 [template = %.loc14_34.5 (constants.%.3cdd63.1)]
// CHECK:STDOUT:   %pattern_type.loc14_31: type = pattern_type %.loc14_34.5 [template = %pattern_type.loc14_31 (constants.%pattern_type.8c644d.1)]
// CHECK:STDOUT:   %n.param_patt.loc14_31.2: @Class.F.loc14.%pattern_type.loc14_31 (%pattern_type.8c644d.1) = value_param_pattern [template = %n.param_patt.loc14_31.2 (constants.%n.param_patt.3a5)]
// CHECK:STDOUT:   %n.patt.loc14_31.2: @Class.F.loc14.%pattern_type.loc14_31 (%pattern_type.8c644d.1) = wrapper_binding_pattern n, %n.param_patt.loc14_31.2 [template = %n.patt.loc14_31.2 (constants.%n.patt.10a)]
// CHECK:STDOUT:   %.loc14_42.1: <instruction> = access_member_action %T.ref.loc14_41, U [template]
// CHECK:STDOUT:   %.loc14_42.2: type = type_of_inst %.loc14_42.1 [template = %.loc14_42.2 (constants.%.96c438.2)]
// CHECK:STDOUT:   %.loc14_42.3: @Class.F.loc14.%.loc14_42.2 (%.96c438.2) = splice_inst %.loc14_42.1 [template = %.loc14_42.3 (constants.%.8220de.2)]
// CHECK:STDOUT:   %.loc14_42.4: <instruction> = convert_to_value_action %.loc14_42.8, type [template]
// CHECK:STDOUT:   %.loc14_42.5: type = splice_inst %.loc14_42.4 [template = %.loc14_42.5 (constants.%.3cdd63.2)]
// CHECK:STDOUT:   %.loc14_42.6: Core.Form = init_form %.loc14_42.5 [template = %.loc14_42.6 (constants.%.a428e6.1)]
// CHECK:STDOUT:   %pattern_type.loc14_42: type = pattern_type %.loc14_42.5 [template = %pattern_type.loc14_42 (constants.%pattern_type.8c644d.2)]
// CHECK:STDOUT:   %return.param_patt.loc14_42.2: @Class.F.loc14.%pattern_type.loc14_42 (%pattern_type.8c644d.2) = out_param_pattern [template = %return.param_patt.loc14_42.2 (constants.%return.param_patt.eed287.1)]
// CHECK:STDOUT:   %return.patt.loc14_38.2: @Class.F.loc14.%pattern_type.loc14_42 (%pattern_type.8c644d.2) = return_slot_pattern %return.param_patt.loc14_42.2, %.loc14_42.5 [template = %return.patt.loc14_38.2 (constants.%return.patt.7881ad.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc14_31: <witness> = require_complete_type %.loc14_34.5 [template = %require_complete.loc14_31 (constants.%require_complete.b6b0e1.1)]
// CHECK:STDOUT:   %require_complete.loc14_38: <witness> = require_complete_type %.loc14_42.5 [template = %require_complete.loc14_38 (constants.%require_complete.b6b0e1.2)]
// CHECK:STDOUT:   %.loc15_11.11: <instruction> = call_action (constants.%ImplicitAs.generic, constants.%.3cdd63.2), false [template]
// CHECK:STDOUT:   %.loc15_11.12: type = type_of_inst %.loc15_11.11 [template = %.loc15_11.12 (constants.%.3a748a.1)]
// CHECK:STDOUT:   %.loc15_11.13: @Class.F.loc14.%.loc15_11.12 (%.3a748a.1) = splice_inst %.loc15_11.11 [template = %.loc15_11.13 (constants.%.c3d21a.1)]
// CHECK:STDOUT:   %.loc15_11.14: <instruction> = access_member_action %.loc15_11.2, Convert [template]
// CHECK:STDOUT:   %.loc15_11.15: type = type_of_inst %.loc15_11.14 [template = %.loc15_11.15 (constants.%.e294f6.1)]
// CHECK:STDOUT:   %.loc15_11.16: @Class.F.loc14.%.loc15_11.15 (%.e294f6.1) = splice_inst %.loc15_11.14 [template = %.loc15_11.16 (constants.%.491b27.1)]
// CHECK:STDOUT:   %.loc15_11.17: <instruction> = compound_member_access_action %n.ref, %.loc15_11.4 [template]
// CHECK:STDOUT:   %.loc15_11.18: type = type_of_inst %.loc15_11.17 [template = %.loc15_11.18 (constants.%.516)]
// CHECK:STDOUT:   %.loc15_11.19: @Class.F.loc14.%.loc15_11.18 (%.516) = splice_inst %.loc15_11.17 [template = %.loc15_11.19 (constants.%.3f2)]
// CHECK:STDOUT:   %.loc15_11.20: <instruction> = call_action (%.loc15_11.6), true [template]
// CHECK:STDOUT:   %.loc15_11.21: type = type_of_inst %.loc15_11.20 [template = %.loc15_11.21 (constants.%.8db)]
// CHECK:STDOUT:   %.loc15_11.22: @Class.F.loc14.%.loc15_11.21 (%.8db) = splice_inst %.loc15_11.20 [template = %.loc15_11.22 (constants.%.d67)]
// CHECK:STDOUT:   %.loc15_11.23: <instruction> = convert_to_category_action %.loc15_11.9, element10 [template]
// CHECK:STDOUT:   %.loc15_11.24: @Class.F.loc14.%.loc14_42.5 (%.3cdd63.2) = splice_inst %.loc15_11.23 [template = %.loc15_11.24 (constants.%.a56)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param: @Class.F.loc14.%.loc14_34.5 (%.3cdd63.1)) -> out %return.param: @Class.F.loc14.%.loc14_42.5 (%.3cdd63.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %n.ref: @Class.F.loc14.%.loc14_34.5 (%.3cdd63.1) = name_ref n, %n
// CHECK:STDOUT:     %.loc15_11.1: type = type_of_inst %.loc15_11.11 [template = %.loc15_11.12 (constants.%.3a748a.1)]
// CHECK:STDOUT:     %.loc15_11.2: @Class.F.loc14.%.loc15_11.12 (%.3a748a.1) = splice_inst %.loc15_11.11 [template = %.loc15_11.13 (constants.%.c3d21a.1)]
// CHECK:STDOUT:     %.loc15_11.3: type = type_of_inst %.loc15_11.14 [template = %.loc15_11.15 (constants.%.e294f6.1)]
// CHECK:STDOUT:     %.loc15_11.4: @Class.F.loc14.%.loc15_11.15 (%.e294f6.1) = splice_inst %.loc15_11.14 [template = %.loc15_11.16 (constants.%.491b27.1)]
// CHECK:STDOUT:     %.loc15_11.5: type = type_of_inst %.loc15_11.17 [template = %.loc15_11.18 (constants.%.516)]
// CHECK:STDOUT:     %.loc15_11.6: @Class.F.loc14.%.loc15_11.18 (%.516) = splice_inst %.loc15_11.17 [template = %.loc15_11.19 (constants.%.3f2)]
// CHECK:STDOUT:     %.loc15_11.7: type = type_of_inst %.loc15_11.20 [template = %.loc15_11.21 (constants.%.8db)]
// CHECK:STDOUT:     %.loc15_11.8: @Class.F.loc14.%.loc15_11.21 (%.8db) = splice_inst %.loc15_11.20 [template = %.loc15_11.22 (constants.%.d67)]
// CHECK:STDOUT:     %.loc15_11.9: @Class.F.loc14.%.loc14_42.5 (%.3cdd63.2) = converted %n.ref, %.loc15_11.8 [template = %.loc15_11.22 (constants.%.d67)]
// CHECK:STDOUT:     %.loc15_11.10: @Class.F.loc14.%.loc14_42.5 (%.3cdd63.2) = splice_inst %.loc15_11.23 [template = %.loc15_11.24 (constants.%.a56)]
// CHECK:STDOUT:     return %.loc15_11.10 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.G.loc21(%T.loc21_20: type) {
// CHECK:STDOUT:   %T.loc21_30: type = symbolic_binding T, 0, template [template = %T.loc21_30 (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc21_30) [template = %Class (constants.%Class)]
// CHECK:STDOUT:   %pattern_type.loc21_30: type = pattern_type %Class [template = %pattern_type.loc21_30 (constants.%pattern_type.36e)]
// CHECK:STDOUT:   %self.param_patt.loc21_30.2: @Class.G.loc21.%pattern_type.loc21_30 (%pattern_type.36e) = value_param_pattern [template = %self.param_patt.loc21_30.2 (constants.%self.param_patt.f93)]
// CHECK:STDOUT:   %self.patt.loc21_30.2: @Class.G.loc21.%pattern_type.loc21_30 (%pattern_type.36e) = wrapper_binding_pattern self, %self.param_patt.loc21_30.2 [template = %self.patt.loc21_30.2 (constants.%self.patt.1dd)]
// CHECK:STDOUT:   %.loc21_40.1: <instruction> = access_member_action %T.ref, U [template]
// CHECK:STDOUT:   %.loc21_40.2: type = type_of_inst %.loc21_40.1 [template = %.loc21_40.2 (constants.%.96c438.3)]
// CHECK:STDOUT:   %.loc21_40.3: @Class.G.loc21.%.loc21_40.2 (%.96c438.3) = splice_inst %.loc21_40.1 [template = %.loc21_40.3 (constants.%.8220de.3)]
// CHECK:STDOUT:   %.loc21_40.4: <instruction> = convert_to_value_action %.loc21_40.8, type [template]
// CHECK:STDOUT:   %.loc21_40.5: type = splice_inst %.loc21_40.4 [template = %.loc21_40.5 (constants.%.3cdd63.3)]
// CHECK:STDOUT:   %.loc21_40.6: Core.Form = init_form %.loc21_40.5 [template = %.loc21_40.6 (constants.%.a428e6.2)]
// CHECK:STDOUT:   %pattern_type.loc21_40: type = pattern_type %.loc21_40.5 [template = %pattern_type.loc21_40 (constants.%pattern_type.8c644d.3)]
// CHECK:STDOUT:   %return.param_patt.loc21_40.2: @Class.G.loc21.%pattern_type.loc21_40 (%pattern_type.8c644d.3) = out_param_pattern [template = %return.param_patt.loc21_40.2 (constants.%return.param_patt.eed287.2)]
// CHECK:STDOUT:   %return.patt.loc21_36.2: @Class.G.loc21.%pattern_type.loc21_40 (%pattern_type.8c644d.3) = return_slot_pattern %return.param_patt.loc21_40.2, %.loc21_40.5 [template = %return.patt.loc21_36.2 (constants.%return.patt.7881ad.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc21_30: <witness> = require_complete_type %Class [template = %require_complete.loc21_30 (constants.%require_complete.4d9)]
// CHECK:STDOUT:   %require_complete.loc21_36: <witness> = require_complete_type %.loc21_40.5 [template = %require_complete.loc21_36 (constants.%require_complete.b6b0e1.3)]
// CHECK:STDOUT:   %.loc22_14.3: <instruction> = access_member_action %self.ref, n [template]
// CHECK:STDOUT:   %.loc22_14.4: type = type_of_inst %.loc22_14.3 [template = %.loc22_14.4 (constants.%.dd9)]
// CHECK:STDOUT:   %.loc22_14.5: @Class.G.loc21.%.loc22_14.4 (%.dd9) = splice_inst %.loc22_14.3 [template = %.loc22_14.5 (constants.%.732)]
// CHECK:STDOUT:   %.loc22_16.11: <instruction> = call_action (constants.%ImplicitAs.generic, constants.%.3cdd63.3), false [template]
// CHECK:STDOUT:   %.loc22_16.12: type = type_of_inst %.loc22_16.11 [template = %.loc22_16.12 (constants.%.3a748a.2)]
// CHECK:STDOUT:   %.loc22_16.13: @Class.G.loc21.%.loc22_16.12 (%.3a748a.2) = splice_inst %.loc22_16.11 [template = %.loc22_16.13 (constants.%.c3d21a.2)]
// CHECK:STDOUT:   %.loc22_16.14: <instruction> = access_member_action %.loc22_16.2, Convert [template]
// CHECK:STDOUT:   %.loc22_16.15: type = type_of_inst %.loc22_16.14 [template = %.loc22_16.15 (constants.%.e294f6.2)]
// CHECK:STDOUT:   %.loc22_16.16: @Class.G.loc21.%.loc22_16.15 (%.e294f6.2) = splice_inst %.loc22_16.14 [template = %.loc22_16.16 (constants.%.491b27.2)]
// CHECK:STDOUT:   %.loc22_16.17: <instruction> = compound_member_access_action %.loc22_14.2, %.loc22_16.4 [template]
// CHECK:STDOUT:   %.loc22_16.18: type = type_of_inst %.loc22_16.17 [template = %.loc22_16.18 (constants.%.b0f)]
// CHECK:STDOUT:   %.loc22_16.19: @Class.G.loc21.%.loc22_16.18 (%.b0f) = splice_inst %.loc22_16.17 [template = %.loc22_16.19 (constants.%.adc)]
// CHECK:STDOUT:   %.loc22_16.20: <instruction> = call_action (%.loc22_16.6), true [template]
// CHECK:STDOUT:   %.loc22_16.21: type = type_of_inst %.loc22_16.20 [template = %.loc22_16.21 (constants.%.09a)]
// CHECK:STDOUT:   %.loc22_16.22: @Class.G.loc21.%.loc22_16.21 (%.09a) = splice_inst %.loc22_16.20 [template = %.loc22_16.22 (constants.%.cb0)]
// CHECK:STDOUT:   %.loc22_16.23: <instruction> = convert_to_category_action %.loc22_16.9, element10 [template]
// CHECK:STDOUT:   %.loc22_16.24: @Class.G.loc21.%.loc21_40.5 (%.3cdd63.3) = splice_inst %.loc22_16.23 [template = %.loc22_16.24 (constants.%.957)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.G.loc21.%Class (%Class)) -> out %return.param: @Class.G.loc21.%.loc21_40.5 (%.3cdd63.3) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.G.loc21.%Class (%Class) = name_ref self, %self
// CHECK:STDOUT:     %.loc22_14.1: type = type_of_inst %.loc22_14.3 [template = %.loc22_14.4 (constants.%.dd9)]
// CHECK:STDOUT:     %.loc22_14.2: @Class.G.loc21.%.loc22_14.4 (%.dd9) = splice_inst %.loc22_14.3 [template = %.loc22_14.5 (constants.%.732)]
// CHECK:STDOUT:     %.loc22_16.1: type = type_of_inst %.loc22_16.11 [template = %.loc22_16.12 (constants.%.3a748a.2)]
// CHECK:STDOUT:     %.loc22_16.2: @Class.G.loc21.%.loc22_16.12 (%.3a748a.2) = splice_inst %.loc22_16.11 [template = %.loc22_16.13 (constants.%.c3d21a.2)]
// CHECK:STDOUT:     %.loc22_16.3: type = type_of_inst %.loc22_16.14 [template = %.loc22_16.15 (constants.%.e294f6.2)]
// CHECK:STDOUT:     %.loc22_16.4: @Class.G.loc21.%.loc22_16.15 (%.e294f6.2) = splice_inst %.loc22_16.14 [template = %.loc22_16.16 (constants.%.491b27.2)]
// CHECK:STDOUT:     %.loc22_16.5: type = type_of_inst %.loc22_16.17 [template = %.loc22_16.18 (constants.%.b0f)]
// CHECK:STDOUT:     %.loc22_16.6: @Class.G.loc21.%.loc22_16.18 (%.b0f) = splice_inst %.loc22_16.17 [template = %.loc22_16.19 (constants.%.adc)]
// CHECK:STDOUT:     %.loc22_16.7: type = type_of_inst %.loc22_16.20 [template = %.loc22_16.21 (constants.%.09a)]
// CHECK:STDOUT:     %.loc22_16.8: @Class.G.loc21.%.loc22_16.21 (%.09a) = splice_inst %.loc22_16.20 [template = %.loc22_16.22 (constants.%.cb0)]
// CHECK:STDOUT:     %.loc22_16.9: @Class.G.loc21.%.loc21_40.5 (%.3cdd63.3) = converted %.loc22_14.2, %.loc22_16.8 [template = %.loc22_16.22 (constants.%.cb0)]
// CHECK:STDOUT:     %.loc22_16.10: @Class.G.loc21.%.loc21_40.5 (%.3cdd63.3) = splice_inst %.loc22_16.23 [template = %.loc22_16.24 (constants.%.957)]
// CHECK:STDOUT:     return %.loc22_16.10 to %return.param
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
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT:   %n.param_patt.loc6_9.2 => constants.%n.param_patt.fb2
// CHECK:STDOUT:   %n.patt.loc6_9.2 => constants.%n.patt.955
// CHECK:STDOUT:   %.loc6_17.1 => constants.%.184347.1
// CHECK:STDOUT:   %return.param_patt.loc6_17.2 => constants.%return.param_patt.41d2d9.1
// CHECK:STDOUT:   %return.patt.loc6_14.2 => constants.%return.patt.b39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.G.loc7(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %pattern_type.loc7_8 => constants.%pattern_type.36e
// CHECK:STDOUT:   %self.param_patt.loc7_8.2 => constants.%self.param_patt.f93
// CHECK:STDOUT:   %self.patt.loc7_8.2 => constants.%self.patt.1dd
// CHECK:STDOUT:   %.loc7_17.1 => constants.%.184347.1
// CHECK:STDOUT:   %pattern_type.loc7_17 => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT:   %return.param_patt.loc7_17.2 => constants.%return.param_patt.41d2d9.1
// CHECK:STDOUT:   %return.patt.loc7_14.2 => constants.%return.patt.b39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.F.loc14(constants.%T) {
// CHECK:STDOUT:   %T.loc14_33 => constants.%T
// CHECK:STDOUT:   %.loc14_34.1 => constants.%.3d9
// CHECK:STDOUT:   %.loc14_34.2 => constants.%.cf6
// CHECK:STDOUT:   %.loc14_34.3 => constants.%.2db
// CHECK:STDOUT:   %.loc14_34.4 => constants.%.dd0
// CHECK:STDOUT:   %.loc14_34.5 => constants.%.3e1
// CHECK:STDOUT:   %pattern_type.loc14_31 => constants.%pattern_type.e0b
// CHECK:STDOUT:   %n.param_patt.loc14_31.2 => constants.%n.param_patt.396
// CHECK:STDOUT:   %n.patt.loc14_31.2 => constants.%n.patt.f9c
// CHECK:STDOUT:   %.loc14_42.1 => constants.%.3d9
// CHECK:STDOUT:   %.loc14_42.2 => constants.%.cf6
// CHECK:STDOUT:   %.loc14_42.3 => constants.%.2db
// CHECK:STDOUT:   %.loc14_42.4 => constants.%.dd0
// CHECK:STDOUT:   %.loc14_42.5 => constants.%.3e1
// CHECK:STDOUT:   %.loc14_42.6 => constants.%.764
// CHECK:STDOUT:   %pattern_type.loc14_42 => constants.%pattern_type.e0b
// CHECK:STDOUT:   %return.param_patt.loc14_42.2 => constants.%return.param_patt.fc1
// CHECK:STDOUT:   %return.patt.loc14_38.2 => constants.%return.patt.fcc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.G.loc21(constants.%T) {
// CHECK:STDOUT:   %T.loc21_30 => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %pattern_type.loc21_30 => constants.%pattern_type.36e
// CHECK:STDOUT:   %self.param_patt.loc21_30.2 => constants.%self.param_patt.f93
// CHECK:STDOUT:   %self.patt.loc21_30.2 => constants.%self.patt.1dd
// CHECK:STDOUT:   %.loc21_40.1 => constants.%.3d9
// CHECK:STDOUT:   %.loc21_40.2 => constants.%.cf6
// CHECK:STDOUT:   %.loc21_40.3 => constants.%.2db
// CHECK:STDOUT:   %.loc21_40.4 => constants.%.dd0
// CHECK:STDOUT:   %.loc21_40.5 => constants.%.3e1
// CHECK:STDOUT:   %.loc21_40.6 => constants.%.764
// CHECK:STDOUT:   %pattern_type.loc21_40 => constants.%pattern_type.e0b
// CHECK:STDOUT:   %return.param_patt.loc21_40.2 => constants.%return.param_patt.fc1
// CHECK:STDOUT:   %return.patt.loc21_36.2 => constants.%return.patt.fcc
// CHECK:STDOUT: }
// CHECK:STDOUT:
