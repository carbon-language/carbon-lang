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

// CHECK:STDERR: fail_todo_basic.carbon:[[@LINE+7]]:30: error: type `<pattern for <splice of <cannot stringify inst78000080: {kind: ConvertToValueAction, arg0: inst7800007D, arg1: inst(TypeType), type: type(inst(InstType))}>>>` of parameter 1 in redeclaration differs from previous parameter type `<pattern for T>` [RedeclParamDiffersType]
// CHECK:STDERR: fn Class(template T: type).F(n: T.U) -> T.U {
// CHECK:STDERR:                              ^~~~~~
// CHECK:STDERR: fail_todo_basic.carbon:[[@LINE-8]]:8: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR:   fn F(n: T) -> T;
// CHECK:STDERR:        ^~~~
// CHECK:STDERR:
fn Class(template T: type).F(n: T.U) -> T.U {
  return n;
}

// CHECK:STDERR: fail_todo_basic.carbon:[[@LINE+7]]:1: error: function redeclaration differs because return type is `<splice of <cannot stringify inst7800014E: {kind: ConvertToValueAction, arg0: inst7800014B, arg1: inst(TypeType), type: type(inst(InstType))}>>` [FunctionRedeclReturnTypeDiffers]
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
// CHECK:STDOUT:   %.eec69b.1: type = splice_inst @Class.F.loc18.%.loc18_34.1 [template]
// CHECK:STDOUT:   %.6af1e3.1: type = type_of_inst @Class.F.loc18.%.loc18_34.3 [template]
// CHECK:STDOUT:   %.75e939.1: %.6af1e3.1 = splice_inst @Class.F.loc18.%.loc18_34.3 [template]
// CHECK:STDOUT:   %.a9ead5.1: type = splice_inst @Class.F.loc18.%.loc18_34.6 [template]
// CHECK:STDOUT:   %pattern_type.fcfd9a.1: type = pattern_type %.a9ead5.1 [template]
// CHECK:STDOUT:   %n.param_patt.618: %pattern_type.fcfd9a.1 = value_param_pattern [template]
// CHECK:STDOUT:   %n.patt.329: %pattern_type.fcfd9a.1 = wrapper_binding_pattern n, %n.param_patt.618 [template]
// CHECK:STDOUT:   %.eec69b.2: type = splice_inst @Class.F.loc18.%.loc18_42.1 [template]
// CHECK:STDOUT:   %.6af1e3.2: type = type_of_inst @Class.F.loc18.%.loc18_42.3 [template]
// CHECK:STDOUT:   %.75e939.2: %.6af1e3.2 = splice_inst @Class.F.loc18.%.loc18_42.3 [template]
// CHECK:STDOUT:   %.a9ead5.2: type = splice_inst @Class.F.loc18.%.loc18_42.6 [template]
// CHECK:STDOUT:   %.cad8b1.1: Core.Form = init_form %.a9ead5.2 [template]
// CHECK:STDOUT:   %pattern_type.fcfd9a.2: type = pattern_type %.a9ead5.2 [template]
// CHECK:STDOUT:   %return.param_patt.70fba7.1: %pattern_type.fcfd9a.2 = out_param_pattern [template]
// CHECK:STDOUT:   %return.patt.0d3581.1: %pattern_type.fcfd9a.2 = return_slot_pattern %return.param_patt.70fba7.1, %.a9ead5.2 [template]
// CHECK:STDOUT:   %.dc4: <instruction> = refine_inst_action %T [template]
// CHECK:STDOUT:   %.40e: type = splice_inst %.dc4 [template]
// CHECK:STDOUT:   %.ac6: <instruction> = access_member_action %.40e, U [template]
// CHECK:STDOUT:   %.86f: type = type_of_inst %.ac6 [template]
// CHECK:STDOUT:   %.e76: %.86f = splice_inst %.ac6 [template]
// CHECK:STDOUT:   %.3c7: <instruction> = convert_to_value_action %.e76, type [template]
// CHECK:STDOUT:   %.126: type = splice_inst %.3c7 [template]
// CHECK:STDOUT:   %pattern_type.426: type = pattern_type %.126 [template]
// CHECK:STDOUT:   %n.param_patt.e32: %pattern_type.426 = value_param_pattern [template]
// CHECK:STDOUT:   %n.patt.e5e: %pattern_type.426 = wrapper_binding_pattern n, %n.param_patt.e32 [template]
// CHECK:STDOUT:   %.53f: Core.Form = init_form %.126 [template]
// CHECK:STDOUT:   %return.param_patt.894: %pattern_type.426 = out_param_pattern [template]
// CHECK:STDOUT:   %return.patt.a56: %pattern_type.426 = return_slot_pattern %return.param_patt.894, %.126 [template]
// CHECK:STDOUT:   %Class.F.type.9829b4.2: type = fn_type @Class.F.loc18, @Class(%T) [template]
// CHECK:STDOUT:   %Class.F.c6358f.2: %Class.F.type.9829b4.2 = struct_value () [template]
// CHECK:STDOUT:   %require_complete.f034fc.1: <witness> = require_complete_type %.a9ead5.1 [template]
// CHECK:STDOUT:   %require_complete.f034fc.2: <witness> = require_complete_type %.a9ead5.2 [template]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %.a1c6f1.1: type = type_of_inst @Class.F.loc18.%.loc19_11.13 [template]
// CHECK:STDOUT:   %.488baf.1: %.a1c6f1.1 = splice_inst @Class.F.loc18.%.loc19_11.13 [template]
// CHECK:STDOUT:   %.21542d.1: type = type_of_inst @Class.F.loc18.%.loc19_11.16 [template]
// CHECK:STDOUT:   %.e90b03.1: %.21542d.1 = splice_inst @Class.F.loc18.%.loc19_11.16 [template]
// CHECK:STDOUT:   %.a32: %.a9ead5.1 = splice_inst @Class.F.loc18.%.loc19_11.19 [template]
// CHECK:STDOUT:   %.0d8: type = type_of_inst @Class.F.loc18.%.loc19_11.21 [template]
// CHECK:STDOUT:   %.a51: %.0d8 = splice_inst @Class.F.loc18.%.loc19_11.21 [template]
// CHECK:STDOUT:   %.097: type = type_of_inst @Class.F.loc18.%.loc19_11.24 [template]
// CHECK:STDOUT:   %.d6e: %.097 = splice_inst @Class.F.loc18.%.loc19_11.24 [template]
// CHECK:STDOUT:   %.1d1: %.a9ead5.2 = splice_inst @Class.F.loc18.%.loc19_11.27 [template]
// CHECK:STDOUT:   %.778: %.a9ead5.2 = splice_inst @Class.F.loc18.%.loc19_11.29 [template]
// CHECK:STDOUT:   %.eec69b.3: type = splice_inst @Class.G.loc29.%.loc29_40.1 [template]
// CHECK:STDOUT:   %.6af1e3.3: type = type_of_inst @Class.G.loc29.%.loc29_40.3 [template]
// CHECK:STDOUT:   %.75e939.3: %.6af1e3.3 = splice_inst @Class.G.loc29.%.loc29_40.3 [template]
// CHECK:STDOUT:   %.a9ead5.3: type = splice_inst @Class.G.loc29.%.loc29_40.6 [template]
// CHECK:STDOUT:   %.cad8b1.2: Core.Form = init_form %.a9ead5.3 [template]
// CHECK:STDOUT:   %pattern_type.fcfd9a.3: type = pattern_type %.a9ead5.3 [template]
// CHECK:STDOUT:   %return.param_patt.70fba7.2: %pattern_type.fcfd9a.3 = out_param_pattern [template]
// CHECK:STDOUT:   %return.patt.0d3581.2: %pattern_type.fcfd9a.3 = return_slot_pattern %return.param_patt.70fba7.2, %.a9ead5.3 [template]
// CHECK:STDOUT:   %Class.G.type.ea69e8.2: type = fn_type @Class.G.loc29, @Class(%T) [template]
// CHECK:STDOUT:   %Class.G.5aff6a.2: %Class.G.type.ea69e8.2 = struct_value () [template]
// CHECK:STDOUT:   %require_complete.4d9: <witness> = require_complete_type %Class [template]
// CHECK:STDOUT:   %require_complete.f034fc.3: <witness> = require_complete_type %.a9ead5.3 [template]
// CHECK:STDOUT:   %.aec: %Class = splice_inst @Class.G.loc29.%.loc30_14.4 [template]
// CHECK:STDOUT:   %.521: type = type_of_inst @Class.G.loc29.%.loc30_14.6 [template]
// CHECK:STDOUT:   %.54b: %.521 = splice_inst @Class.G.loc29.%.loc30_14.6 [template]
// CHECK:STDOUT:   %.a1c6f1.2: type = type_of_inst @Class.G.loc29.%.loc30_16.12 [template]
// CHECK:STDOUT:   %.488baf.2: %.a1c6f1.2 = splice_inst @Class.G.loc29.%.loc30_16.12 [template]
// CHECK:STDOUT:   %.21542d.2: type = type_of_inst @Class.G.loc29.%.loc30_16.15 [template]
// CHECK:STDOUT:   %.e90b03.2: %.21542d.2 = splice_inst @Class.G.loc29.%.loc30_16.15 [template]
// CHECK:STDOUT:   %.7d0: type = type_of_inst @Class.G.loc29.%.loc30_16.18 [template]
// CHECK:STDOUT:   %.d0d: %.7d0 = splice_inst @Class.G.loc29.%.loc30_16.18 [template]
// CHECK:STDOUT:   %.dff: type = type_of_inst @Class.G.loc29.%.loc30_16.21 [template]
// CHECK:STDOUT:   %.bc2: %.dff = splice_inst @Class.G.loc29.%.loc30_16.21 [template]
// CHECK:STDOUT:   %.e26: %.a9ead5.3 = splice_inst @Class.G.loc29.%.loc30_16.24 [template]
// CHECK:STDOUT:   %.08d: %.a9ead5.3 = splice_inst @Class.G.loc29.%.loc30_16.26 [template]
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
// CHECK:STDOUT:     %n.param_patt.loc18_31.1: @Class.F.loc18.%pattern_type.loc18_31 (%pattern_type.fcfd9a.1) = value_param_pattern [template = %n.param_patt.loc18_31.2 (constants.%n.param_patt.618)]
// CHECK:STDOUT:     %n.patt.loc18_31.1: @Class.F.loc18.%pattern_type.loc18_31 (%pattern_type.fcfd9a.1) = wrapper_binding_pattern n, %n.param_patt.loc18_31.1 [template = %n.patt.loc18_31.2 (constants.%n.patt.329)]
// CHECK:STDOUT:     %return.param_patt.loc18_42.1: @Class.F.loc18.%pattern_type.loc18_42 (%pattern_type.fcfd9a.2) = out_param_pattern [template = %return.param_patt.loc18_42.2 (constants.%return.param_patt.70fba7.1)]
// CHECK:STDOUT:     %return.patt.loc18_38.1: @Class.F.loc18.%pattern_type.loc18_42 (%pattern_type.fcfd9a.2) = return_slot_pattern %return.param_patt.loc18_42.1, %.loc18_42.12 [template = %return.patt.loc18_38.2 (constants.%return.patt.0d3581.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18_22.1: type = splice_block %.loc18_22.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc18_22.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc18_20: type = symbolic_binding T, 0, template [template = @Class.%T.loc5_23.1 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc18_41: type = name_ref T, %T.loc18_20 [template = %T.loc18_33 (constants.%T)]
// CHECK:STDOUT:     %.loc18_42.9: type = splice_inst %.loc18_42.1 [template = %.loc18_42.2 (constants.%.eec69b.2)]
// CHECK:STDOUT:     %.loc18_42.10: type = type_of_inst %.loc18_42.3 [template = %.loc18_42.4 (constants.%.6af1e3.2)]
// CHECK:STDOUT:     %.loc18_42.11: @Class.F.loc18.%.loc18_42.4 (%.6af1e3.2) = splice_inst %.loc18_42.3 [template = %.loc18_42.5 (constants.%.75e939.2)]
// CHECK:STDOUT:     %.loc18_42.12: type = splice_inst %.loc18_42.6 [template = %.loc18_42.7 (constants.%.a9ead5.2)]
// CHECK:STDOUT:     %.loc18_42.13: Core.Form = init_form %.loc18_42.12 [template = %.loc18_42.8 (constants.%.cad8b1.1)]
// CHECK:STDOUT:     %n.param: @Class.F.loc18.%.loc18_34.7 (%.a9ead5.1) = value_param call_param0
// CHECK:STDOUT:     %.loc18_34.8: type = splice_block %.loc18_34.12 [template = %.loc18_34.7 (constants.%.a9ead5.1)] {
// CHECK:STDOUT:       %T.ref.loc18_33: type = name_ref T, %T.loc18_20 [template = %T.loc18_33 (constants.%T)]
// CHECK:STDOUT:       %.loc18_34.9: type = splice_inst %.loc18_34.1 [template = %.loc18_34.2 (constants.%.eec69b.1)]
// CHECK:STDOUT:       %.loc18_34.10: type = type_of_inst %.loc18_34.3 [template = %.loc18_34.4 (constants.%.6af1e3.1)]
// CHECK:STDOUT:       %.loc18_34.11: @Class.F.loc18.%.loc18_34.4 (%.6af1e3.1) = splice_inst %.loc18_34.3 [template = %.loc18_34.5 (constants.%.75e939.1)]
// CHECK:STDOUT:       %.loc18_34.12: type = splice_inst %.loc18_34.6 [template = %.loc18_34.7 (constants.%.a9ead5.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: @Class.F.loc18.%.loc18_34.7 (%.a9ead5.1) = wrapper_binding n, %n.param
// CHECK:STDOUT:     %return.param: ref @Class.F.loc18.%.loc18_42.7 (%.a9ead5.2) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Class.F.loc18.%.loc18_42.7 (%.a9ead5.2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.G.decl: %Class.G.type.ea69e8.2 = fn_decl @Class.G.loc29 [template = constants.%Class.G.5aff6a.2] {
// CHECK:STDOUT:     %self.param_patt.loc29_30.1: @Class.G.loc29.%pattern_type.loc29_30 (%pattern_type.36e) = value_param_pattern [template = %self.param_patt.loc29_30.2 (constants.%self.param_patt.f93)]
// CHECK:STDOUT:     %self.patt.loc29_30.1: @Class.G.loc29.%pattern_type.loc29_30 (%pattern_type.36e) = wrapper_binding_pattern self, %self.param_patt.loc29_30.1 [template = %self.patt.loc29_30.2 (constants.%self.patt.1dd)]
// CHECK:STDOUT:     %return.param_patt.loc29_40.1: @Class.G.loc29.%pattern_type.loc29_40 (%pattern_type.fcfd9a.3) = out_param_pattern [template = %return.param_patt.loc29_40.2 (constants.%return.param_patt.70fba7.2)]
// CHECK:STDOUT:     %return.patt.loc29_36.1: @Class.G.loc29.%pattern_type.loc29_40 (%pattern_type.fcfd9a.3) = return_slot_pattern %return.param_patt.loc29_40.1, %.loc29_40.12 [template = %return.patt.loc29_36.2 (constants.%return.patt.0d3581.2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc29_22.1: type = splice_block %.loc29_22.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc29_22.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc29_20: type = symbolic_binding T, 0, template [template = @Class.%T.loc5_23.1 (constants.%T)]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc29_20 [template = %T.loc29_30 (constants.%T)]
// CHECK:STDOUT:     %.loc29_40.9: type = splice_inst %.loc29_40.1 [template = %.loc29_40.2 (constants.%.eec69b.3)]
// CHECK:STDOUT:     %.loc29_40.10: type = type_of_inst %.loc29_40.3 [template = %.loc29_40.4 (constants.%.6af1e3.3)]
// CHECK:STDOUT:     %.loc29_40.11: @Class.G.loc29.%.loc29_40.4 (%.6af1e3.3) = splice_inst %.loc29_40.3 [template = %.loc29_40.5 (constants.%.75e939.3)]
// CHECK:STDOUT:     %.loc29_40.12: type = splice_inst %.loc29_40.6 [template = %.loc29_40.7 (constants.%.a9ead5.3)]
// CHECK:STDOUT:     %.loc29_40.13: Core.Form = init_form %.loc29_40.12 [template = %.loc29_40.8 (constants.%.cad8b1.2)]
// CHECK:STDOUT:     %self.param: @Class.G.loc29.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:     %.loc29_30.1: type = splice_block %Self.ref [template = %Class (constants.%Class)] {
// CHECK:STDOUT:       %.loc29_30.2: type = specific_constant constants.%Class, @Class(constants.%T) [template = %Class (constants.%Class)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc29_30.2 [template = %Class (constants.%Class)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Class.G.loc29.%Class (%Class) = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref @Class.G.loc29.%.loc29_40.7 (%.a9ead5.3) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Class.G.loc29.%.loc29_40.7 (%.a9ead5.3) = return_slot %return.param
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
// CHECK:STDOUT: generic fn @Class.F.loc18(%T.loc18_20: type) {
// CHECK:STDOUT:   %T.loc18_33: type = symbolic_binding T, 0, template [template = %T.loc18_33 (constants.%T)]
// CHECK:STDOUT:   %.loc18_34.1: <instruction> = refine_inst_action %T.ref.loc18_33 [template]
// CHECK:STDOUT:   %.loc18_34.2: type = splice_inst %.loc18_34.1 [template = %.loc18_34.2 (constants.%.eec69b.1)]
// CHECK:STDOUT:   %.loc18_34.3: <instruction> = access_member_action %.loc18_34.9, U [template]
// CHECK:STDOUT:   %.loc18_34.4: type = type_of_inst %.loc18_34.3 [template = %.loc18_34.4 (constants.%.6af1e3.1)]
// CHECK:STDOUT:   %.loc18_34.5: @Class.F.loc18.%.loc18_34.4 (%.6af1e3.1) = splice_inst %.loc18_34.3 [template = %.loc18_34.5 (constants.%.75e939.1)]
// CHECK:STDOUT:   %.loc18_34.6: <instruction> = convert_to_value_action %.loc18_34.11, type [template]
// CHECK:STDOUT:   %.loc18_34.7: type = splice_inst %.loc18_34.6 [template = %.loc18_34.7 (constants.%.a9ead5.1)]
// CHECK:STDOUT:   %pattern_type.loc18_31: type = pattern_type %.loc18_34.7 [template = %pattern_type.loc18_31 (constants.%pattern_type.fcfd9a.1)]
// CHECK:STDOUT:   %n.param_patt.loc18_31.2: @Class.F.loc18.%pattern_type.loc18_31 (%pattern_type.fcfd9a.1) = value_param_pattern [template = %n.param_patt.loc18_31.2 (constants.%n.param_patt.618)]
// CHECK:STDOUT:   %n.patt.loc18_31.2: @Class.F.loc18.%pattern_type.loc18_31 (%pattern_type.fcfd9a.1) = wrapper_binding_pattern n, %n.param_patt.loc18_31.2 [template = %n.patt.loc18_31.2 (constants.%n.patt.329)]
// CHECK:STDOUT:   %.loc18_42.1: <instruction> = refine_inst_action %T.ref.loc18_41 [template]
// CHECK:STDOUT:   %.loc18_42.2: type = splice_inst %.loc18_42.1 [template = %.loc18_42.2 (constants.%.eec69b.2)]
// CHECK:STDOUT:   %.loc18_42.3: <instruction> = access_member_action %.loc18_42.9, U [template]
// CHECK:STDOUT:   %.loc18_42.4: type = type_of_inst %.loc18_42.3 [template = %.loc18_42.4 (constants.%.6af1e3.2)]
// CHECK:STDOUT:   %.loc18_42.5: @Class.F.loc18.%.loc18_42.4 (%.6af1e3.2) = splice_inst %.loc18_42.3 [template = %.loc18_42.5 (constants.%.75e939.2)]
// CHECK:STDOUT:   %.loc18_42.6: <instruction> = convert_to_value_action %.loc18_42.11, type [template]
// CHECK:STDOUT:   %.loc18_42.7: type = splice_inst %.loc18_42.6 [template = %.loc18_42.7 (constants.%.a9ead5.2)]
// CHECK:STDOUT:   %.loc18_42.8: Core.Form = init_form %.loc18_42.7 [template = %.loc18_42.8 (constants.%.cad8b1.1)]
// CHECK:STDOUT:   %pattern_type.loc18_42: type = pattern_type %.loc18_42.7 [template = %pattern_type.loc18_42 (constants.%pattern_type.fcfd9a.2)]
// CHECK:STDOUT:   %return.param_patt.loc18_42.2: @Class.F.loc18.%pattern_type.loc18_42 (%pattern_type.fcfd9a.2) = out_param_pattern [template = %return.param_patt.loc18_42.2 (constants.%return.param_patt.70fba7.1)]
// CHECK:STDOUT:   %return.patt.loc18_38.2: @Class.F.loc18.%pattern_type.loc18_42 (%pattern_type.fcfd9a.2) = return_slot_pattern %return.param_patt.loc18_42.2, %.loc18_42.7 [template = %return.patt.loc18_38.2 (constants.%return.patt.0d3581.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc18_31: <witness> = require_complete_type %.loc18_34.7 [template = %require_complete.loc18_31 (constants.%require_complete.f034fc.1)]
// CHECK:STDOUT:   %require_complete.loc18_38: <witness> = require_complete_type %.loc18_42.7 [template = %require_complete.loc18_38 (constants.%require_complete.f034fc.2)]
// CHECK:STDOUT:   %.loc19_11.13: <instruction> = call_action (constants.%ImplicitAs.generic, constants.%.a9ead5.2), false [template]
// CHECK:STDOUT:   %.loc19_11.14: type = type_of_inst %.loc19_11.13 [template = %.loc19_11.14 (constants.%.a1c6f1.1)]
// CHECK:STDOUT:   %.loc19_11.15: @Class.F.loc18.%.loc19_11.14 (%.a1c6f1.1) = splice_inst %.loc19_11.13 [template = %.loc19_11.15 (constants.%.488baf.1)]
// CHECK:STDOUT:   %.loc19_11.16: <instruction> = access_member_action %.loc19_11.2, Convert [template]
// CHECK:STDOUT:   %.loc19_11.17: type = type_of_inst %.loc19_11.16 [template = %.loc19_11.17 (constants.%.21542d.1)]
// CHECK:STDOUT:   %.loc19_11.18: @Class.F.loc18.%.loc19_11.17 (%.21542d.1) = splice_inst %.loc19_11.16 [template = %.loc19_11.18 (constants.%.e90b03.1)]
// CHECK:STDOUT:   %.loc19_11.19: <instruction> = refine_inst_action %n.ref [template]
// CHECK:STDOUT:   %.loc19_11.20: @Class.F.loc18.%.loc18_34.7 (%.a9ead5.1) = splice_inst %.loc19_11.19 [template = %.loc19_11.20 (constants.%.a32)]
// CHECK:STDOUT:   %.loc19_11.21: <instruction> = compound_member_access_action %.loc19_11.5, %.loc19_11.4 [template]
// CHECK:STDOUT:   %.loc19_11.22: type = type_of_inst %.loc19_11.21 [template = %.loc19_11.22 (constants.%.0d8)]
// CHECK:STDOUT:   %.loc19_11.23: @Class.F.loc18.%.loc19_11.22 (%.0d8) = splice_inst %.loc19_11.21 [template = %.loc19_11.23 (constants.%.a51)]
// CHECK:STDOUT:   %.loc19_11.24: <instruction> = call_action (%.loc19_11.7), true [template]
// CHECK:STDOUT:   %.loc19_11.25: type = type_of_inst %.loc19_11.24 [template = %.loc19_11.25 (constants.%.097)]
// CHECK:STDOUT:   %.loc19_11.26: @Class.F.loc18.%.loc19_11.25 (%.097) = splice_inst %.loc19_11.24 [template = %.loc19_11.26 (constants.%.d6e)]
// CHECK:STDOUT:   %.loc19_11.27: <instruction> = refine_inst_action %.loc19_11.10 [template]
// CHECK:STDOUT:   %.loc19_11.28: @Class.F.loc18.%.loc18_42.7 (%.a9ead5.2) = splice_inst %.loc19_11.27 [template = %.loc19_11.28 (constants.%.1d1)]
// CHECK:STDOUT:   %.loc19_11.29: <instruction> = convert_to_category_action %.loc19_11.11, element10 [template]
// CHECK:STDOUT:   %.loc19_11.30: @Class.F.loc18.%.loc18_42.7 (%.a9ead5.2) = splice_inst %.loc19_11.29 [template = %.loc19_11.30 (constants.%.778)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param: @Class.F.loc18.%.loc18_34.7 (%.a9ead5.1)) -> out %return.param: @Class.F.loc18.%.loc18_42.7 (%.a9ead5.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %n.ref: @Class.F.loc18.%.loc18_34.7 (%.a9ead5.1) = name_ref n, %n
// CHECK:STDOUT:     %.loc19_11.1: type = type_of_inst %.loc19_11.13 [template = %.loc19_11.14 (constants.%.a1c6f1.1)]
// CHECK:STDOUT:     %.loc19_11.2: @Class.F.loc18.%.loc19_11.14 (%.a1c6f1.1) = splice_inst %.loc19_11.13 [template = %.loc19_11.15 (constants.%.488baf.1)]
// CHECK:STDOUT:     %.loc19_11.3: type = type_of_inst %.loc19_11.16 [template = %.loc19_11.17 (constants.%.21542d.1)]
// CHECK:STDOUT:     %.loc19_11.4: @Class.F.loc18.%.loc19_11.17 (%.21542d.1) = splice_inst %.loc19_11.16 [template = %.loc19_11.18 (constants.%.e90b03.1)]
// CHECK:STDOUT:     %.loc19_11.5: @Class.F.loc18.%.loc18_34.7 (%.a9ead5.1) = splice_inst %.loc19_11.19 [template = %.loc19_11.20 (constants.%.a32)]
// CHECK:STDOUT:     %.loc19_11.6: type = type_of_inst %.loc19_11.21 [template = %.loc19_11.22 (constants.%.0d8)]
// CHECK:STDOUT:     %.loc19_11.7: @Class.F.loc18.%.loc19_11.22 (%.0d8) = splice_inst %.loc19_11.21 [template = %.loc19_11.23 (constants.%.a51)]
// CHECK:STDOUT:     %.loc19_11.8: type = type_of_inst %.loc19_11.24 [template = %.loc19_11.25 (constants.%.097)]
// CHECK:STDOUT:     %.loc19_11.9: @Class.F.loc18.%.loc19_11.25 (%.097) = splice_inst %.loc19_11.24 [template = %.loc19_11.26 (constants.%.d6e)]
// CHECK:STDOUT:     %.loc19_11.10: @Class.F.loc18.%.loc18_42.7 (%.a9ead5.2) = converted %n.ref, %.loc19_11.9 [template = %.loc19_11.26 (constants.%.d6e)]
// CHECK:STDOUT:     %.loc19_11.11: @Class.F.loc18.%.loc18_42.7 (%.a9ead5.2) = splice_inst %.loc19_11.27 [template = %.loc19_11.28 (constants.%.1d1)]
// CHECK:STDOUT:     %.loc19_11.12: @Class.F.loc18.%.loc18_42.7 (%.a9ead5.2) = splice_inst %.loc19_11.29 [template = %.loc19_11.30 (constants.%.778)]
// CHECK:STDOUT:     return %.loc19_11.12 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.G.loc29(%T.loc29_20: type) {
// CHECK:STDOUT:   %T.loc29_30: type = symbolic_binding T, 0, template [template = %T.loc29_30 (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc29_30) [template = %Class (constants.%Class)]
// CHECK:STDOUT:   %pattern_type.loc29_30: type = pattern_type %Class [template = %pattern_type.loc29_30 (constants.%pattern_type.36e)]
// CHECK:STDOUT:   %self.param_patt.loc29_30.2: @Class.G.loc29.%pattern_type.loc29_30 (%pattern_type.36e) = value_param_pattern [template = %self.param_patt.loc29_30.2 (constants.%self.param_patt.f93)]
// CHECK:STDOUT:   %self.patt.loc29_30.2: @Class.G.loc29.%pattern_type.loc29_30 (%pattern_type.36e) = wrapper_binding_pattern self, %self.param_patt.loc29_30.2 [template = %self.patt.loc29_30.2 (constants.%self.patt.1dd)]
// CHECK:STDOUT:   %.loc29_40.1: <instruction> = refine_inst_action %T.ref [template]
// CHECK:STDOUT:   %.loc29_40.2: type = splice_inst %.loc29_40.1 [template = %.loc29_40.2 (constants.%.eec69b.3)]
// CHECK:STDOUT:   %.loc29_40.3: <instruction> = access_member_action %.loc29_40.9, U [template]
// CHECK:STDOUT:   %.loc29_40.4: type = type_of_inst %.loc29_40.3 [template = %.loc29_40.4 (constants.%.6af1e3.3)]
// CHECK:STDOUT:   %.loc29_40.5: @Class.G.loc29.%.loc29_40.4 (%.6af1e3.3) = splice_inst %.loc29_40.3 [template = %.loc29_40.5 (constants.%.75e939.3)]
// CHECK:STDOUT:   %.loc29_40.6: <instruction> = convert_to_value_action %.loc29_40.11, type [template]
// CHECK:STDOUT:   %.loc29_40.7: type = splice_inst %.loc29_40.6 [template = %.loc29_40.7 (constants.%.a9ead5.3)]
// CHECK:STDOUT:   %.loc29_40.8: Core.Form = init_form %.loc29_40.7 [template = %.loc29_40.8 (constants.%.cad8b1.2)]
// CHECK:STDOUT:   %pattern_type.loc29_40: type = pattern_type %.loc29_40.7 [template = %pattern_type.loc29_40 (constants.%pattern_type.fcfd9a.3)]
// CHECK:STDOUT:   %return.param_patt.loc29_40.2: @Class.G.loc29.%pattern_type.loc29_40 (%pattern_type.fcfd9a.3) = out_param_pattern [template = %return.param_patt.loc29_40.2 (constants.%return.param_patt.70fba7.2)]
// CHECK:STDOUT:   %return.patt.loc29_36.2: @Class.G.loc29.%pattern_type.loc29_40 (%pattern_type.fcfd9a.3) = return_slot_pattern %return.param_patt.loc29_40.2, %.loc29_40.7 [template = %return.patt.loc29_36.2 (constants.%return.patt.0d3581.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc29_30: <witness> = require_complete_type %Class [template = %require_complete.loc29_30 (constants.%require_complete.4d9)]
// CHECK:STDOUT:   %require_complete.loc29_36: <witness> = require_complete_type %.loc29_40.7 [template = %require_complete.loc29_36 (constants.%require_complete.f034fc.3)]
// CHECK:STDOUT:   %.loc30_14.4: <instruction> = refine_inst_action %self.ref [template]
// CHECK:STDOUT:   %.loc30_14.5: @Class.G.loc29.%Class (%Class) = splice_inst %.loc30_14.4 [template = %.loc30_14.5 (constants.%.aec)]
// CHECK:STDOUT:   %.loc30_14.6: <instruction> = access_member_action %.loc30_14.1, n [template]
// CHECK:STDOUT:   %.loc30_14.7: type = type_of_inst %.loc30_14.6 [template = %.loc30_14.7 (constants.%.521)]
// CHECK:STDOUT:   %.loc30_14.8: @Class.G.loc29.%.loc30_14.7 (%.521) = splice_inst %.loc30_14.6 [template = %.loc30_14.8 (constants.%.54b)]
// CHECK:STDOUT:   %.loc30_16.12: <instruction> = call_action (constants.%ImplicitAs.generic, constants.%.a9ead5.3), false [template]
// CHECK:STDOUT:   %.loc30_16.13: type = type_of_inst %.loc30_16.12 [template = %.loc30_16.13 (constants.%.a1c6f1.2)]
// CHECK:STDOUT:   %.loc30_16.14: @Class.G.loc29.%.loc30_16.13 (%.a1c6f1.2) = splice_inst %.loc30_16.12 [template = %.loc30_16.14 (constants.%.488baf.2)]
// CHECK:STDOUT:   %.loc30_16.15: <instruction> = access_member_action %.loc30_16.2, Convert [template]
// CHECK:STDOUT:   %.loc30_16.16: type = type_of_inst %.loc30_16.15 [template = %.loc30_16.16 (constants.%.21542d.2)]
// CHECK:STDOUT:   %.loc30_16.17: @Class.G.loc29.%.loc30_16.16 (%.21542d.2) = splice_inst %.loc30_16.15 [template = %.loc30_16.17 (constants.%.e90b03.2)]
// CHECK:STDOUT:   %.loc30_16.18: <instruction> = compound_member_access_action %.loc30_14.3, %.loc30_16.4 [template]
// CHECK:STDOUT:   %.loc30_16.19: type = type_of_inst %.loc30_16.18 [template = %.loc30_16.19 (constants.%.7d0)]
// CHECK:STDOUT:   %.loc30_16.20: @Class.G.loc29.%.loc30_16.19 (%.7d0) = splice_inst %.loc30_16.18 [template = %.loc30_16.20 (constants.%.d0d)]
// CHECK:STDOUT:   %.loc30_16.21: <instruction> = call_action (%.loc30_16.6), true [template]
// CHECK:STDOUT:   %.loc30_16.22: type = type_of_inst %.loc30_16.21 [template = %.loc30_16.22 (constants.%.dff)]
// CHECK:STDOUT:   %.loc30_16.23: @Class.G.loc29.%.loc30_16.22 (%.dff) = splice_inst %.loc30_16.21 [template = %.loc30_16.23 (constants.%.bc2)]
// CHECK:STDOUT:   %.loc30_16.24: <instruction> = refine_inst_action %.loc30_16.9 [template]
// CHECK:STDOUT:   %.loc30_16.25: @Class.G.loc29.%.loc29_40.7 (%.a9ead5.3) = splice_inst %.loc30_16.24 [template = %.loc30_16.25 (constants.%.e26)]
// CHECK:STDOUT:   %.loc30_16.26: <instruction> = convert_to_category_action %.loc30_16.10, element10 [template]
// CHECK:STDOUT:   %.loc30_16.27: @Class.G.loc29.%.loc29_40.7 (%.a9ead5.3) = splice_inst %.loc30_16.26 [template = %.loc30_16.27 (constants.%.08d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.G.loc29.%Class (%Class)) -> out %return.param: @Class.G.loc29.%.loc29_40.7 (%.a9ead5.3) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.G.loc29.%Class (%Class) = name_ref self, %self
// CHECK:STDOUT:     %.loc30_14.1: @Class.G.loc29.%Class (%Class) = splice_inst %.loc30_14.4 [template = %.loc30_14.5 (constants.%.aec)]
// CHECK:STDOUT:     %.loc30_14.2: type = type_of_inst %.loc30_14.6 [template = %.loc30_14.7 (constants.%.521)]
// CHECK:STDOUT:     %.loc30_14.3: @Class.G.loc29.%.loc30_14.7 (%.521) = splice_inst %.loc30_14.6 [template = %.loc30_14.8 (constants.%.54b)]
// CHECK:STDOUT:     %.loc30_16.1: type = type_of_inst %.loc30_16.12 [template = %.loc30_16.13 (constants.%.a1c6f1.2)]
// CHECK:STDOUT:     %.loc30_16.2: @Class.G.loc29.%.loc30_16.13 (%.a1c6f1.2) = splice_inst %.loc30_16.12 [template = %.loc30_16.14 (constants.%.488baf.2)]
// CHECK:STDOUT:     %.loc30_16.3: type = type_of_inst %.loc30_16.15 [template = %.loc30_16.16 (constants.%.21542d.2)]
// CHECK:STDOUT:     %.loc30_16.4: @Class.G.loc29.%.loc30_16.16 (%.21542d.2) = splice_inst %.loc30_16.15 [template = %.loc30_16.17 (constants.%.e90b03.2)]
// CHECK:STDOUT:     %.loc30_16.5: type = type_of_inst %.loc30_16.18 [template = %.loc30_16.19 (constants.%.7d0)]
// CHECK:STDOUT:     %.loc30_16.6: @Class.G.loc29.%.loc30_16.19 (%.7d0) = splice_inst %.loc30_16.18 [template = %.loc30_16.20 (constants.%.d0d)]
// CHECK:STDOUT:     %.loc30_16.7: type = type_of_inst %.loc30_16.21 [template = %.loc30_16.22 (constants.%.dff)]
// CHECK:STDOUT:     %.loc30_16.8: @Class.G.loc29.%.loc30_16.22 (%.dff) = splice_inst %.loc30_16.21 [template = %.loc30_16.23 (constants.%.bc2)]
// CHECK:STDOUT:     %.loc30_16.9: @Class.G.loc29.%.loc29_40.7 (%.a9ead5.3) = converted %.loc30_14.3, %.loc30_16.8 [template = %.loc30_16.23 (constants.%.bc2)]
// CHECK:STDOUT:     %.loc30_16.10: @Class.G.loc29.%.loc29_40.7 (%.a9ead5.3) = splice_inst %.loc30_16.24 [template = %.loc30_16.25 (constants.%.e26)]
// CHECK:STDOUT:     %.loc30_16.11: @Class.G.loc29.%.loc29_40.7 (%.a9ead5.3) = splice_inst %.loc30_16.26 [template = %.loc30_16.27 (constants.%.08d)]
// CHECK:STDOUT:     return %.loc30_16.11 to %return.param
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
// CHECK:STDOUT: specific @Class.F.loc18(constants.%T) {
// CHECK:STDOUT:   %T.loc18_33 => constants.%T
// CHECK:STDOUT:   %.loc18_34.1 => constants.%.dc4
// CHECK:STDOUT:   %.loc18_34.2 => constants.%.40e
// CHECK:STDOUT:   %.loc18_34.3 => constants.%.ac6
// CHECK:STDOUT:   %.loc18_34.4 => constants.%.86f
// CHECK:STDOUT:   %.loc18_34.5 => constants.%.e76
// CHECK:STDOUT:   %.loc18_34.6 => constants.%.3c7
// CHECK:STDOUT:   %.loc18_34.7 => constants.%.126
// CHECK:STDOUT:   %pattern_type.loc18_31 => constants.%pattern_type.426
// CHECK:STDOUT:   %n.param_patt.loc18_31.2 => constants.%n.param_patt.e32
// CHECK:STDOUT:   %n.patt.loc18_31.2 => constants.%n.patt.e5e
// CHECK:STDOUT:   %.loc18_42.1 => constants.%.dc4
// CHECK:STDOUT:   %.loc18_42.2 => constants.%.40e
// CHECK:STDOUT:   %.loc18_42.3 => constants.%.ac6
// CHECK:STDOUT:   %.loc18_42.4 => constants.%.86f
// CHECK:STDOUT:   %.loc18_42.5 => constants.%.e76
// CHECK:STDOUT:   %.loc18_42.6 => constants.%.3c7
// CHECK:STDOUT:   %.loc18_42.7 => constants.%.126
// CHECK:STDOUT:   %.loc18_42.8 => constants.%.53f
// CHECK:STDOUT:   %pattern_type.loc18_42 => constants.%pattern_type.426
// CHECK:STDOUT:   %return.param_patt.loc18_42.2 => constants.%return.param_patt.894
// CHECK:STDOUT:   %return.patt.loc18_38.2 => constants.%return.patt.a56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.G.loc29(constants.%T) {
// CHECK:STDOUT:   %T.loc29_30 => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %pattern_type.loc29_30 => constants.%pattern_type.36e
// CHECK:STDOUT:   %self.param_patt.loc29_30.2 => constants.%self.param_patt.f93
// CHECK:STDOUT:   %self.patt.loc29_30.2 => constants.%self.patt.1dd
// CHECK:STDOUT:   %.loc29_40.1 => constants.%.dc4
// CHECK:STDOUT:   %.loc29_40.2 => constants.%.40e
// CHECK:STDOUT:   %.loc29_40.3 => constants.%.ac6
// CHECK:STDOUT:   %.loc29_40.4 => constants.%.86f
// CHECK:STDOUT:   %.loc29_40.5 => constants.%.e76
// CHECK:STDOUT:   %.loc29_40.6 => constants.%.3c7
// CHECK:STDOUT:   %.loc29_40.7 => constants.%.126
// CHECK:STDOUT:   %.loc29_40.8 => constants.%.53f
// CHECK:STDOUT:   %pattern_type.loc29_40 => constants.%pattern_type.426
// CHECK:STDOUT:   %return.param_patt.loc29_40.2 => constants.%return.param_patt.894
// CHECK:STDOUT:   %return.patt.loc29_36.2 => constants.%return.patt.a56
// CHECK:STDOUT: }
// CHECK:STDOUT:
