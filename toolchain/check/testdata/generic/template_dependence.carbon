// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/template_dependence.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/template_dependence.carbon

// --- fail_todo_type.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F[template T: type](x: T**) -> T* {
  return *x;
}
//@dump-sem-ir-end

// --- mixed.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F(template T: type, generic U: type) -> (T, U) {
  return F(T, U);
}
//@dump-sem-ir-end


// CHECK:STDOUT: --- fail_todo_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T [template]
// CHECK:STDOUT:   %ptr.125: type = ptr_type %ptr.e8f [template]
// CHECK:STDOUT:   %pattern_type.8bb: type = pattern_type %ptr.125 [template]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.8bb = value_param_pattern [template]
// CHECK:STDOUT:   %x.patt: %pattern_type.8bb = wrapper_binding_pattern x, %x.param_patt [template]
// CHECK:STDOUT:   %.cb6: Core.Form = init_form %ptr.e8f [template]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f [template]
// CHECK:STDOUT:   %return.param_patt.27f: %pattern_type.4f4 = out_param_pattern [template]
// CHECK:STDOUT:   %return.patt.d67: %pattern_type.4f4 = return_slot_pattern %return.param_patt.27f, %ptr.e8f [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.fbe: <witness> = require_complete_type %ptr.125 [template]
// CHECK:STDOUT:   %require_complete.ef1: <witness> = require_complete_type %ptr.e8f [template]
// CHECK:STDOUT:   %Copy.assoc_type: type = assoc_entity_type @Copy [concrete]
// CHECK:STDOUT:   %assoc0: %Copy.assoc_type = assoc_entity element0, imports.%Core.import_ref.098 [concrete]
// CHECK:STDOUT:   %.f57: %ptr.e8f = splice_inst @F.%.loc6_10.6 [template]
// CHECK:STDOUT:   %.7cf: @F.%.loc6_10.9 (@F.%.loc6_10.9) = splice_inst @F.%.loc6_10.8 [template]
// CHECK:STDOUT:   %.c15: @F.%.loc6_10.12 (@F.%.loc6_10.12) = splice_inst @F.%.loc6_10.11 [template]
// CHECK:STDOUT:   %.168: @F.%.loc6_10.12 (@F.%.loc6_10.12) = splice_inst @F.%.loc6_12.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.098 = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc5_16.1: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.param_patt.loc5_25.1: @F.%pattern_type.loc5_25 (%pattern_type.8bb) = value_param_pattern [template = %x.param_patt.loc5_25.2 (constants.%x.param_patt)]
// CHECK:STDOUT:     %x.patt.loc5_25.1: @F.%pattern_type.loc5_25 (%pattern_type.8bb) = wrapper_binding_pattern x, %x.param_patt.loc5_25.1 [template = %x.patt.loc5_25.2 (constants.%x.patt)]
// CHECK:STDOUT:     %return.param_patt.loc5_36.1: @F.%pattern_type.loc5_36 (%pattern_type.4f4) = out_param_pattern [template = %return.param_patt.loc5_36.2 (constants.%return.param_patt.27f)]
// CHECK:STDOUT:     %return.patt.loc5_32.1: @F.%pattern_type.loc5_36 (%pattern_type.4f4) = return_slot_pattern %return.param_patt.loc5_36.1, %ptr.loc5_36 [template = %return.patt.loc5_32.2 (constants.%return.patt.d67)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc5_35: type = name_ref T, %T.loc5_16.2 [template = %T.loc5_16.1 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc5_36: type = ptr_type %T.ref.loc5_35 [template = %ptr.loc5_28.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:     %.loc5_36.2: Core.Form = init_form %ptr.loc5_36 [template = %.loc5_36.1 (constants.%.cb6)]
// CHECK:STDOUT:     %.loc5_18.1: type = splice_block %.loc5_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc5_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_16.2: type = symbolic_binding T, 0, template [template = %T.loc5_16.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%ptr.loc5_29.1 (%ptr.125) = value_param call_param0
// CHECK:STDOUT:     %.loc5_29: type = splice_block %ptr.loc5_29.2 [template = %ptr.loc5_29.1 (constants.%ptr.125)] {
// CHECK:STDOUT:       %T.ref.loc5_27: type = name_ref T, %T.loc5_16.2 [template = %T.loc5_16.1 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc5_28.2: type = ptr_type %T.ref.loc5_27 [template = %ptr.loc5_28.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:       %ptr.loc5_29.2: type = ptr_type %ptr.loc5_28.2 [template = %ptr.loc5_29.1 (constants.%ptr.125)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%ptr.loc5_29.1 (%ptr.125) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr.loc5_28.1 (%ptr.e8f) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr.loc5_28.1 (%ptr.e8f) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc5_16.2: type) {
// CHECK:STDOUT:   %T.patt.loc5_16.2: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc5_16.1: type = symbolic_binding T, 0, template [template = %T.loc5_16.1 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc5_28.1: type = ptr_type %T.loc5_16.1 [template = %ptr.loc5_28.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:   %ptr.loc5_29.1: type = ptr_type %ptr.loc5_28.1 [template = %ptr.loc5_29.1 (constants.%ptr.125)]
// CHECK:STDOUT:   %pattern_type.loc5_25: type = pattern_type %ptr.loc5_29.1 [template = %pattern_type.loc5_25 (constants.%pattern_type.8bb)]
// CHECK:STDOUT:   %x.param_patt.loc5_25.2: @F.%pattern_type.loc5_25 (%pattern_type.8bb) = value_param_pattern [template = %x.param_patt.loc5_25.2 (constants.%x.param_patt)]
// CHECK:STDOUT:   %x.patt.loc5_25.2: @F.%pattern_type.loc5_25 (%pattern_type.8bb) = wrapper_binding_pattern x, %x.param_patt.loc5_25.2 [template = %x.patt.loc5_25.2 (constants.%x.patt)]
// CHECK:STDOUT:   %.loc5_36.1: Core.Form = init_form %ptr.loc5_28.1 [template = %.loc5_36.1 (constants.%.cb6)]
// CHECK:STDOUT:   %pattern_type.loc5_36: type = pattern_type %ptr.loc5_28.1 [template = %pattern_type.loc5_36 (constants.%pattern_type.4f4)]
// CHECK:STDOUT:   %return.param_patt.loc5_36.2: @F.%pattern_type.loc5_36 (%pattern_type.4f4) = out_param_pattern [template = %return.param_patt.loc5_36.2 (constants.%return.param_patt.27f)]
// CHECK:STDOUT:   %return.patt.loc5_32.2: @F.%pattern_type.loc5_36 (%pattern_type.4f4) = return_slot_pattern %return.param_patt.loc5_36.2, %ptr.loc5_28.1 [template = %return.patt.loc5_32.2 (constants.%return.patt.d67)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_25: <witness> = require_complete_type %ptr.loc5_29.1 [template = %require_complete.loc5_25 (constants.%require_complete.fbe)]
// CHECK:STDOUT:   %require_complete.loc5_32: <witness> = require_complete_type %ptr.loc5_28.1 [template = %require_complete.loc5_32 (constants.%require_complete.ef1)]
// CHECK:STDOUT:   %.loc6_10.6: <instruction> = refine_type_action %.loc6_10.2, %ptr.loc5_28.1 [template]
// CHECK:STDOUT:   %.loc6_10.7: @F.%ptr.loc5_28.1 (%ptr.e8f) = splice_inst %.loc6_10.6 [template = %.loc6_10.7 (constants.%.f57)]
// CHECK:STDOUT:   %.loc6_10.8: <instruction> = compound_member_access_action %.loc6_10.3, constants.%assoc0 [template]
// CHECK:STDOUT:   %.loc6_10.9: type = type_of_inst %.loc6_10.8 [template]
// CHECK:STDOUT:   %.loc6_10.10: @F.%.loc6_10.9 (@F.%.loc6_10.9) = splice_inst %.loc6_10.8 [template = %.loc6_10.10 (constants.%.7cf)]
// CHECK:STDOUT:   %.loc6_10.11: <instruction> = call_action %.loc6_10.4, (), true [template]
// CHECK:STDOUT:   %.loc6_10.12: type = type_of_inst %.loc6_10.11 [template]
// CHECK:STDOUT:   %.loc6_10.13: @F.%.loc6_10.12 (@F.%.loc6_10.12) = splice_inst %.loc6_10.11 [template = %.loc6_10.13 (constants.%.c15)]
// CHECK:STDOUT:   %.loc6_12.2: <instruction> = convert_to_category_action %.loc6_10.5, element10 [template]
// CHECK:STDOUT:   %.loc6_12.3: @F.%.loc6_10.12 (@F.%.loc6_10.12) = splice_inst %.loc6_12.2 [template = %.loc6_12.3 (constants.%.168)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%ptr.loc5_29.1 (%ptr.125)) -> out %return.param: @F.%ptr.loc5_28.1 (%ptr.e8f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @F.%ptr.loc5_29.1 (%ptr.125) = name_ref x, %x
// CHECK:STDOUT:     %.loc6_10.1: ref @F.%ptr.loc5_28.1 (%ptr.e8f) = deref %x.ref
// CHECK:STDOUT:     %.loc6_10.2: @F.%ptr.loc5_28.1 (%ptr.e8f) = acquire_value %.loc6_10.1
// CHECK:STDOUT:     %.loc6_10.3: @F.%ptr.loc5_28.1 (%ptr.e8f) = splice_inst %.loc6_10.6 [template = %.loc6_10.7 (constants.%.f57)]
// CHECK:STDOUT:     %.loc6_10.4: @F.%.loc6_10.9 (@F.%.loc6_10.9) = splice_inst %.loc6_10.8 [template = %.loc6_10.10 (constants.%.7cf)]
// CHECK:STDOUT:     %.loc6_10.5: @F.%.loc6_10.12 (@F.%.loc6_10.12) = splice_inst %.loc6_10.11 [template = %.loc6_10.13 (constants.%.c15)]
// CHECK:STDOUT:     %.loc6_12.1: @F.%.loc6_10.12 (@F.%.loc6_10.12) = splice_inst %.loc6_12.2 [template = %.loc6_12.3 (constants.%.168)]
// CHECK:STDOUT:     return %.loc6_12.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc5_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc5_16.1 => constants.%T
// CHECK:STDOUT:   %ptr.loc5_28.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %ptr.loc5_29.1 => constants.%ptr.125
// CHECK:STDOUT:   %pattern_type.loc5_25 => constants.%pattern_type.8bb
// CHECK:STDOUT:   %x.param_patt.loc5_25.2 => constants.%x.param_patt
// CHECK:STDOUT:   %x.patt.loc5_25.2 => constants.%x.patt
// CHECK:STDOUT:   %.loc5_36.1 => constants.%.cb6
// CHECK:STDOUT:   %pattern_type.loc5_36 => constants.%pattern_type.4f4
// CHECK:STDOUT:   %return.param_patt.loc5_36.2 => constants.%return.param_patt.27f
// CHECK:STDOUT:   %return.patt.loc5_32.2 => constants.%return.patt.d67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- mixed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%T, %U) [template]
// CHECK:STDOUT:   %tuple.type.a5e: type = tuple_type (%T, %U) [template]
// CHECK:STDOUT:   %.f18: Core.Form = init_form %tuple.type.a5e [template]
// CHECK:STDOUT:   %pattern_type.eee: type = pattern_type %tuple.type.a5e [template]
// CHECK:STDOUT:   %return.param_patt.a41: %pattern_type.eee = out_param_pattern [template]
// CHECK:STDOUT:   %return.patt.453: %pattern_type.eee = return_slot_pattern %return.param_patt.a41, %tuple.type.a5e [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type.a5e [template]
// CHECK:STDOUT:   %.27a: @F.%.loc6_16.3 (@F.%.loc6_16.3) = splice_inst @F.%.loc6_16.2 [template]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %.577: @F.%.loc6_17.9 (@F.%.loc6_17.9) = splice_inst @F.%.loc6_17.8 [template]
// CHECK:STDOUT:   %.32b: @F.%.loc6_17.12 (@F.%.loc6_17.12) = splice_inst @F.%.loc6_17.11 [template]
// CHECK:STDOUT:   %.88a: @F.%.loc6_17.15 (@F.%.loc6_17.15) = splice_inst @F.%.loc6_17.14 [template]
// CHECK:STDOUT:   %.f68: @F.%.loc6_17.18 (@F.%.loc6_17.18) = splice_inst @F.%.loc6_17.17 [template]
// CHECK:STDOUT:   %.e9d: %tuple.type.a5e = splice_inst @F.%.loc6_17.20 [template]
// CHECK:STDOUT:   %.70a: %tuple.type.a5e = splice_inst @F.%.loc6_17.22 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc5_16.1: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc5_33.1: %pattern_type.98f = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc5_33.2 (constants.%U.patt)]
// CHECK:STDOUT:     %return.param_patt.loc5_49.1: @F.%pattern_type (%pattern_type.eee) = out_param_pattern [template = %return.param_patt.loc5_49.2 (constants.%return.param_patt.a41)]
// CHECK:STDOUT:     %return.patt.loc5_41.1: @F.%pattern_type (%pattern_type.eee) = return_slot_pattern %return.param_patt.loc5_49.1, %.loc5_49.3 [template = %return.patt.loc5_41.2 (constants.%return.patt.453)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc5: type = name_ref T, %T.loc5_16.2 [template = %T.loc5_16.1 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc5: type = name_ref U, %U.loc5_33.2 [symbolic = %U.loc5_33.1 (constants.%U)]
// CHECK:STDOUT:     %.loc5_49.2: %tuple.type.24b = tuple_literal (%T.ref.loc5, %U.ref.loc5) [template = %tuple (constants.%tuple)]
// CHECK:STDOUT:     %.loc5_49.3: type = converted %.loc5_49.2, constants.%tuple.type.a5e [template = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:     %.loc5_49.4: Core.Form = init_form %.loc5_49.3 [template = %.loc5_49.1 (constants.%.f18)]
// CHECK:STDOUT:     %.loc5_18.1: type = splice_block %.loc5_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc5_16: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc5_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_16.2: type = symbolic_binding T, 0, template [template = %T.loc5_16.1 (constants.%T)]
// CHECK:STDOUT:     %.loc5_35.1: type = splice_block %.loc5_35.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc5_33: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc5_35.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc5_33.2: type = symbolic_binding U, 1 [symbolic = %U.loc5_33.1 (constants.%U)]
// CHECK:STDOUT:     %return.param: ref @F.%tuple.type (%tuple.type.a5e) = out_param call_param0
// CHECK:STDOUT:     %return: ref @F.%tuple.type (%tuple.type.a5e) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc5_16.2: type, %U.loc5_33.2: type) {
// CHECK:STDOUT:   %T.patt.loc5_16.2: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc5_16.1: type = symbolic_binding T, 0, template [template = %T.loc5_16.1 (constants.%T)]
// CHECK:STDOUT:   %U.patt.loc5_33.2: %pattern_type.98f = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc5_33.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc5_33.1: type = symbolic_binding U, 1 [symbolic = %U.loc5_33.1 (constants.%U)]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%T.loc5_16.1, %U.loc5_33.1) [template = %tuple (constants.%tuple)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%T.loc5_16.1, %U.loc5_33.1) [template = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:   %.loc5_49.1: Core.Form = init_form %tuple.type [template = %.loc5_49.1 (constants.%.f18)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %tuple.type [template = %pattern_type (constants.%pattern_type.eee)]
// CHECK:STDOUT:   %return.param_patt.loc5_49.2: @F.%pattern_type (%pattern_type.eee) = out_param_pattern [template = %return.param_patt.loc5_49.2 (constants.%return.param_patt.a41)]
// CHECK:STDOUT:   %return.patt.loc5_41.2: @F.%pattern_type (%pattern_type.eee) = return_slot_pattern %return.param_patt.loc5_49.2, %tuple.type [template = %return.patt.loc5_41.2 (constants.%return.patt.453)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type [template = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %.loc6_16.2: <instruction> = call_action %F.ref, (%T.ref.loc6, %U.ref.loc6), false [template]
// CHECK:STDOUT:   %.loc6_16.3: type = type_of_inst %.loc6_16.2 [template]
// CHECK:STDOUT:   %.loc6_16.4: @F.%.loc6_16.3 (@F.%.loc6_16.3) = splice_inst %.loc6_16.2 [template = %.loc6_16.4 (constants.%.27a)]
// CHECK:STDOUT:   %.loc6_17.8: <instruction> = call_action constants.%ImplicitAs.generic, (constants.%tuple.type.a5e), false [template]
// CHECK:STDOUT:   %.loc6_17.9: type = type_of_inst %.loc6_17.8 [template]
// CHECK:STDOUT:   %.loc6_17.10: @F.%.loc6_17.9 (@F.%.loc6_17.9) = splice_inst %.loc6_17.8 [template = %.loc6_17.10 (constants.%.577)]
// CHECK:STDOUT:   %.loc6_17.11: <instruction> = access_member_action %.loc6_17.1, Convert [template]
// CHECK:STDOUT:   %.loc6_17.12: type = type_of_inst %.loc6_17.11 [template]
// CHECK:STDOUT:   %.loc6_17.13: @F.%.loc6_17.12 (@F.%.loc6_17.12) = splice_inst %.loc6_17.11 [template = %.loc6_17.13 (constants.%.32b)]
// CHECK:STDOUT:   %.loc6_17.14: <instruction> = compound_member_access_action %.loc6_16.1, %.loc6_17.2 [template]
// CHECK:STDOUT:   %.loc6_17.15: type = type_of_inst %.loc6_17.14 [template]
// CHECK:STDOUT:   %.loc6_17.16: @F.%.loc6_17.15 (@F.%.loc6_17.15) = splice_inst %.loc6_17.14 [template = %.loc6_17.16 (constants.%.88a)]
// CHECK:STDOUT:   %.loc6_17.17: <instruction> = call_action %.loc6_17.3, (), true [template]
// CHECK:STDOUT:   %.loc6_17.18: type = type_of_inst %.loc6_17.17 [template]
// CHECK:STDOUT:   %.loc6_17.19: @F.%.loc6_17.18 (@F.%.loc6_17.18) = splice_inst %.loc6_17.17 [template = %.loc6_17.19 (constants.%.f68)]
// CHECK:STDOUT:   %.loc6_17.20: <instruction> = refine_type_action %.loc6_17.5, %tuple.type [template]
// CHECK:STDOUT:   %.loc6_17.21: @F.%tuple.type (%tuple.type.a5e) = splice_inst %.loc6_17.20 [template = %.loc6_17.21 (constants.%.e9d)]
// CHECK:STDOUT:   %.loc6_17.22: <instruction> = convert_to_category_action %.loc6_17.6, element10 [template]
// CHECK:STDOUT:   %.loc6_17.23: @F.%tuple.type (%tuple.type.a5e) = splice_inst %.loc6_17.22 [template = %.loc6_17.23 (constants.%.70a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @F.%tuple.type (%tuple.type.a5e) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref.loc6: type = name_ref T, %T.loc5_16.2 [template = %T.loc5_16.1 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc6: type = name_ref U, %U.loc5_33.2 [symbolic = %U.loc5_33.1 (constants.%U)]
// CHECK:STDOUT:     %.loc6_16.1: @F.%.loc6_16.3 (@F.%.loc6_16.3) = splice_inst %.loc6_16.2 [template = %.loc6_16.4 (constants.%.27a)]
// CHECK:STDOUT:     %.loc6_17.1: @F.%.loc6_17.9 (@F.%.loc6_17.9) = splice_inst %.loc6_17.8 [template = %.loc6_17.10 (constants.%.577)]
// CHECK:STDOUT:     %.loc6_17.2: @F.%.loc6_17.12 (@F.%.loc6_17.12) = splice_inst %.loc6_17.11 [template = %.loc6_17.13 (constants.%.32b)]
// CHECK:STDOUT:     %.loc6_17.3: @F.%.loc6_17.15 (@F.%.loc6_17.15) = splice_inst %.loc6_17.14 [template = %.loc6_17.16 (constants.%.88a)]
// CHECK:STDOUT:     %.loc6_17.4: @F.%.loc6_17.18 (@F.%.loc6_17.18) = splice_inst %.loc6_17.17 [template = %.loc6_17.19 (constants.%.f68)]
// CHECK:STDOUT:     %.loc6_17.5: @F.%tuple.type (%tuple.type.a5e) = converted %.loc6_16.1, %.loc6_17.4 [template = %.loc6_17.19 (constants.%.f68)]
// CHECK:STDOUT:     %.loc6_17.6: @F.%tuple.type (%tuple.type.a5e) = splice_inst %.loc6_17.20 [template = %.loc6_17.21 (constants.%.e9d)]
// CHECK:STDOUT:     %.loc6_17.7: @F.%tuple.type (%tuple.type.a5e) = splice_inst %.loc6_17.22 [template = %.loc6_17.23 (constants.%.70a)]
// CHECK:STDOUT:     return %.loc6_17.7 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.patt.loc5_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc5_16.1 => constants.%T
// CHECK:STDOUT:   %U.patt.loc5_33.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc5_33.1 => constants.%U
// CHECK:STDOUT:   %tuple => constants.%tuple
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a5e
// CHECK:STDOUT:   %.loc5_49.1 => constants.%.f18
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.eee
// CHECK:STDOUT:   %return.param_patt.loc5_49.2 => constants.%return.param_patt.a41
// CHECK:STDOUT:   %return.patt.loc5_41.2 => constants.%return.patt.453
// CHECK:STDOUT: }
// CHECK:STDOUT:
