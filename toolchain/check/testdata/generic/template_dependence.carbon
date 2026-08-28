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
  // CHECK:STDERR: fail_todo_type.carbon:[[@LINE+2]]:10-11: error: value of type `<dependent type>` is not callable [CallToNonCallable]
  // CHECK:STDERR:
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
// CHECK:STDOUT:   %.f57: %ptr.e8f = splice_inst @F.%.loc8_10.5 [template]
// CHECK:STDOUT:   %.7cf: @F.%.loc8_10.8 (@F.%.loc8_10.8) = splice_inst @F.%.loc8_10.7 [template]
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
// CHECK:STDOUT:   %.loc8_10.5: <instruction> = refine_type_action %.loc8_10.2, %ptr.loc5_28.1 [template]
// CHECK:STDOUT:   %.loc8_10.6: @F.%ptr.loc5_28.1 (%ptr.e8f) = splice_inst %.loc8_10.5 [template = %.loc8_10.6 (constants.%.f57)]
// CHECK:STDOUT:   %.loc8_10.7: <instruction> = compound_member_access_action %.loc8_10.3, constants.%assoc0 [template]
// CHECK:STDOUT:   %.loc8_10.8: type = type_of_inst %.loc8_10.7 [template]
// CHECK:STDOUT:   %.loc8_10.9: @F.%.loc8_10.8 (@F.%.loc8_10.8) = splice_inst %.loc8_10.7 [template = %.loc8_10.9 (constants.%.7cf)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%ptr.loc5_29.1 (%ptr.125)) -> out %return.param: @F.%ptr.loc5_28.1 (%ptr.e8f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @F.%ptr.loc5_29.1 (%ptr.125) = name_ref x, %x
// CHECK:STDOUT:     %.loc8_10.1: ref @F.%ptr.loc5_28.1 (%ptr.e8f) = deref %x.ref
// CHECK:STDOUT:     %.loc8_10.2: @F.%ptr.loc5_28.1 (%ptr.e8f) = acquire_value %.loc8_10.1
// CHECK:STDOUT:     %.loc8_10.3: @F.%ptr.loc5_28.1 (%ptr.e8f) = splice_inst %.loc8_10.5 [template = %.loc8_10.6 (constants.%.f57)]
// CHECK:STDOUT:     %.loc8_10.4: @F.%.loc8_10.8 (@F.%.loc8_10.8) = splice_inst %.loc8_10.7 [template = %.loc8_10.9 (constants.%.7cf)]
// CHECK:STDOUT:     return <error>
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
// CHECK:STDOUT:   %return.param_patt: %pattern_type.eee = out_param_pattern [template]
// CHECK:STDOUT:   %return.patt: %pattern_type.eee = return_slot_pattern %return.param_patt, %tuple.type.a5e [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type.a5e [template]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%T, %U) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc5_16.1: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc5_33.1: %pattern_type.98f = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc5_33.2 (constants.%U.patt)]
// CHECK:STDOUT:     %return.param_patt.loc5_49.1: @F.%pattern_type (%pattern_type.eee) = out_param_pattern [template = %return.param_patt.loc5_49.2 (constants.%return.param_patt)]
// CHECK:STDOUT:     %return.patt.loc5_41.1: @F.%pattern_type (%pattern_type.eee) = return_slot_pattern %return.param_patt.loc5_49.1, %.loc5_49.4 [template = %return.patt.loc5_41.2 (constants.%return.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc5: type = name_ref T, %T.loc5_16.2 [template = %T.loc5_16.1 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc5: type = name_ref U, %U.loc5_33.2 [symbolic = %U.loc5_33.1 (constants.%U)]
// CHECK:STDOUT:     %.loc5_49.3: %tuple.type.24b = tuple_literal (%T.ref.loc5, %U.ref.loc5) [template = %tuple (constants.%tuple)]
// CHECK:STDOUT:     %.loc5_49.4: type = converted %.loc5_49.3, constants.%tuple.type.a5e [template = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:     %.loc5_49.5: Core.Form = init_form %.loc5_49.4 [template = %.loc5_49.2 (constants.%.f18)]
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
// CHECK:STDOUT:   %.loc5_49.2: Core.Form = init_form %tuple.type [template = %.loc5_49.2 (constants.%.f18)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %tuple.type [template = %pattern_type (constants.%pattern_type.eee)]
// CHECK:STDOUT:   %return.param_patt.loc5_49.2: @F.%pattern_type (%pattern_type.eee) = out_param_pattern [template = %return.param_patt.loc5_49.2 (constants.%return.param_patt)]
// CHECK:STDOUT:   %return.patt.loc5_41.2: @F.%pattern_type (%pattern_type.eee) = return_slot_pattern %return.param_patt.loc5_49.2, %tuple.type [template = %return.patt.loc5_41.2 (constants.%return.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type [template = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %F.specific_fn.loc6_10.2: <specific function> = specific_function constants.%F, @F(%T.loc5_16.1, %U.loc5_33.1) [template = %F.specific_fn.loc6_10.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @F.%tuple.type (%tuple.type.a5e) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref.loc6: type = name_ref T, %T.loc5_16.2 [template = %T.loc5_16.1 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc6: type = name_ref U, %U.loc5_33.2 [symbolic = %U.loc5_33.1 (constants.%U)]
// CHECK:STDOUT:     %F.specific_fn.loc6_10.1: <specific function> = specific_function %F.ref, @F(constants.%T, constants.%U) [template = %F.specific_fn.loc6_10.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:     %.loc5_49.1: ref @F.%tuple.type (%tuple.type.a5e) = splice_block %return.param {}
// CHECK:STDOUT:     %F.call: init @F.%tuple.type (%tuple.type.a5e) to %.loc5_49.1 = call %F.specific_fn.loc6_10.1()
// CHECK:STDOUT:     return %F.call to %return.param
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
// CHECK:STDOUT:   %.loc5_49.2 => constants.%.f18
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.eee
// CHECK:STDOUT:   %return.param_patt.loc5_49.2 => constants.%return.param_patt
// CHECK:STDOUT:   %return.patt.loc5_41.2 => constants.%return.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %F.specific_fn.loc6_10.2 => constants.%F.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
