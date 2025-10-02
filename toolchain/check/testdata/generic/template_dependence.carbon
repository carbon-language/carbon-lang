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

// --- type.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F[template T:! type](x: T**) -> T* {
  return *x;
}
//@dump-sem-ir-end

// --- mixed.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F(template T:! type, U:! type) -> (T, U) {
  return F(T, U);
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.8b3d5d.1: type = bind_symbolic_name T, 0, template [template]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %ptr.79f131.1: type = ptr_type %T.8b3d5d.1 [template]
// CHECK:STDOUT:   %ptr.a13: type = ptr_type %ptr.79f131.1 [template]
// CHECK:STDOUT:   %pattern_type.12d: type = pattern_type %ptr.a13 [template]
// CHECK:STDOUT:   %pattern_type.afe393.1: type = pattern_type %ptr.79f131.1 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.6e5e64.1: <witness> = require_complete_type %ptr.79f131.1 [template]
// CHECK:STDOUT:   %require_complete.132: <witness> = require_complete_type %ptr.a13 [template]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.cb2: <witness> = lookup_impl_witness %ptr.79f131.1, @Copy [template]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.79f131.1, (%Copy.lookup_impl_witness.cb2) [template]
// CHECK:STDOUT:   %.1cc: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [template]
// CHECK:STDOUT:   %impl.elem0.751: %.1cc = impl_witness_access %Copy.lookup_impl_witness.cb2, element0 [template]
// CHECK:STDOUT:   %specific_impl_fn.f44: <specific function> = specific_impl_function %impl.elem0.751, @Copy.Op(%Copy.facet) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0, template [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type.loc5_25 (%pattern_type.12d) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type.loc5_25 (%pattern_type.12d) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc5_33 (%pattern_type.afe393.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc5_33 (%pattern_type.afe393.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc5_36: type = name_ref T, %T.loc5_15.2 [template = %T.loc5_15.1 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:     %ptr.loc5_37: type = ptr_type %T.ref.loc5_36 [template = %ptr.loc5_29.1 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %T.loc5_15.2: type = bind_symbolic_name T, 0, template [template = %T.loc5_15.1 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:     %x.param: @F.%ptr.loc5_30.1 (%ptr.a13) = value_param call_param0
// CHECK:STDOUT:     %.loc5: type = splice_block %ptr.loc5_30.2 [template = %ptr.loc5_30.1 (constants.%ptr.a13)] {
// CHECK:STDOUT:       %T.ref.loc5_28: type = name_ref T, %T.loc5_15.2 [template = %T.loc5_15.1 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:       %ptr.loc5_29.2: type = ptr_type %T.ref.loc5_28 [template = %ptr.loc5_29.1 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:       %ptr.loc5_30.2: type = ptr_type %ptr.loc5_29.2 [template = %ptr.loc5_30.1 (constants.%ptr.a13)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%ptr.loc5_30.1 (%ptr.a13) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr.loc5_29.1 (%ptr.79f131.1) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr.loc5_29.1 (%ptr.79f131.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc5_15.2: type) {
// CHECK:STDOUT:   %T.loc5_15.1: type = bind_symbolic_name T, 0, template [template = %T.loc5_15.1 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:   %ptr.loc5_29.1: type = ptr_type %T.loc5_15.1 [template = %ptr.loc5_29.1 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:   %ptr.loc5_30.1: type = ptr_type %ptr.loc5_29.1 [template = %ptr.loc5_30.1 (constants.%ptr.a13)]
// CHECK:STDOUT:   %pattern_type.loc5_25: type = pattern_type %ptr.loc5_30.1 [template = %pattern_type.loc5_25 (constants.%pattern_type.12d)]
// CHECK:STDOUT:   %pattern_type.loc5_33: type = pattern_type %ptr.loc5_29.1 [template = %pattern_type.loc5_33 (constants.%pattern_type.afe393.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_33: <witness> = require_complete_type %ptr.loc5_29.1 [template = %require_complete.loc5_33 (constants.%require_complete.6e5e64.1)]
// CHECK:STDOUT:   %require_complete.loc5_26: <witness> = require_complete_type %ptr.loc5_30.1 [template = %require_complete.loc5_26 (constants.%require_complete.132)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc5_29.1, @Copy [template = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.cb2)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc5_29.1, (%Copy.lookup_impl_witness) [template = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %.loc6_10.4: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [template = %.loc6_10.4 (constants.%.1cc)]
// CHECK:STDOUT:   %impl.elem0.loc6_10.2: @F.%.loc6_10.4 (%.1cc) = impl_witness_access %Copy.lookup_impl_witness, element0 [template = %impl.elem0.loc6_10.2 (constants.%impl.elem0.751)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_10.2: <specific function> = specific_impl_function %impl.elem0.loc6_10.2, @Copy.Op(%Copy.facet) [template = %specific_impl_fn.loc6_10.2 (constants.%specific_impl_fn.f44)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%ptr.loc5_30.1 (%ptr.a13)) -> @F.%ptr.loc5_29.1 (%ptr.79f131.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @F.%ptr.loc5_30.1 (%ptr.a13) = name_ref x, %x
// CHECK:STDOUT:     %.loc6_10.1: ref @F.%ptr.loc5_29.1 (%ptr.79f131.1) = deref %x.ref
// CHECK:STDOUT:     %.loc6_10.2: @F.%ptr.loc5_29.1 (%ptr.79f131.1) = bind_value %.loc6_10.1
// CHECK:STDOUT:     %impl.elem0.loc6_10.1: @F.%.loc6_10.4 (%.1cc) = impl_witness_access constants.%Copy.lookup_impl_witness.cb2, element0 [template = %impl.elem0.loc6_10.2 (constants.%impl.elem0.751)]
// CHECK:STDOUT:     %bound_method.loc6_10.1: <bound method> = bound_method %.loc6_10.2, %impl.elem0.loc6_10.1
// CHECK:STDOUT:     %specific_impl_fn.loc6_10.1: <specific function> = specific_impl_function %impl.elem0.loc6_10.1, @Copy.Op(constants.%Copy.facet) [template = %specific_impl_fn.loc6_10.2 (constants.%specific_impl_fn.f44)]
// CHECK:STDOUT:     %bound_method.loc6_10.2: <bound method> = bound_method %.loc6_10.2, %specific_impl_fn.loc6_10.1
// CHECK:STDOUT:     %.loc6_10.3: init @F.%ptr.loc5_29.1 (%ptr.79f131.1) = call %bound_method.loc6_10.2(%.loc6_10.2)
// CHECK:STDOUT:     return %.loc6_10.3 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.8b3d5d.1) {
// CHECK:STDOUT:   %T.loc5_15.1 => constants.%T.8b3d5d.1
// CHECK:STDOUT:   %ptr.loc5_29.1 => constants.%ptr.79f131.1
// CHECK:STDOUT:   %ptr.loc5_30.1 => constants.%ptr.a13
// CHECK:STDOUT:   %pattern_type.loc5_25 => constants.%pattern_type.12d
// CHECK:STDOUT:   %pattern_type.loc5_33 => constants.%pattern_type.afe393.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- mixed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0, template [template]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.30b: type = tuple_type (%T, %U) [template]
// CHECK:STDOUT:   %pattern_type.65c: type = pattern_type %tuple.type.30b [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type.30b [template]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%T, %U) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0, template [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type (%pattern_type.65c) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type (%pattern_type.65c) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc5: type = name_ref T, %T.loc5_15.2 [template = %T.loc5_15.1 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc5: type = name_ref U, %U.loc5_25.2 [symbolic = %U.loc5_25.1 (constants.%U)]
// CHECK:STDOUT:     %.loc5_43.1: %tuple.type.24b = tuple_literal (%T.ref.loc5, %U.ref.loc5)
// CHECK:STDOUT:     %.loc5_43.2: type = converted %.loc5_43.1, constants.%tuple.type.30b [template = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %T.loc5_15.2: type = bind_symbolic_name T, 0, template [template = %T.loc5_15.1 (constants.%T)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %U.loc5_25.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc5_25.1 (constants.%U)]
// CHECK:STDOUT:     %return.param: ref @F.%tuple.type (%tuple.type.30b) = out_param call_param0
// CHECK:STDOUT:     %return: ref @F.%tuple.type (%tuple.type.30b) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc5_15.2: type, %U.loc5_25.2: type) {
// CHECK:STDOUT:   %T.loc5_15.1: type = bind_symbolic_name T, 0, template [template = %T.loc5_15.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc5_25.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc5_25.1 (constants.%U)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%T.loc5_15.1, %U.loc5_25.1) [template = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %tuple.type [template = %pattern_type (constants.%pattern_type.65c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type [template = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %F.specific_fn.loc6_10.2: <specific function> = specific_function constants.%F, @F(%T.loc5_15.1, %U.loc5_25.1) [template = %F.specific_fn.loc6_10.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return.param: @F.%tuple.type (%tuple.type.30b) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref.loc6: type = name_ref T, %T.loc5_15.2 [template = %T.loc5_15.1 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc6: type = name_ref U, %U.loc5_25.2 [symbolic = %U.loc5_25.1 (constants.%U)]
// CHECK:STDOUT:     %F.specific_fn.loc6_10.1: <specific function> = specific_function %F.ref, @F(constants.%T, constants.%U) [template = %F.specific_fn.loc6_10.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:     %.loc5_35: ref @F.%tuple.type (%tuple.type.30b) = splice_block %return {}
// CHECK:STDOUT:     %F.call: init @F.%tuple.type (%tuple.type.30b) = call %F.specific_fn.loc6_10.1() to %.loc5_35
// CHECK:STDOUT:     return %F.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc5_15.1 => constants.%T
// CHECK:STDOUT:   %U.loc5_25.1 => constants.%U
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.30b
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.65c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %F.specific_fn.loc6_10.2 => constants.%F.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
