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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.67db0b.1: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %ptr.e8f8f9.1: type = ptr_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %ptr.125: type = ptr_type %ptr.e8f8f9.1 [template]
// CHECK:STDOUT:   %pattern_type.8bb: type = pattern_type %ptr.125 [template]
// CHECK:STDOUT:   %.cb6cb9.1: Core.Form = init_form %ptr.e8f8f9.1 [template]
// CHECK:STDOUT:   %pattern_type.4f4b84.1: type = pattern_type %ptr.e8f8f9.1 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.fbe: <witness> = require_complete_type %ptr.125 [template]
// CHECK:STDOUT:   %require_complete.ef162c.1: <witness> = require_complete_type %ptr.e8f8f9.1 [template]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.1da: <witness> = lookup_impl_witness %ptr.e8f8f9.1, @Copy [template]
// CHECK:STDOUT:   %.0e9: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.67db0b.1) [template]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.e8f8f9.1, (%Copy.lookup_impl_witness.1da) [template]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.884: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [template]
// CHECK:STDOUT:   %.be5: type = fn_type_with_self_type %Copy.WithSelf.Op.type.884, %Copy.facet [template]
// CHECK:STDOUT:   %impl.elem0.484: %.be5 = impl_witness_access %Copy.lookup_impl_witness.1da, element0 [template]
// CHECK:STDOUT:   %specific_impl_fn.a76: <specific function> = specific_impl_function %impl.elem0.484, @Copy.WithSelf.Op(%Copy.facet) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0, template [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type.loc5_26 (%pattern_type.8bb) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type.loc5_26 (%pattern_type.8bb) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc5_37 (%pattern_type.4f4b84.1) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc5_37 (%pattern_type.4f4b84.1) = return_slot_pattern %return.param_patt, %ptr.loc5_37 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc5_36: type = name_ref T, %T.loc5_16.2 [template = %T.loc5_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:     %ptr.loc5_37: type = ptr_type %T.ref.loc5_36 [template = %ptr.loc5_29.1 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:     %.loc5_37.2: Core.Form = init_form %ptr.loc5_37 [template = %.loc5_37.1 (constants.%.cb6cb9.1)]
// CHECK:STDOUT:     %.loc5_19.1: type = splice_block %.loc5_19.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc5_19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_16.2: type = symbolic_binding T, 0, template [template = %T.loc5_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:     %x.param: @F.%ptr.loc5_30.1 (%ptr.125) = value_param call_param0
// CHECK:STDOUT:     %.loc5_30: type = splice_block %ptr.loc5_30.2 [template = %ptr.loc5_30.1 (constants.%ptr.125)] {
// CHECK:STDOUT:       %T.ref.loc5_28: type = name_ref T, %T.loc5_16.2 [template = %T.loc5_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:       %ptr.loc5_29.2: type = ptr_type %T.ref.loc5_28 [template = %ptr.loc5_29.1 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:       %ptr.loc5_30.2: type = ptr_type %ptr.loc5_29.2 [template = %ptr.loc5_30.1 (constants.%ptr.125)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%ptr.loc5_30.1 (%ptr.125) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr.loc5_29.1 (%ptr.e8f8f9.1) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr.loc5_29.1 (%ptr.e8f8f9.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc5_16.2: type) {
// CHECK:STDOUT:   %T.loc5_16.1: type = symbolic_binding T, 0, template [template = %T.loc5_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:   %ptr.loc5_29.1: type = ptr_type %T.loc5_16.1 [template = %ptr.loc5_29.1 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:   %ptr.loc5_30.1: type = ptr_type %ptr.loc5_29.1 [template = %ptr.loc5_30.1 (constants.%ptr.125)]
// CHECK:STDOUT:   %pattern_type.loc5_26: type = pattern_type %ptr.loc5_30.1 [template = %pattern_type.loc5_26 (constants.%pattern_type.8bb)]
// CHECK:STDOUT:   %.loc5_37.1: Core.Form = init_form %ptr.loc5_29.1 [template = %.loc5_37.1 (constants.%.cb6cb9.1)]
// CHECK:STDOUT:   %pattern_type.loc5_37: type = pattern_type %ptr.loc5_29.1 [template = %pattern_type.loc5_37 (constants.%pattern_type.4f4b84.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_26: <witness> = require_complete_type %ptr.loc5_30.1 [template = %require_complete.loc5_26 (constants.%require_complete.fbe)]
// CHECK:STDOUT:   %require_complete.loc5_33: <witness> = require_complete_type %ptr.loc5_29.1 [template = %require_complete.loc5_33 (constants.%require_complete.ef162c.1)]
// CHECK:STDOUT:   %.loc6_10.3: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.loc5_16.1) [template = %.loc6_10.3 (constants.%.0e9)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc5_29.1, @Copy [template = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.1da)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc5_29.1, (%Copy.lookup_impl_witness) [template = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [template = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.884)]
// CHECK:STDOUT:   %.loc6_10.4: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet [template = %.loc6_10.4 (constants.%.be5)]
// CHECK:STDOUT:   %impl.elem0.loc6_10.2: @F.%.loc6_10.4 (%.be5) = impl_witness_access %Copy.lookup_impl_witness, element0 [template = %impl.elem0.loc6_10.2 (constants.%impl.elem0.484)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_10.2: <specific function> = specific_impl_function %impl.elem0.loc6_10.2, @Copy.WithSelf.Op(%Copy.facet) [template = %specific_impl_fn.loc6_10.2 (constants.%specific_impl_fn.a76)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%ptr.loc5_30.1 (%ptr.125)) -> out %return.param: @F.%ptr.loc5_29.1 (%ptr.e8f8f9.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @F.%ptr.loc5_30.1 (%ptr.125) = name_ref x, %x
// CHECK:STDOUT:     %.loc6_10.1: ref @F.%ptr.loc5_29.1 (%ptr.e8f8f9.1) = deref %x.ref
// CHECK:STDOUT:     %.loc6_10.2: @F.%ptr.loc5_29.1 (%ptr.e8f8f9.1) = acquire_value %.loc6_10.1
// CHECK:STDOUT:     %impl.elem0.loc6_10.1: @F.%.loc6_10.4 (%.be5) = impl_witness_access constants.%Copy.lookup_impl_witness.1da, element0 [template = %impl.elem0.loc6_10.2 (constants.%impl.elem0.484)]
// CHECK:STDOUT:     %bound_method.loc6_10.1: <bound method> = bound_method %.loc6_10.2, %impl.elem0.loc6_10.1
// CHECK:STDOUT:     %specific_impl_fn.loc6_10.1: <specific function> = specific_impl_function %impl.elem0.loc6_10.1, @Copy.WithSelf.Op(constants.%Copy.facet) [template = %specific_impl_fn.loc6_10.2 (constants.%specific_impl_fn.a76)]
// CHECK:STDOUT:     %bound_method.loc6_10.2: <bound method> = bound_method %.loc6_10.2, %specific_impl_fn.loc6_10.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @F.%ptr.loc5_29.1 (%ptr.e8f8f9.1) = call %bound_method.loc6_10.2(%.loc6_10.2)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.67db0b.1) {
// CHECK:STDOUT:   %T.loc5_16.1 => constants.%T.67db0b.1
// CHECK:STDOUT:   %ptr.loc5_29.1 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %ptr.loc5_30.1 => constants.%ptr.125
// CHECK:STDOUT:   %pattern_type.loc5_26 => constants.%pattern_type.8bb
// CHECK:STDOUT:   %.loc5_37.1 => constants.%.cb6cb9.1
// CHECK:STDOUT:   %pattern_type.loc5_37 => constants.%pattern_type.4f4b84.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- mixed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%T, %U) [template]
// CHECK:STDOUT:   %tuple.type.a5e: type = tuple_type (%T, %U) [template]
// CHECK:STDOUT:   %.f18: Core.Form = init_form %tuple.type.a5e [template]
// CHECK:STDOUT:   %pattern_type.eee: type = pattern_type %tuple.type.a5e [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type.a5e [template]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%T, %U) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0, template [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type (%pattern_type.eee) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type (%pattern_type.eee) = return_slot_pattern %return.param_patt, %.loc5_43.4 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc5: type = name_ref T, %T.loc5_16.2 [template = %T.loc5_16.1 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc5: type = name_ref U, %U.loc5_26.2 [symbolic = %U.loc5_26.1 (constants.%U)]
// CHECK:STDOUT:     %.loc5_43.3: %tuple.type.24b = tuple_literal (%T.ref.loc5, %U.ref.loc5) [template = %tuple (constants.%tuple)]
// CHECK:STDOUT:     %.loc5_43.4: type = converted %.loc5_43.3, constants.%tuple.type.a5e [template = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:     %.loc5_43.5: Core.Form = init_form %.loc5_43.4 [template = %.loc5_43.2 (constants.%.f18)]
// CHECK:STDOUT:     %.loc5_19.1: type = splice_block %.loc5_19.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc5_16: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc5_19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_16.2: type = symbolic_binding T, 0, template [template = %T.loc5_16.1 (constants.%T)]
// CHECK:STDOUT:     %.loc5_29.1: type = splice_block %.loc5_29.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc5_26: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc5_29.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc5_26.2: type = symbolic_binding U, 1 [symbolic = %U.loc5_26.1 (constants.%U)]
// CHECK:STDOUT:     %return.param: ref @F.%tuple.type (%tuple.type.a5e) = out_param call_param0
// CHECK:STDOUT:     %return: ref @F.%tuple.type (%tuple.type.a5e) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc5_16.2: type, %U.loc5_26.2: type) {
// CHECK:STDOUT:   %T.loc5_16.1: type = symbolic_binding T, 0, template [template = %T.loc5_16.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc5_26.1: type = symbolic_binding U, 1 [symbolic = %U.loc5_26.1 (constants.%U)]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%T.loc5_16.1, %U.loc5_26.1) [template = %tuple (constants.%tuple)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%T.loc5_16.1, %U.loc5_26.1) [template = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:   %.loc5_43.2: Core.Form = init_form %tuple.type [template = %.loc5_43.2 (constants.%.f18)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %tuple.type [template = %pattern_type (constants.%pattern_type.eee)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %tuple.type [template = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %F.specific_fn.loc6_10.2: <specific function> = specific_function constants.%F, @F(%T.loc5_16.1, %U.loc5_26.1) [template = %F.specific_fn.loc6_10.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @F.%tuple.type (%tuple.type.a5e) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref.loc6: type = name_ref T, %T.loc5_16.2 [template = %T.loc5_16.1 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc6: type = name_ref U, %U.loc5_26.2 [symbolic = %U.loc5_26.1 (constants.%U)]
// CHECK:STDOUT:     %F.specific_fn.loc6_10.1: <specific function> = specific_function %F.ref, @F(constants.%T, constants.%U) [template = %F.specific_fn.loc6_10.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:     %.loc5_43.1: ref @F.%tuple.type (%tuple.type.a5e) = splice_block %return.param {}
// CHECK:STDOUT:     %F.call: init @F.%tuple.type (%tuple.type.a5e) to %.loc5_43.1 = call %F.specific_fn.loc6_10.1()
// CHECK:STDOUT:     return %F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc5_16.1 => constants.%T
// CHECK:STDOUT:   %U.loc5_26.1 => constants.%U
// CHECK:STDOUT:   %tuple => constants.%tuple
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a5e
// CHECK:STDOUT:   %.loc5_43.2 => constants.%.f18
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.eee
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %F.specific_fn.loc6_10.2 => constants.%F.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
