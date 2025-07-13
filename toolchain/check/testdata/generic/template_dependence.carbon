// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/template_dependence.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/template_dependence.carbon

// --- type.carbon

library "[[@TEST_NAME]]";

fn F[template T:! type](x: T**) -> T* {
  return *x;
}

// --- mixed.carbon

library "[[@TEST_NAME]]";

fn F(template T:! type, U:! type) -> (T, U) {
  return F(T, U);
}

// CHECK:STDOUT: --- type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0, template [template]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [template]
// CHECK:STDOUT:   %ptr.a13: type = ptr_type %ptr.79f [template]
// CHECK:STDOUT:   %pattern_type.12d: type = pattern_type %ptr.a13 [template]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %ptr.79f [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [template]
// CHECK:STDOUT:   %require_complete.132: <witness> = require_complete_type %ptr.a13 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0, template [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type.loc4_25 (%pattern_type.12d) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type.loc4_25 (%pattern_type.12d) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc4_33 (%pattern_type.afe) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc4_33 (%pattern_type.afe) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4_36: type = name_ref T, %T.loc4_15.1 [template = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc4_37: type = ptr_type %T.ref.loc4_36 [template = %ptr.loc4_29.2 (constants.%ptr.79f)]
// CHECK:STDOUT:     %T.loc4_15.1: type = bind_symbolic_name T, 0, template [template = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%ptr.loc4_30.2 (%ptr.a13) = value_param call_param0
// CHECK:STDOUT:     %.loc4: type = splice_block %ptr.loc4_30.1 [template = %ptr.loc4_30.2 (constants.%ptr.a13)] {
// CHECK:STDOUT:       %T.ref.loc4_28: type = name_ref T, %T.loc4_15.1 [template = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc4_29.1: type = ptr_type %T.ref.loc4_28 [template = %ptr.loc4_29.2 (constants.%ptr.79f)]
// CHECK:STDOUT:       %ptr.loc4_30.1: type = ptr_type %ptr.loc4_29.1 [template = %ptr.loc4_30.2 (constants.%ptr.a13)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%ptr.loc4_30.2 (%ptr.a13) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr.loc4_29.2 (%ptr.79f) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr.loc4_29.2 (%ptr.79f) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_15.1: type) {
// CHECK:STDOUT:   %T.loc4_15.2: type = bind_symbolic_name T, 0, template [template = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc4_29.2: type = ptr_type %T.loc4_15.2 [template = %ptr.loc4_29.2 (constants.%ptr.79f)]
// CHECK:STDOUT:   %ptr.loc4_30.2: type = ptr_type %ptr.loc4_29.2 [template = %ptr.loc4_30.2 (constants.%ptr.a13)]
// CHECK:STDOUT:   %pattern_type.loc4_25: type = pattern_type %ptr.loc4_30.2 [template = %pattern_type.loc4_25 (constants.%pattern_type.12d)]
// CHECK:STDOUT:   %pattern_type.loc4_33: type = pattern_type %ptr.loc4_29.2 [template = %pattern_type.loc4_33 (constants.%pattern_type.afe)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc4_33: <witness> = require_complete_type %ptr.loc4_29.2 [template = %require_complete.loc4_33 (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %require_complete.loc4_26: <witness> = require_complete_type %ptr.loc4_30.2 [template = %require_complete.loc4_26 (constants.%require_complete.132)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%ptr.loc4_30.2 (%ptr.a13)) -> @F.%ptr.loc4_29.2 (%ptr.79f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @F.%ptr.loc4_30.2 (%ptr.a13) = name_ref x, %x
// CHECK:STDOUT:     %.loc5_10.1: ref @F.%ptr.loc4_29.2 (%ptr.79f) = deref %x.ref
// CHECK:STDOUT:     %.loc5_10.2: @F.%ptr.loc4_29.2 (%ptr.79f) = bind_value %.loc5_10.1
// CHECK:STDOUT:     return %.loc5_10.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%T
// CHECK:STDOUT:   %ptr.loc4_29.2 => constants.%ptr.79f
// CHECK:STDOUT:   %ptr.loc4_30.2 => constants.%ptr.a13
// CHECK:STDOUT:   %pattern_type.loc4_25 => constants.%pattern_type.12d
// CHECK:STDOUT:   %pattern_type.loc4_33 => constants.%pattern_type.afe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- mixed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.8b3d5d.1: type = bind_symbolic_name T, 0, template [template]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.30b: type = tuple_type (%T.8b3d5d.1, %U) [template]
// CHECK:STDOUT:   %pattern_type.65c: type = pattern_type %tuple.type.30b [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.937: type = ptr_type %tuple.type.30b [template]
// CHECK:STDOUT:   %require_complete.fe1: <witness> = require_complete_type %tuple.type.30b [template]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%T.8b3d5d.1, %U) [template]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %require_complete.8fa: <witness> = require_complete_type %ptr.937 [template]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type.30b, @Destroy [template]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %tuple.type.30b, (%Destroy.lookup_impl_witness) [template]
// CHECK:STDOUT:   %.1e9: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet [template]
// CHECK:STDOUT:   %impl.elem0: %.1e9 = impl_witness_access %Destroy.lookup_impl_witness, element0 [template]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Op.1(%Destroy.facet) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0, template [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type (%pattern_type.65c) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type (%pattern_type.65c) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4: type = name_ref T, %T.loc4_15.1 [template = %T.loc4_15.2 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:     %U.ref.loc4: type = name_ref U, %U.loc4_25.1 [symbolic = %U.loc4_25.2 (constants.%U)]
// CHECK:STDOUT:     %.loc4_43.1: %tuple.type.24b = tuple_literal (%T.ref.loc4, %U.ref.loc4)
// CHECK:STDOUT:     %.loc4_43.2: type = converted %.loc4_43.1, constants.%tuple.type.30b [template = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:     %T.loc4_15.1: type = bind_symbolic_name T, 0, template [template = %T.loc4_15.2 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:     %U.loc4_25.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc4_25.2 (constants.%U)]
// CHECK:STDOUT:     %return.param: ref @F.%tuple.type (%tuple.type.30b) = out_param call_param0
// CHECK:STDOUT:     %return: ref @F.%tuple.type (%tuple.type.30b) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_15.1: type, %U.loc4_25.1: type) {
// CHECK:STDOUT:   %T.loc4_15.2: type = bind_symbolic_name T, 0, template [template = %T.loc4_15.2 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:   %U.loc4_25.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc4_25.2 (constants.%U)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%T.loc4_15.2, %U.loc4_25.2) [template = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %tuple.type [template = %pattern_type (constants.%pattern_type.65c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc4_35.1: <witness> = require_complete_type %tuple.type [template = %require_complete.loc4_35.1 (constants.%require_complete.fe1)]
// CHECK:STDOUT:   %F.specific_fn.loc5_10.2: <specific function> = specific_function constants.%F, @F(%T.loc4_15.2, %U.loc4_25.2) [template = %F.specific_fn.loc5_10.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %tuple.type, @Destroy [template = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %tuple.type, (%Destroy.lookup_impl_witness) [template = %Destroy.facet (constants.%Destroy.facet)]
// CHECK:STDOUT:   %.loc4_35.3: type = fn_type_with_self_type constants.%Op.type.bae, %Destroy.facet [template = %.loc4_35.3 (constants.%.1e9)]
// CHECK:STDOUT:   %impl.elem0.loc4_35.2: @F.%.loc4_35.3 (%.1e9) = impl_witness_access %Destroy.lookup_impl_witness, element0 [template = %impl.elem0.loc4_35.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc4_35.2: <specific function> = specific_impl_function %impl.elem0.loc4_35.2, @Op.1(%Destroy.facet) [template = %specific_impl_fn.loc4_35.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:   %ptr: type = ptr_type %tuple.type [template = %ptr (constants.%ptr.937)]
// CHECK:STDOUT:   %require_complete.loc4_35.2: <witness> = require_complete_type %ptr [template = %require_complete.loc4_35.2 (constants.%require_complete.8fa)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return.param: @F.%tuple.type (%tuple.type.30b) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref.loc5: type = name_ref T, %T.loc4_15.1 [template = %T.loc4_15.2 (constants.%T.8b3d5d.1)]
// CHECK:STDOUT:     %U.ref.loc5: type = name_ref U, %U.loc4_25.1 [symbolic = %U.loc4_25.2 (constants.%U)]
// CHECK:STDOUT:     %F.specific_fn.loc5_10.1: <specific function> = specific_function %F.ref, @F(constants.%T.8b3d5d.1, constants.%U) [template = %F.specific_fn.loc5_10.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:     %.loc4_35.1: ref @F.%tuple.type (%tuple.type.30b) = splice_block %return {}
// CHECK:STDOUT:     %F.call: init @F.%tuple.type (%tuple.type.30b) = call %F.specific_fn.loc5_10.1() to %.loc4_35.1
// CHECK:STDOUT:     %impl.elem0.loc4_35.1: @F.%.loc4_35.3 (%.1e9) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [template = %impl.elem0.loc4_35.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc4_35.1: <bound method> = bound_method %.loc4_35.1, %impl.elem0.loc4_35.1
// CHECK:STDOUT:     %specific_impl_fn.loc4_35.1: <specific function> = specific_impl_function %impl.elem0.loc4_35.1, @Op.1(constants.%Destroy.facet) [template = %specific_impl_fn.loc4_35.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc4_35.2: <bound method> = bound_method %.loc4_35.1, %specific_impl_fn.loc4_35.1
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.937) = addr_of %.loc4_35.1
// CHECK:STDOUT:     %.loc4_35.2: init %empty_tuple.type = call %bound_method.loc4_35.2(%addr)
// CHECK:STDOUT:     return %F.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.8b3d5d.1, constants.%U) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%T.8b3d5d.1
// CHECK:STDOUT:   %U.loc4_25.2 => constants.%U
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.30b
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.65c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc4_35.1 => constants.%require_complete.fe1
// CHECK:STDOUT:   %F.specific_fn.loc5_10.2 => constants.%F.specific_fn
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.lookup_impl_witness
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet
// CHECK:STDOUT:   %.loc4_35.3 => constants.%.1e9
// CHECK:STDOUT:   %impl.elem0.loc4_35.2 => constants.%impl.elem0
// CHECK:STDOUT:   %specific_impl_fn.loc4_35.2 => constants.%specific_impl_fn
// CHECK:STDOUT:   %ptr => constants.%ptr.937
// CHECK:STDOUT:   %require_complete.loc4_35.2 => constants.%require_complete.8fa
// CHECK:STDOUT: }
// CHECK:STDOUT:
