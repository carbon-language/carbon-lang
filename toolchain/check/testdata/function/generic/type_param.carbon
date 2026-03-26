// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/type_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/type_param.carbon

fn F(T:! type) {
  var p: T*;
  let unused n: T = *p;
}

// CHECK:STDOUT: --- type_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %require_complete.ef1: <witness> = require_complete_type %ptr [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67d [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T.67d [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @DefaultOrUnformed [symbolic]
// CHECK:STDOUT:   %.713: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%ptr) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %ptr, (%DefaultOrUnformed.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.d74: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet) [symbolic]
// CHECK:STDOUT:   %.425: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.d74, %DefaultOrUnformed.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.45d: %.425 = impl_witness_access %DefaultOrUnformed.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a93: <specific function> = specific_impl_function %impl.elem0.45d, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %ptr, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.bb2: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet) [symbolic]
// CHECK:STDOUT:   %.c3f: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.bb2, %Destroy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.b55: %.c3f = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5c7: <specific function> = specific_impl_function %impl.elem0.b55, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .DefaultOrUnformed = %Core.DefaultOrUnformed
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/parts/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_10.1: type = splice_block %.loc15_10.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc15_10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15_6.2: type = symbolic_binding T, 0 [symbolic = %T.loc15_6.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc15_6.2: type) {
// CHECK:STDOUT:   %T.loc15_6.1: type = symbolic_binding T, 0 [symbolic = %T.loc15_6.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.loc16_11.2: type = ptr_type %T.loc15_6.1 [symbolic = %ptr.loc16_11.2 (constants.%ptr)]
// CHECK:STDOUT:   %require_complete.loc16: <witness> = require_complete_type %ptr.loc16_11.2 [symbolic = %require_complete.loc16 (constants.%require_complete.ef1)]
// CHECK:STDOUT:   %pattern_type.loc16: type = pattern_type %ptr.loc16_11.2 [symbolic = %pattern_type.loc16 (constants.%pattern_type.4f4)]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc16_11.2, @DefaultOrUnformed [symbolic = %DefaultOrUnformed.lookup_impl_witness (constants.%DefaultOrUnformed.lookup_impl_witness)]
// CHECK:STDOUT:   %.loc16_12.3: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%ptr.loc16_11.2) [symbolic = %.loc16_12.3 (constants.%.713)]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc16_12.2: %DefaultOrUnformed.type = facet_value %ptr.loc16_11.2, (%DefaultOrUnformed.lookup_impl_witness) [symbolic = %DefaultOrUnformed.facet.loc16_12.2 (constants.%DefaultOrUnformed.facet)]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.loc16_12.2) [symbolic = %DefaultOrUnformed.WithSelf.Op.type (constants.%DefaultOrUnformed.WithSelf.Op.type.d74)]
// CHECK:STDOUT:   %.loc16_12.4: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type, %DefaultOrUnformed.facet.loc16_12.2 [symbolic = %.loc16_12.4 (constants.%.425)]
// CHECK:STDOUT:   %impl.elem0.loc16_12.2: @F.%.loc16_12.4 (%.425) = impl_witness_access %DefaultOrUnformed.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc16_12.2 (constants.%impl.elem0.45d)]
// CHECK:STDOUT:   %specific_impl_fn.loc16_12.2: <specific function> = specific_impl_function %impl.elem0.loc16_12.2, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.loc16_12.2) [symbolic = %specific_impl_fn.loc16_12.2 (constants.%specific_impl_fn.a93)]
// CHECK:STDOUT:   %require_complete.loc17: <witness> = require_complete_type %T.loc15_6.1 [symbolic = %require_complete.loc17 (constants.%require_complete.944)]
// CHECK:STDOUT:   %pattern_type.loc17: type = pattern_type %T.loc15_6.1 [symbolic = %pattern_type.loc17 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc16_11.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %ptr.loc16_11.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.bb2)]
// CHECK:STDOUT:   %.loc16_3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet [symbolic = %.loc16_3 (constants.%.c3f)]
// CHECK:STDOUT:   %impl.elem0.loc16_3.2: @F.%.loc16_3 (%.c3f) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc16_3.2 (constants.%impl.elem0.b55)]
// CHECK:STDOUT:   %specific_impl_fn.loc16_3.2: <specific function> = specific_impl_function %impl.elem0.loc16_3.2, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc16_3.2 (constants.%specific_impl_fn.5c7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %p.patt: @F.%pattern_type.loc16 (%pattern_type.4f4) = ref_binding_pattern p [concrete]
// CHECK:STDOUT:       %p.var_patt: @F.%pattern_type.loc16 (%pattern_type.4f4) = var_pattern %p.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p.var: ref @F.%ptr.loc16_11.2 (%ptr) = var %p.var_patt
// CHECK:STDOUT:     %DefaultOrUnformed.facet.loc16_12.1: %DefaultOrUnformed.type = facet_value constants.%ptr, (constants.%DefaultOrUnformed.lookup_impl_witness) [symbolic = %DefaultOrUnformed.facet.loc16_12.2 (constants.%DefaultOrUnformed.facet)]
// CHECK:STDOUT:     %.loc16_12.1: %DefaultOrUnformed.type = converted constants.%ptr, %DefaultOrUnformed.facet.loc16_12.1 [symbolic = %DefaultOrUnformed.facet.loc16_12.2 (constants.%DefaultOrUnformed.facet)]
// CHECK:STDOUT:     %as_type: type = facet_access_type %.loc16_12.1 [symbolic = %ptr.loc16_11.2 (constants.%ptr)]
// CHECK:STDOUT:     %.loc16_12.2: type = converted %.loc16_12.1, %as_type [symbolic = %ptr.loc16_11.2 (constants.%ptr)]
// CHECK:STDOUT:     %impl.elem0.loc16_12.1: @F.%.loc16_12.4 (%.425) = impl_witness_access constants.%DefaultOrUnformed.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc16_12.2 (constants.%impl.elem0.45d)]
// CHECK:STDOUT:     %specific_impl_fn.loc16_12.1: <specific function> = specific_impl_function %impl.elem0.loc16_12.1, @DefaultOrUnformed.WithSelf.Op(constants.%DefaultOrUnformed.facet) [symbolic = %specific_impl_fn.loc16_12.2 (constants.%specific_impl_fn.a93)]
// CHECK:STDOUT:     %DefaultOrUnformed.WithSelf.Op.call: init @F.%ptr.loc16_11.2 (%ptr) = call %specific_impl_fn.loc16_12.1()
// CHECK:STDOUT:     assign %p.var, %DefaultOrUnformed.WithSelf.Op.call
// CHECK:STDOUT:     %.loc16_11: type = splice_block %ptr.loc16_11.1 [symbolic = %ptr.loc16_11.2 (constants.%ptr)] {
// CHECK:STDOUT:       %T.ref.loc16: type = name_ref T, %T.loc15_6.2 [symbolic = %T.loc15_6.1 (constants.%T.67d)]
// CHECK:STDOUT:       %ptr.loc16_11.1: type = ptr_type %T.ref.loc16 [symbolic = %ptr.loc16_11.2 (constants.%ptr)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: ref @F.%ptr.loc16_11.2 (%ptr) = ref_binding p, %p.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %n.patt: @F.%pattern_type.loc17 (%pattern_type.51d) = value_binding_pattern n [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p.ref: ref @F.%ptr.loc16_11.2 (%ptr) = name_ref p, %p
// CHECK:STDOUT:     %.loc17_22: @F.%ptr.loc16_11.2 (%ptr) = acquire_value %p.ref
// CHECK:STDOUT:     %.loc17_21.1: ref @F.%T.loc15_6.1 (%T.67d) = deref %.loc17_22
// CHECK:STDOUT:     %T.ref.loc17: type = name_ref T, %T.loc15_6.2 [symbolic = %T.loc15_6.1 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc17_21.2: @F.%T.loc15_6.1 (%T.67d) = acquire_value %.loc17_21.1
// CHECK:STDOUT:     %n: @F.%T.loc15_6.1 (%T.67d) = value_binding n, %.loc17_21.2
// CHECK:STDOUT:     %impl.elem0.loc16_3.1: @F.%.loc16_3 (%.c3f) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc16_3.2 (constants.%impl.elem0.b55)]
// CHECK:STDOUT:     %bound_method.loc16_3.1: <bound method> = bound_method %p.var, %impl.elem0.loc16_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc16_3.1: <specific function> = specific_impl_function %impl.elem0.loc16_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet) [symbolic = %specific_impl_fn.loc16_3.2 (constants.%specific_impl_fn.5c7)]
// CHECK:STDOUT:     %bound_method.loc16_3.2: <bound method> = bound_method %p.var, %specific_impl_fn.loc16_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc16_3.2(%p.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc15_6.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
