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
  let n: T = *p;
}

// CHECK:STDOUT: --- type_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.482: <witness> = require_complete_type %ptr [symbolic]
// CHECK:STDOUT:   %pattern_type.a60: type = pattern_type %ptr [symbolic]
// CHECK:STDOUT:   %require_complete.4b7: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.e68: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %DestroyT: %type_where = symbolic_binding DestroyT, 0 [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%DestroyT) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.fd5: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %ptr, () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.60c: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.e46: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.9ef: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.e46 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %ptr, (%Destroy.impl_witness.60c) [symbolic]
// CHECK:STDOUT:   %.17f: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.9ef, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.5c4: @DestroyT.binding.as_type.as.Destroy.impl.%DestroyT.binding.as_type.as.Destroy.impl.Op.type (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @DestroyT.binding.as_type.as.Destroy.impl.%DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.fd5)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.5c4), @DestroyT.binding.as_type.as.Destroy.impl [concrete]
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
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc15_6.2: type = symbolic_binding T, 0 [symbolic = %T.loc15_6.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc15_6.2: type) {
// CHECK:STDOUT:   %T.loc15_6.1: type = symbolic_binding T, 0 [symbolic = %T.loc15_6.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.loc16_11.2: type = ptr_type %T.loc15_6.1 [symbolic = %ptr.loc16_11.2 (constants.%ptr)]
// CHECK:STDOUT:   %require_complete.loc16: <witness> = require_complete_type %ptr.loc16_11.2 [symbolic = %require_complete.loc16 (constants.%require_complete.482)]
// CHECK:STDOUT:   %pattern_type.loc16: type = pattern_type %ptr.loc16_11.2 [symbolic = %pattern_type.loc16 (constants.%pattern_type.a60)]
// CHECK:STDOUT:   %require_complete.loc17: <witness> = require_complete_type %T.loc15_6.1 [symbolic = %require_complete.loc17 (constants.%require_complete.4b7)]
// CHECK:STDOUT:   %pattern_type.loc17: type = pattern_type %T.loc15_6.1 [symbolic = %pattern_type.loc17 (constants.%pattern_type.e68)]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %ptr.loc16_11.2, () [symbolic = %facet_value (constants.%facet_value)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.60c)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %ptr.loc16_11.2, (%Destroy.impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet)]
// CHECK:STDOUT:   %.loc16_3: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc16_3 (constants.%.17f)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.type (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.e46)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op: @F.%DestroyT.binding.as_type.as.Destroy.impl.Op.type (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.e46) = struct_value () [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.9ef)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %p.patt: @F.%pattern_type.loc16 (%pattern_type.a60) = ref_binding_pattern p [concrete]
// CHECK:STDOUT:       %p.var_patt: @F.%pattern_type.loc16 (%pattern_type.a60) = var_pattern %p.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p.var: ref @F.%ptr.loc16_11.2 (%ptr) = var %p.var_patt
// CHECK:STDOUT:     %.loc16_11: type = splice_block %ptr.loc16_11.1 [symbolic = %ptr.loc16_11.2 (constants.%ptr)] {
// CHECK:STDOUT:       %T.ref.loc16: type = name_ref T, %T.loc15_6.2 [symbolic = %T.loc15_6.1 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc16_11.1: type = ptr_type %T.ref.loc16 [symbolic = %ptr.loc16_11.2 (constants.%ptr)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: ref @F.%ptr.loc16_11.2 (%ptr) = ref_binding p, %p.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %n.patt: @F.%pattern_type.loc17 (%pattern_type.e68) = value_binding_pattern n [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p.ref: ref @F.%ptr.loc16_11.2 (%ptr) = name_ref p, %p
// CHECK:STDOUT:     %.loc17_15: @F.%ptr.loc16_11.2 (%ptr) = acquire_value %p.ref
// CHECK:STDOUT:     %.loc17_14.1: ref @F.%T.loc15_6.1 (%T) = deref %.loc17_15
// CHECK:STDOUT:     %T.ref.loc17: type = name_ref T, %T.loc15_6.2 [symbolic = %T.loc15_6.1 (constants.%T)]
// CHECK:STDOUT:     %.loc17_14.2: @F.%T.loc15_6.1 (%T) = acquire_value %.loc17_14.1
// CHECK:STDOUT:     %n: @F.%T.loc15_6.1 (%T) = value_binding n, %.loc17_14.2
// CHECK:STDOUT:     %impl.elem0: @F.%.loc16_3 (%.17f) = impl_witness_access constants.%Destroy.impl_witness.60c, element0 [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.9ef)]
// CHECK:STDOUT:     %bound_method.loc16_3.1: <bound method> = bound_method %p.var, %impl.elem0
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn)]
// CHECK:STDOUT:     %bound_method.loc16_3.2: <bound method> = bound_method %p.var, %specific_fn
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc16_3.2(%p.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc15_6.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
