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
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %ptr.a13: type = ptr_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %require_complete.132: <witness> = require_complete_type %ptr.a13 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.79f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %ptr.79f, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.dc4: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.dc4 = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Destroy.Op(%Destroy.facet) [symbolic]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc15_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_6.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc15_6.2: type) {
// CHECK:STDOUT:   %T.loc15_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_6.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.loc16_11.2: type = ptr_type %T.loc15_6.1 [symbolic = %ptr.loc16_11.2 (constants.%ptr.79f)]
// CHECK:STDOUT:   %require_complete.loc16_11: <witness> = require_complete_type %ptr.loc16_11.2 [symbolic = %require_complete.loc16_11 (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %pattern_type.loc16: type = pattern_type %ptr.loc16_11.2 [symbolic = %pattern_type.loc16 (constants.%pattern_type.afe)]
// CHECK:STDOUT:   %require_complete.loc17: <witness> = require_complete_type %T.loc15_6.1 [symbolic = %require_complete.loc17 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %pattern_type.loc17: type = pattern_type %T.loc15_6.1 [symbolic = %pattern_type.loc17 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc16_11.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %ptr.loc16_11.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet)]
// CHECK:STDOUT:   %.loc16_3.2: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc16_3.2 (constants.%.dc4)]
// CHECK:STDOUT:   %impl.elem0.loc16_3.2: @F.%.loc16_3.2 (%.dc4) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc16_3.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc16_3.2: <specific function> = specific_impl_function %impl.elem0.loc16_3.2, @Destroy.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc16_3.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:   %ptr.loc16_3: type = ptr_type %ptr.loc16_11.2 [symbolic = %ptr.loc16_3 (constants.%ptr.a13)]
// CHECK:STDOUT:   %require_complete.loc16_3: <witness> = require_complete_type %ptr.loc16_3 [symbolic = %require_complete.loc16_3 (constants.%require_complete.132)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %p.patt: @F.%pattern_type.loc16 (%pattern_type.afe) = binding_pattern p [concrete]
// CHECK:STDOUT:       %p.var_patt: @F.%pattern_type.loc16 (%pattern_type.afe) = var_pattern %p.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p.var: ref @F.%ptr.loc16_11.2 (%ptr.79f) = var %p.var_patt
// CHECK:STDOUT:     %.loc16_11: type = splice_block %ptr.loc16_11.1 [symbolic = %ptr.loc16_11.2 (constants.%ptr.79f)] {
// CHECK:STDOUT:       %T.ref.loc16: type = name_ref T, %T.loc15_6.2 [symbolic = %T.loc15_6.1 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc16_11.1: type = ptr_type %T.ref.loc16 [symbolic = %ptr.loc16_11.2 (constants.%ptr.79f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: ref @F.%ptr.loc16_11.2 (%ptr.79f) = bind_name p, %p.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %n.patt: @F.%pattern_type.loc17 (%pattern_type.7dc) = binding_pattern n [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p.ref: ref @F.%ptr.loc16_11.2 (%ptr.79f) = name_ref p, %p
// CHECK:STDOUT:     %.loc17_15: @F.%ptr.loc16_11.2 (%ptr.79f) = bind_value %p.ref
// CHECK:STDOUT:     %.loc17_14.1: ref @F.%T.loc15_6.1 (%T) = deref %.loc17_15
// CHECK:STDOUT:     %T.ref.loc17: type = name_ref T, %T.loc15_6.2 [symbolic = %T.loc15_6.1 (constants.%T)]
// CHECK:STDOUT:     %.loc17_14.2: @F.%T.loc15_6.1 (%T) = bind_value %.loc17_14.1
// CHECK:STDOUT:     %n: @F.%T.loc15_6.1 (%T) = bind_name n, %.loc17_14.2
// CHECK:STDOUT:     %impl.elem0.loc16_3.1: @F.%.loc16_3.2 (%.dc4) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc16_3.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc16_3.1: <bound method> = bound_method %p.var, %impl.elem0.loc16_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc16_3.1: <specific function> = specific_impl_function %impl.elem0.loc16_3.1, @Destroy.Op(constants.%Destroy.facet) [symbolic = %specific_impl_fn.loc16_3.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc16_3.2: <bound method> = bound_method %p.var, %specific_impl_fn.loc16_3.1
// CHECK:STDOUT:     %addr: @F.%ptr.loc16_3 (%ptr.a13) = addr_of %p.var
// CHECK:STDOUT:     %.loc16_3.1: init %empty_tuple.type = call %bound_method.loc16_3.2(%addr)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc15_6.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
