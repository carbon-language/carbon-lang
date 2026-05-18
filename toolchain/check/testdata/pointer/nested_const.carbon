// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/pointer/nested_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/pointer/nested_const.carbon

// TODO: The `const` in the return type should not be necessary.
fn F(p: const (const (const i32*)*)) -> const i32 {
  return **p;
}

// CHECK:STDOUT: --- nested_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %const.52e: type = const_type %i32 [concrete]
// CHECK:STDOUT:   %ptr.04d: type = ptr_type %const.52e [concrete]
// CHECK:STDOUT:   %const.d36: type = const_type %ptr.04d [concrete]
// CHECK:STDOUT:   %ptr.f5f: type = ptr_type %const.d36 [concrete]
// CHECK:STDOUT:   %const.fca: type = const_type %ptr.f5f [concrete]
// CHECK:STDOUT:   %pattern_type.2d0: type = pattern_type %const.fca [concrete]
// CHECK:STDOUT:   %.734: Core.Form = init_form %const.52e [concrete]
// CHECK:STDOUT:   %pattern_type.20a: type = pattern_type %const.52e [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.521: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.783: %Int.as.Copy.impl.Op.type.521 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.0e9: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.type.d7f: type = fn_type @const.as.Copy.impl.Op, @const.as.Copy.impl(%T.0e9) [symbolic]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.fa4: %const.as.Copy.impl.Op.type.d7f = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f1f: <witness> = impl_witness imports.%Copy.impl_witness_table.592, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Copy.facet.ddd: %Copy.type = facet_value %i32, (%Copy.impl_witness.f1f) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.a8d: <witness> = impl_witness imports.%Copy.impl_witness_table.c9c, @const.as.Copy.impl(%Copy.facet.ddd) [concrete]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.type.f75: type = fn_type @const.as.Copy.impl.Op, @const.as.Copy.impl(%Copy.facet.ddd) [concrete]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.f8c: %const.as.Copy.impl.Op.type.f75 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.163: %Copy.type = facet_value %const.52e, (%Copy.impl_witness.a8d) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d2d: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.163) [concrete]
// CHECK:STDOUT:   %.aff: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d2d, %Copy.facet.163 [concrete]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %const.as.Copy.impl.Op.f8c, @const.as.Copy.impl.Op(%Copy.facet.ddd) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.0bb: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.521) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.783)]
// CHECK:STDOUT:   %Copy.impl_witness_table.592 = impl_witness_table (%Core.import_ref.0bb), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.c2b: @const.as.Copy.impl.%const.as.Copy.impl.Op.type (%const.as.Copy.impl.Op.type.d7f) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @const.as.Copy.impl.%const.as.Copy.impl.Op (constants.%const.as.Copy.impl.Op.fa4)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c9c = impl_witness_table (%Core.import_ref.c2b), @const.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.2d0 = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.2d0 = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.20a = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.20a = return_slot_pattern %return.param_patt, %const.loc16_41 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc16_47: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %const.loc16_41: type = const_type %i32.loc16_47 [concrete = constants.%const.52e]
// CHECK:STDOUT:     %.loc16_41: Core.Form = init_form %const.loc16_41 [concrete = constants.%.734]
// CHECK:STDOUT:     %p.param: %const.fca = value_param call_param0
// CHECK:STDOUT:     %.loc16_9: type = splice_block %const.loc16_9 [concrete = constants.%const.fca] {
// CHECK:STDOUT:       %i32.loc16_29: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %const.loc16_23: type = const_type %i32.loc16_29 [concrete = constants.%const.52e]
// CHECK:STDOUT:       %ptr.loc16_32: type = ptr_type %const.loc16_23 [concrete = constants.%ptr.04d]
// CHECK:STDOUT:       %const.loc16_16: type = const_type %ptr.loc16_32 [concrete = constants.%const.d36]
// CHECK:STDOUT:       %ptr.loc16_34: type = ptr_type %const.loc16_16 [concrete = constants.%ptr.f5f]
// CHECK:STDOUT:       %const.loc16_9: type = const_type %ptr.loc16_34 [concrete = constants.%const.fca]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %const.fca = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %const.52e = out_param call_param1
// CHECK:STDOUT:     %return: ref %const.52e = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p.param: %const.fca) -> out %return.param: %const.52e {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %const.fca = name_ref p, %p
// CHECK:STDOUT:   %.loc17_11.1: ref %const.d36 = deref %p.ref
// CHECK:STDOUT:   %.loc17_11.2: %const.d36 = acquire_value %.loc17_11.1
// CHECK:STDOUT:   %.loc17_10.1: ref %const.52e = deref %.loc17_11.2
// CHECK:STDOUT:   %.loc17_10.2: %const.52e = acquire_value %.loc17_10.1
// CHECK:STDOUT:   %impl.elem0: %.aff = impl_witness_access constants.%Copy.impl_witness.a8d, element0 [concrete = constants.%const.as.Copy.impl.Op.f8c]
// CHECK:STDOUT:   %bound_method.loc17_10.1: <bound method> = bound_method %.loc17_10.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @const.as.Copy.impl.Op(constants.%Copy.facet.ddd) [concrete = constants.%const.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_10.2: <bound method> = bound_method %.loc17_10.2, %specific_fn
// CHECK:STDOUT:   %const.as.Copy.impl.Op.call: init %const.52e = call %bound_method.loc17_10.2(%.loc17_10.2)
// CHECK:STDOUT:   return %const.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
