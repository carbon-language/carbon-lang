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
// CHECK:STDOUT:   %const.72d: type = const_type %i32 [concrete]
// CHECK:STDOUT:   %ptr.9a7: type = ptr_type %const.72d [concrete]
// CHECK:STDOUT:   %const.4d4: type = const_type %ptr.9a7 [concrete]
// CHECK:STDOUT:   %ptr.47a: type = ptr_type %const.4d4 [concrete]
// CHECK:STDOUT:   %const.4bc: type = const_type %ptr.47a [concrete]
// CHECK:STDOUT:   %pattern_type.b74: type = pattern_type %const.4bc [concrete]
// CHECK:STDOUT:   %.f3b: Core.Form = init_form %const.72d [concrete]
// CHECK:STDOUT:   %pattern_type.68c: type = pattern_type %const.72d [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.2f2: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.a64: %Int.as.Copy.impl.Op.type.2f2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.f49: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.type.6d8: type = fn_type @const.as.Copy.impl.Op, @const.as.Copy.impl(%T.f49) [symbolic]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.e95: %const.as.Copy.impl.Op.type.6d8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.1a4: <witness> = impl_witness imports.%Copy.impl_witness_table.0ac, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Copy.facet.14d: %Copy.type = facet_value %i32, (%Copy.impl_witness.1a4) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.c9a: <witness> = impl_witness imports.%Copy.impl_witness_table.b24, @const.as.Copy.impl(%Copy.facet.14d) [concrete]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.type.41e: type = fn_type @const.as.Copy.impl.Op, @const.as.Copy.impl(%Copy.facet.14d) [concrete]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.8c2: %const.as.Copy.impl.Op.type.41e = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.022: %Copy.type = facet_value %const.72d, (%Copy.impl_witness.c9a) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.dd5: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.022) [concrete]
// CHECK:STDOUT:   %.c35: type = fn_type_with_self_type %Copy.WithSelf.Op.type.dd5, %Copy.facet.022 [concrete]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %const.as.Copy.impl.Op.8c2, @const.as.Copy.impl.Op(%Copy.facet.14d) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.55f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.2f2) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.a64)]
// CHECK:STDOUT:   %Copy.impl_witness_table.0ac = impl_witness_table (%Core.import_ref.55f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.60b: @const.as.Copy.impl.%const.as.Copy.impl.Op.type (%const.as.Copy.impl.Op.type.6d8) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @const.as.Copy.impl.%const.as.Copy.impl.Op (constants.%const.as.Copy.impl.Op.e95)]
// CHECK:STDOUT:   %Copy.impl_witness_table.b24 = impl_witness_table (%Core.import_ref.60b), @const.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.b74 = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.b74 = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.68c = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.68c = return_slot_pattern %return.param_patt, %const.loc16_41 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc16_47: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %const.loc16_41: type = const_type %i32.loc16_47 [concrete = constants.%const.72d]
// CHECK:STDOUT:     %.loc16_41: Core.Form = init_form %const.loc16_41 [concrete = constants.%.f3b]
// CHECK:STDOUT:     %p.param: %const.4bc = value_param call_param0
// CHECK:STDOUT:     %.loc16_9: type = splice_block %const.loc16_9 [concrete = constants.%const.4bc] {
// CHECK:STDOUT:       %i32.loc16_29: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %const.loc16_23: type = const_type %i32.loc16_29 [concrete = constants.%const.72d]
// CHECK:STDOUT:       %ptr.loc16_32: type = ptr_type %const.loc16_23 [concrete = constants.%ptr.9a7]
// CHECK:STDOUT:       %const.loc16_16: type = const_type %ptr.loc16_32 [concrete = constants.%const.4d4]
// CHECK:STDOUT:       %ptr.loc16_34: type = ptr_type %const.loc16_16 [concrete = constants.%ptr.47a]
// CHECK:STDOUT:       %const.loc16_9: type = const_type %ptr.loc16_34 [concrete = constants.%const.4bc]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %const.4bc = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %const.72d = out_param call_param1
// CHECK:STDOUT:     %return: ref %const.72d = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p.param: %const.4bc) -> out %return.param: %const.72d {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %const.4bc = name_ref p, %p
// CHECK:STDOUT:   %.loc17_11.1: ref %const.4d4 = deref %p.ref
// CHECK:STDOUT:   %.loc17_11.2: %const.4d4 = acquire_value %.loc17_11.1
// CHECK:STDOUT:   %.loc17_10.1: ref %const.72d = deref %.loc17_11.2
// CHECK:STDOUT:   %.loc17_10.2: %const.72d = acquire_value %.loc17_10.1
// CHECK:STDOUT:   %impl.elem0: %.c35 = impl_witness_access constants.%Copy.impl_witness.c9a, element0 [concrete = constants.%const.as.Copy.impl.Op.8c2]
// CHECK:STDOUT:   %bound_method.loc17_10.1: <bound method> = bound_method %.loc17_10.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @const.as.Copy.impl.Op(constants.%Copy.facet.14d) [concrete = constants.%const.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_10.2: <bound method> = bound_method %.loc17_10.2, %specific_fn
// CHECK:STDOUT:   %const.as.Copy.impl.Op.call: init %const.72d = call %bound_method.loc17_10.2(%.loc17_10.2)
// CHECK:STDOUT:   return %const.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
