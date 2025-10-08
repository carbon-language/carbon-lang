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
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %const.20a: type = const_type %i32 [concrete]
// CHECK:STDOUT:   %ptr.36b: type = ptr_type %const.20a [concrete]
// CHECK:STDOUT:   %const.58f: type = const_type %ptr.36b [concrete]
// CHECK:STDOUT:   %ptr.6e8: type = ptr_type %const.58f [concrete]
// CHECK:STDOUT:   %const.fa2: type = const_type %ptr.6e8 [concrete]
// CHECK:STDOUT:   %pattern_type.800: type = pattern_type %const.fa2 [concrete]
// CHECK:STDOUT:   %pattern_type.a65: type = pattern_type %const.20a [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.afd: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6cd: %Int.as.Copy.impl.Op.type.afd = struct_value () [symbolic]
// CHECK:STDOUT:   %T.be8: %Copy.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.type.c2e: type = fn_type @const.as.Copy.impl.Op, @const.as.Copy.impl(%T.be8) [symbolic]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.a33: %const.as.Copy.impl.Op.type.c2e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a32: <witness> = impl_witness imports.%Copy.impl_witness_table.1ed, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Copy.facet.c49: %Copy.type = facet_value %i32, (%Copy.impl_witness.a32) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.e96: <witness> = impl_witness imports.%Copy.impl_witness_table.9cd, @const.as.Copy.impl(%Copy.facet.c49) [concrete]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.type.1fc: type = fn_type @const.as.Copy.impl.Op, @const.as.Copy.impl(%Copy.facet.c49) [concrete]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.c18: %const.as.Copy.impl.Op.type.1fc = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.b9f: %Copy.type = facet_value %const.20a, (%Copy.impl_witness.e96) [concrete]
// CHECK:STDOUT:   %.78c: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.b9f [concrete]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %const.as.Copy.impl.Op.c18, @const.as.Copy.impl.Op(%Copy.facet.c49) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.d0f6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.afd) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6cd)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ed = impl_witness_table (%Core.import_ref.d0f6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.722: @const.as.Copy.impl.%const.as.Copy.impl.Op.type (%const.as.Copy.impl.Op.type.c2e) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @const.as.Copy.impl.%const.as.Copy.impl.Op (constants.%const.as.Copy.impl.Op.a33)]
// CHECK:STDOUT:   %Copy.impl_witness_table.9cd = impl_witness_table (%Core.import_ref.722), @const.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %p.patt: %pattern_type.800 = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.800 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.a65 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a65 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc16_47: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc16_47: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %const.loc16_41: type = const_type %i32.loc16_47 [concrete = constants.%const.20a]
// CHECK:STDOUT:     %p.param: %const.fa2 = value_param call_param0
// CHECK:STDOUT:     %.loc16: type = splice_block %const.loc16_9 [concrete = constants.%const.fa2] {
// CHECK:STDOUT:       %int_32.loc16_29: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc16_29: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %const.loc16_23: type = const_type %i32.loc16_29 [concrete = constants.%const.20a]
// CHECK:STDOUT:       %ptr.loc16_32: type = ptr_type %const.loc16_23 [concrete = constants.%ptr.36b]
// CHECK:STDOUT:       %const.loc16_16: type = const_type %ptr.loc16_32 [concrete = constants.%const.58f]
// CHECK:STDOUT:       %ptr.loc16_34: type = ptr_type %const.loc16_16 [concrete = constants.%ptr.6e8]
// CHECK:STDOUT:       %const.loc16_9: type = const_type %ptr.loc16_34 [concrete = constants.%const.fa2]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %const.fa2 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %const.20a = out_param call_param1
// CHECK:STDOUT:     %return: ref %const.20a = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p.param: %const.fa2) -> %const.20a {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %const.fa2 = name_ref p, %p
// CHECK:STDOUT:   %.loc17_11.1: ref %const.58f = deref %p.ref
// CHECK:STDOUT:   %.loc17_11.2: %const.58f = bind_value %.loc17_11.1
// CHECK:STDOUT:   %.loc17_10.1: ref %const.20a = deref %.loc17_11.2
// CHECK:STDOUT:   %.loc17_10.2: %const.20a = bind_value %.loc17_10.1
// CHECK:STDOUT:   %impl.elem0: %.78c = impl_witness_access constants.%Copy.impl_witness.e96, element0 [concrete = constants.%const.as.Copy.impl.Op.c18]
// CHECK:STDOUT:   %bound_method.loc17_10.1: <bound method> = bound_method %.loc17_10.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @const.as.Copy.impl.Op(constants.%Copy.facet.c49) [concrete = constants.%const.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_10.2: <bound method> = bound_method %.loc17_10.2, %specific_fn
// CHECK:STDOUT:   %const.as.Copy.impl.Op.call: init %const.20a = call %bound_method.loc17_10.2(%.loc17_10.2)
// CHECK:STDOUT:   return %const.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
