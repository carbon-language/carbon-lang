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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %const.003: type = const_type %i32 [concrete]
// CHECK:STDOUT:   %ptr.75b: type = ptr_type %const.003 [concrete]
// CHECK:STDOUT:   %const.9c1: type = const_type %ptr.75b [concrete]
// CHECK:STDOUT:   %ptr.932: type = ptr_type %const.9c1 [concrete]
// CHECK:STDOUT:   %const.07d: type = const_type %ptr.932 [concrete]
// CHECK:STDOUT:   %pattern_type.055: type = pattern_type %const.07d [concrete]
// CHECK:STDOUT:   %p.param_patt: %pattern_type.055 = value_param_pattern [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.055 = wrapper_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:   %.776: Core.Form = init_form %const.003 [concrete]
// CHECK:STDOUT:   %pattern_type.a76: type = pattern_type %const.003 [concrete]
// CHECK:STDOUT:   %return.param_patt.747: %pattern_type.a76 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.b17: %pattern_type.a76 = return_slot_pattern %return.param_patt.747, %const.003 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.type.82e: type = fn_type @const.as.Copy.impl.Op, @const.as.Copy.impl(%T.f84) [symbolic]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.46a: %const.as.Copy.impl.Op.type.82e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Copy.facet.95d: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.526: <witness> = impl_witness imports.%Copy.impl_witness_table.aca, @const.as.Copy.impl(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.type.0da: type = fn_type @const.as.Copy.impl.Op, @const.as.Copy.impl(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.594: %const.as.Copy.impl.Op.type.0da = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.f50: %Copy.type = facet_value %const.003, (%Copy.impl_witness.526) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.e9c: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.f50) [concrete]
// CHECK:STDOUT:   %.7ba: type = fn_type_with_self_type %Copy.WithSelf.Op.type.e9c, %Copy.facet.f50 [concrete]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %const.as.Copy.impl.Op.594, @const.as.Copy.impl.Op(%Copy.facet.95d) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.ee9: @const.as.Copy.impl.%const.as.Copy.impl.Op.type (%const.as.Copy.impl.Op.type.82e) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @const.as.Copy.impl.%const.as.Copy.impl.Op (constants.%const.as.Copy.impl.Op.46a)]
// CHECK:STDOUT:   %Copy.impl_witness_table.aca = impl_witness_table (%Core.import_ref.ee9), @const.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.055 = value_param_pattern [concrete = constants.%p.param_patt]
// CHECK:STDOUT:     %p.patt: %pattern_type.055 = wrapper_binding_pattern p, %p.param_patt [concrete = constants.%p.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a76 = out_param_pattern [concrete = constants.%return.param_patt.747]
// CHECK:STDOUT:     %return.patt: %pattern_type.a76 = return_slot_pattern %return.param_patt, %const.loc16_41 [concrete = constants.%return.patt.b17]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc16_47: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %const.loc16_41: type = const_type %i32.loc16_47 [concrete = constants.%const.003]
// CHECK:STDOUT:     %.loc16_41: Core.Form = init_form %const.loc16_41 [concrete = constants.%.776]
// CHECK:STDOUT:     %p.param: %const.07d = value_param call_param0
// CHECK:STDOUT:     %.loc16_9: type = splice_block %const.loc16_9 [concrete = constants.%const.07d] {
// CHECK:STDOUT:       %i32.loc16_29: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %const.loc16_23: type = const_type %i32.loc16_29 [concrete = constants.%const.003]
// CHECK:STDOUT:       %ptr.loc16_32: type = ptr_type %const.loc16_23 [concrete = constants.%ptr.75b]
// CHECK:STDOUT:       %const.loc16_16: type = const_type %ptr.loc16_32 [concrete = constants.%const.9c1]
// CHECK:STDOUT:       %ptr.loc16_34: type = ptr_type %const.loc16_16 [concrete = constants.%ptr.932]
// CHECK:STDOUT:       %const.loc16_9: type = const_type %ptr.loc16_34 [concrete = constants.%const.07d]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %const.07d = wrapper_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %const.003 = out_param call_param1
// CHECK:STDOUT:     %return: ref %const.003 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p.param: %const.07d) -> out %return.param: %const.003 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %const.07d = name_ref p, %p
// CHECK:STDOUT:   %.loc17_11.1: ref %const.9c1 = deref %p.ref
// CHECK:STDOUT:   %.loc17_11.2: %const.9c1 = acquire_value %.loc17_11.1
// CHECK:STDOUT:   %.loc17_10.1: ref %const.003 = deref %.loc17_11.2
// CHECK:STDOUT:   %.loc17_10.2: %const.003 = acquire_value %.loc17_10.1
// CHECK:STDOUT:   %impl.elem0: %.7ba = impl_witness_access constants.%Copy.impl_witness.526, element0 [concrete = constants.%const.as.Copy.impl.Op.594]
// CHECK:STDOUT:   %bound_method.loc17_10.1: <bound method> = bound_method %.loc17_10.2, %impl.elem0
// CHECK:STDOUT:   %Copy.facet.loc17_10.1: %Copy.type = facet_value constants.%i32, (constants.%Copy.impl_witness.8af) [concrete = constants.%Copy.facet.95d]
// CHECK:STDOUT:   %.loc17_10.3: %Copy.type = converted constants.%i32, %Copy.facet.loc17_10.1 [concrete = constants.%Copy.facet.95d]
// CHECK:STDOUT:   %Copy.facet.loc17_10.2: %Copy.type = facet_value constants.%i32, (constants.%Copy.impl_witness.8af) [concrete = constants.%Copy.facet.95d]
// CHECK:STDOUT:   %.loc17_10.4: %Copy.type = converted constants.%i32, %Copy.facet.loc17_10.2 [concrete = constants.%Copy.facet.95d]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @const.as.Copy.impl.Op(constants.%Copy.facet.95d) [concrete = constants.%const.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_10.2: <bound method> = bound_method %.loc17_10.2, %specific_fn
// CHECK:STDOUT:   %const.as.Copy.impl.Op.call: init %const.003 = call %bound_method.loc17_10.2(%.loc17_10.2)
// CHECK:STDOUT:   return %const.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
