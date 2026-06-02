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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/pointer/types.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/pointer/types.carbon

fn Ptr(p: i32*) -> i32* {
  return p;
}

fn ConstPtr(p: const i32*) -> (const i32)* {
  return p;
}

// CHECK:STDOUT: --- types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.847: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7b5: type = pattern_type %ptr.847 [concrete]
// CHECK:STDOUT:   %.060: Core.Form = init_form %ptr.847 [concrete]
// CHECK:STDOUT:   %Ptr.type: type = fn_type @Ptr [concrete]
// CHECK:STDOUT:   %Ptr: %Ptr.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.aa4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.5a1: %ptr.as.Copy.impl.Op.type.aa4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.c37: <witness> = impl_witness imports.%Copy.impl_witness_table.6b9, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.ae4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.6e4: %ptr.as.Copy.impl.Op.type.ae4 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.17a: %Copy.type = facet_value %ptr.847, (%Copy.impl_witness.c37) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.165: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.17a) [concrete]
// CHECK:STDOUT:   %.cd7: type = fn_type_with_self_type %Copy.WithSelf.Op.type.165, %Copy.facet.17a [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.cef: <specific function> = specific_function %ptr.as.Copy.impl.Op.6e4, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %const.72d: type = const_type %i32 [concrete]
// CHECK:STDOUT:   %ptr.9a7: type = ptr_type %const.72d [concrete]
// CHECK:STDOUT:   %pattern_type.9ec: type = pattern_type %ptr.9a7 [concrete]
// CHECK:STDOUT:   %.cf7: Core.Form = init_form %ptr.9a7 [concrete]
// CHECK:STDOUT:   %ConstPtr.type: type = fn_type @ConstPtr [concrete]
// CHECK:STDOUT:   %ConstPtr: %ConstPtr.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.9d6: <witness> = impl_witness imports.%Copy.impl_witness_table.6b9, @ptr.as.Copy.impl(%const.72d) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.af3: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%const.72d) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.dc5: %ptr.as.Copy.impl.Op.type.af3 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.387: %Copy.type = facet_value %ptr.9a7, (%Copy.impl_witness.9d6) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.a88: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.387) [concrete]
// CHECK:STDOUT:   %.5d6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.a88, %Copy.facet.387 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.8aa: <specific function> = specific_function %ptr.as.Copy.impl.Op.dc5, @ptr.as.Copy.impl.Op(%const.72d) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.aa4: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.aa4) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.5a1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.6b9 = impl_witness_table (%Core.import_ref.aa4), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Ptr = %Ptr.decl
// CHECK:STDOUT:     .ConstPtr = %ConstPtr.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Ptr.decl: %Ptr.type = fn_decl @Ptr [concrete = constants.%Ptr] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.7b5 = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.7b5 = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7b5 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7b5 = return_slot_pattern %return.param_patt, %ptr.loc15_23 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc15_20: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc15_23: type = ptr_type %i32.loc15_20 [concrete = constants.%ptr.847]
// CHECK:STDOUT:     %.loc15_23: Core.Form = init_form %ptr.loc15_23 [concrete = constants.%.060]
// CHECK:STDOUT:     %p.param: %ptr.847 = value_param call_param0
// CHECK:STDOUT:     %.loc15_14: type = splice_block %ptr.loc15_14 [concrete = constants.%ptr.847] {
// CHECK:STDOUT:       %i32.loc15_11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %ptr.loc15_14: type = ptr_type %i32.loc15_11 [concrete = constants.%ptr.847]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.847 = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.847 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.847 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConstPtr.decl: %ConstPtr.type = fn_decl @ConstPtr [concrete = constants.%ConstPtr] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.9ec = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.9ec = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9ec = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.9ec = return_slot_pattern %return.param_patt, %ptr.loc19_42 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc19_38: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %const.loc19_32: type = const_type %i32.loc19_38 [concrete = constants.%const.72d]
// CHECK:STDOUT:     %ptr.loc19_42: type = ptr_type %const.loc19_32 [concrete = constants.%ptr.9a7]
// CHECK:STDOUT:     %.loc19_42: Core.Form = init_form %ptr.loc19_42 [concrete = constants.%.cf7]
// CHECK:STDOUT:     %p.param: %ptr.9a7 = value_param call_param0
// CHECK:STDOUT:     %.loc19_25: type = splice_block %ptr.loc19_25 [concrete = constants.%ptr.9a7] {
// CHECK:STDOUT:       %i32.loc19_22: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %const.loc19_16: type = const_type %i32.loc19_22 [concrete = constants.%const.72d]
// CHECK:STDOUT:       %ptr.loc19_25: type = ptr_type %const.loc19_16 [concrete = constants.%ptr.9a7]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.9a7 = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.9a7 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.9a7 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Ptr(%p.param: %ptr.847) -> out %return.param: %ptr.847 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.847 = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.cd7 = impl_witness_access constants.%Copy.impl_witness.c37, element0 [concrete = constants.%ptr.as.Copy.impl.Op.6e4]
// CHECK:STDOUT:   %bound_method.loc16_10.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.cef]
// CHECK:STDOUT:   %bound_method.loc16_10.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.847 = call %bound_method.loc16_10.2(%p.ref)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstPtr(%p.param: %ptr.9a7) -> out %return.param: %ptr.9a7 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.9a7 = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.5d6 = impl_witness_access constants.%Copy.impl_witness.9d6, element0 [concrete = constants.%ptr.as.Copy.impl.Op.dc5]
// CHECK:STDOUT:   %bound_method.loc20_10.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%const.72d) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.8aa]
// CHECK:STDOUT:   %bound_method.loc20_10.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.9a7 = call %bound_method.loc20_10.2(%p.ref)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
