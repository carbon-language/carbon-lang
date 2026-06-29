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
// CHECK:STDOUT:   %ptr.d08: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.f00: type = pattern_type %ptr.d08 [concrete]
// CHECK:STDOUT:   %p.param_patt.b89: %pattern_type.f00 = value_param_pattern [concrete]
// CHECK:STDOUT:   %p.patt.204: %pattern_type.f00 = at_binding_pattern p, %p.param_patt.b89 [concrete]
// CHECK:STDOUT:   %.952: Core.Form = init_form %ptr.d08 [concrete]
// CHECK:STDOUT:   %return.param_patt.7dd: %pattern_type.f00 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.dbe: %pattern_type.f00 = return_slot_pattern %return.param_patt.7dd, %ptr.d08 [concrete]
// CHECK:STDOUT:   %Ptr.type: type = fn_type @Ptr [concrete]
// CHECK:STDOUT:   %Ptr: %Ptr.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.ff5: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ce9: %ptr.as.Copy.impl.Op.type.ff5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.ce1: <witness> = impl_witness imports.%Copy.impl_witness_table.4f1, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.c1c: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.d1c: %ptr.as.Copy.impl.Op.type.c1c = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.b49: %Copy.type = facet_value %ptr.d08, (%Copy.impl_witness.ce1) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f9d: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.b49) [concrete]
// CHECK:STDOUT:   %.ac9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f9d, %Copy.facet.b49 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.b3e: <specific function> = specific_function %ptr.as.Copy.impl.Op.d1c, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %const.6a8: type = const_type %i32 [concrete]
// CHECK:STDOUT:   %ptr.774: type = ptr_type %const.6a8 [concrete]
// CHECK:STDOUT:   %pattern_type.8c1: type = pattern_type %ptr.774 [concrete]
// CHECK:STDOUT:   %p.param_patt.5f6: %pattern_type.8c1 = value_param_pattern [concrete]
// CHECK:STDOUT:   %p.patt.226: %pattern_type.8c1 = at_binding_pattern p, %p.param_patt.5f6 [concrete]
// CHECK:STDOUT:   %.af2: Core.Form = init_form %ptr.774 [concrete]
// CHECK:STDOUT:   %return.param_patt.0fc: %pattern_type.8c1 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e15: %pattern_type.8c1 = return_slot_pattern %return.param_patt.0fc, %ptr.774 [concrete]
// CHECK:STDOUT:   %ConstPtr.type: type = fn_type @ConstPtr [concrete]
// CHECK:STDOUT:   %ConstPtr: %ConstPtr.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.1fe: <witness> = impl_witness imports.%Copy.impl_witness_table.4f1, @ptr.as.Copy.impl(%const.6a8) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.17e: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%const.6a8) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.eb7: %ptr.as.Copy.impl.Op.type.17e = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.82e: %Copy.type = facet_value %ptr.774, (%Copy.impl_witness.1fe) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.18a: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.82e) [concrete]
// CHECK:STDOUT:   %.31f: type = fn_type_with_self_type %Copy.WithSelf.Op.type.18a, %Copy.facet.82e [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.1b4: <specific function> = specific_function %ptr.as.Copy.impl.Op.eb7, @ptr.as.Copy.impl.Op(%const.6a8) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.e5f: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.ff5) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.ce9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.4f1 = impl_witness_table (%Core.import_ref.e5f), @ptr.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %p.param_patt: %pattern_type.f00 = value_param_pattern [concrete = constants.%p.param_patt.b89]
// CHECK:STDOUT:     %p.patt: %pattern_type.f00 = at_binding_pattern p, %p.param_patt [concrete = constants.%p.patt.204]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.f00 = out_param_pattern [concrete = constants.%return.param_patt.7dd]
// CHECK:STDOUT:     %return.patt: %pattern_type.f00 = return_slot_pattern %return.param_patt, %ptr.loc15_23 [concrete = constants.%return.patt.dbe]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc15_20: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc15_23: type = ptr_type %i32.loc15_20 [concrete = constants.%ptr.d08]
// CHECK:STDOUT:     %.loc15_23: Core.Form = init_form %ptr.loc15_23 [concrete = constants.%.952]
// CHECK:STDOUT:     %p.param: %ptr.d08 = value_param call_param0
// CHECK:STDOUT:     %.loc15_14: type = splice_block %ptr.loc15_14 [concrete = constants.%ptr.d08] {
// CHECK:STDOUT:       %i32.loc15_11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %ptr.loc15_14: type = ptr_type %i32.loc15_11 [concrete = constants.%ptr.d08]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.d08 = wrapper_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.d08 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.d08 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConstPtr.decl: %ConstPtr.type = fn_decl @ConstPtr [concrete = constants.%ConstPtr] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.8c1 = value_param_pattern [concrete = constants.%p.param_patt.5f6]
// CHECK:STDOUT:     %p.patt: %pattern_type.8c1 = at_binding_pattern p, %p.param_patt [concrete = constants.%p.patt.226]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.8c1 = out_param_pattern [concrete = constants.%return.param_patt.0fc]
// CHECK:STDOUT:     %return.patt: %pattern_type.8c1 = return_slot_pattern %return.param_patt, %ptr.loc19_42 [concrete = constants.%return.patt.e15]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc19_38: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %const.loc19_32: type = const_type %i32.loc19_38 [concrete = constants.%const.6a8]
// CHECK:STDOUT:     %ptr.loc19_42: type = ptr_type %const.loc19_32 [concrete = constants.%ptr.774]
// CHECK:STDOUT:     %.loc19_42: Core.Form = init_form %ptr.loc19_42 [concrete = constants.%.af2]
// CHECK:STDOUT:     %p.param: %ptr.774 = value_param call_param0
// CHECK:STDOUT:     %.loc19_25: type = splice_block %ptr.loc19_25 [concrete = constants.%ptr.774] {
// CHECK:STDOUT:       %i32.loc19_22: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %const.loc19_16: type = const_type %i32.loc19_22 [concrete = constants.%const.6a8]
// CHECK:STDOUT:       %ptr.loc19_25: type = ptr_type %const.loc19_16 [concrete = constants.%ptr.774]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.774 = wrapper_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.774 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.774 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Ptr(%p.param: %ptr.d08) -> out %return.param: %ptr.d08 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.d08 = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.ac9 = impl_witness_access constants.%Copy.impl_witness.ce1, element0 [concrete = constants.%ptr.as.Copy.impl.Op.d1c]
// CHECK:STDOUT:   %bound_method.loc16_10.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.b3e]
// CHECK:STDOUT:   %bound_method.loc16_10.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.d08 = call %bound_method.loc16_10.2(%p.ref)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstPtr(%p.param: %ptr.774) -> out %return.param: %ptr.774 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.774 = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.31f = impl_witness_access constants.%Copy.impl_witness.1fe, element0 [concrete = constants.%ptr.as.Copy.impl.Op.eb7]
// CHECK:STDOUT:   %bound_method.loc20_10.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%const.6a8) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.1b4]
// CHECK:STDOUT:   %bound_method.loc20_10.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.774 = call %bound_method.loc20_10.2(%p.ref)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
