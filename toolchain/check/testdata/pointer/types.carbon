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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.a17: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %ptr.a17 [concrete]
// CHECK:STDOUT:   %p.param_patt.3c8: %pattern_type.7dc = value_param_pattern [concrete]
// CHECK:STDOUT:   %p.patt.09a: %pattern_type.7dc = wrapper_binding_pattern p, %p.param_patt.3c8 [concrete]
// CHECK:STDOUT:   %.43c: Core.Form = init_form %ptr.a17 [concrete]
// CHECK:STDOUT:   %return.param_patt.6e5: %pattern_type.7dc = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.1b4: %pattern_type.7dc = return_slot_pattern %return.param_patt.6e5, %ptr.a17 [concrete]
// CHECK:STDOUT:   %Ptr.type: type = fn_type @Ptr [concrete]
// CHECK:STDOUT:   %Ptr: %Ptr.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.20f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.f12: %ptr.as.Copy.impl.Op.type.20f = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.83a: <witness> = impl_witness imports.%Copy.impl_witness_table.fb0, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %Copy.facet.652: %Copy.type = facet_value %ptr.a17, (%Copy.impl_witness.83a) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.9a2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.652) [concrete]
// CHECK:STDOUT:   %.309: type = fn_type_with_self_type %Copy.WithSelf.Op.type.9a2, %Copy.facet.652 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.dfc: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.81f: %ptr.as.Copy.impl.Op.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.9a6: <specific function> = specific_function %ptr.as.Copy.impl.Op.81f, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %const.003: type = const_type %i32 [concrete]
// CHECK:STDOUT:   %ptr.75b: type = ptr_type %const.003 [concrete]
// CHECK:STDOUT:   %pattern_type.6f6: type = pattern_type %ptr.75b [concrete]
// CHECK:STDOUT:   %p.param_patt.ab8: %pattern_type.6f6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %p.patt.747: %pattern_type.6f6 = wrapper_binding_pattern p, %p.param_patt.ab8 [concrete]
// CHECK:STDOUT:   %.ceb: Core.Form = init_form %ptr.75b [concrete]
// CHECK:STDOUT:   %return.param_patt.9af: %pattern_type.6f6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.a78: %pattern_type.6f6 = return_slot_pattern %return.param_patt.9af, %ptr.75b [concrete]
// CHECK:STDOUT:   %ConstPtr.type: type = fn_type @ConstPtr [concrete]
// CHECK:STDOUT:   %ConstPtr: %ConstPtr.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.00f: <witness> = impl_witness imports.%Copy.impl_witness_table.fb0, @ptr.as.Copy.impl(%const.003) [concrete]
// CHECK:STDOUT:   %Copy.facet.c90: %Copy.type = facet_value %ptr.75b, (%Copy.impl_witness.00f) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.1e3: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.c90) [concrete]
// CHECK:STDOUT:   %.fb7: type = fn_type_with_self_type %Copy.WithSelf.Op.type.1e3, %Copy.facet.c90 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.f56: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%const.003) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.a57: %ptr.as.Copy.impl.Op.type.f56 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.1ee: <specific function> = specific_function %ptr.as.Copy.impl.Op.a57, @ptr.as.Copy.impl.Op(%const.003) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.06a: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.20f) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.f12)]
// CHECK:STDOUT:   %Copy.impl_witness_table.fb0 = impl_witness_table (%Core.import_ref.06a), @ptr.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %p.param_patt: %pattern_type.7dc = value_param_pattern [concrete = constants.%p.param_patt.3c8]
// CHECK:STDOUT:     %p.patt: %pattern_type.7dc = wrapper_binding_pattern p, %p.param_patt [concrete = constants.%p.patt.09a]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7dc = out_param_pattern [concrete = constants.%return.param_patt.6e5]
// CHECK:STDOUT:     %return.patt: %pattern_type.7dc = return_slot_pattern %return.param_patt, %ptr.loc15_23 [concrete = constants.%return.patt.1b4]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc15_20: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc15_23: type = ptr_type %i32.loc15_20 [concrete = constants.%ptr.a17]
// CHECK:STDOUT:     %.loc15_23: Core.Form = init_form %ptr.loc15_23 [concrete = constants.%.43c]
// CHECK:STDOUT:     %p.param: %ptr.a17 = value_param call_param0
// CHECK:STDOUT:     %.loc15_14: type = splice_block %ptr.loc15_14 [concrete = constants.%ptr.a17] {
// CHECK:STDOUT:       %i32.loc15_11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %ptr.loc15_14: type = ptr_type %i32.loc15_11 [concrete = constants.%ptr.a17]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.a17 = wrapper_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.a17 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.a17 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConstPtr.decl: %ConstPtr.type = fn_decl @ConstPtr [concrete = constants.%ConstPtr] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.6f6 = value_param_pattern [concrete = constants.%p.param_patt.ab8]
// CHECK:STDOUT:     %p.patt: %pattern_type.6f6 = wrapper_binding_pattern p, %p.param_patt [concrete = constants.%p.patt.747]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6f6 = out_param_pattern [concrete = constants.%return.param_patt.9af]
// CHECK:STDOUT:     %return.patt: %pattern_type.6f6 = return_slot_pattern %return.param_patt, %ptr.loc19_42 [concrete = constants.%return.patt.a78]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc19_38: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %const.loc19_32: type = const_type %i32.loc19_38 [concrete = constants.%const.003]
// CHECK:STDOUT:     %ptr.loc19_42: type = ptr_type %const.loc19_32 [concrete = constants.%ptr.75b]
// CHECK:STDOUT:     %.loc19_42: Core.Form = init_form %ptr.loc19_42 [concrete = constants.%.ceb]
// CHECK:STDOUT:     %p.param: %ptr.75b = value_param call_param0
// CHECK:STDOUT:     %.loc19_25: type = splice_block %ptr.loc19_25 [concrete = constants.%ptr.75b] {
// CHECK:STDOUT:       %i32.loc19_22: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %const.loc19_16: type = const_type %i32.loc19_22 [concrete = constants.%const.003]
// CHECK:STDOUT:       %ptr.loc19_25: type = ptr_type %const.loc19_16 [concrete = constants.%ptr.75b]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.75b = wrapper_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.75b = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.75b = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Ptr(%p.param: %ptr.a17) -> out %return.param: %ptr.a17 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.a17 = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.309 = impl_witness_access constants.%Copy.impl_witness.83a, element0 [concrete = constants.%ptr.as.Copy.impl.Op.81f]
// CHECK:STDOUT:   %bound_method.loc16_10.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.9a6]
// CHECK:STDOUT:   %bound_method.loc16_10.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.a17 = call %bound_method.loc16_10.2(%p.ref)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstPtr(%p.param: %ptr.75b) -> out %return.param: %ptr.75b {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.75b = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.fb7 = impl_witness_access constants.%Copy.impl_witness.00f, element0 [concrete = constants.%ptr.as.Copy.impl.Op.a57]
// CHECK:STDOUT:   %bound_method.loc20_10.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%const.003) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.1ee]
// CHECK:STDOUT:   %bound_method.loc20_10.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.75b = call %bound_method.loc20_10.2(%p.ref)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
