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
// CHECK:STDOUT:   %ptr.49e: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.50c: type = pattern_type %ptr.49e [concrete]
// CHECK:STDOUT:   %p.param_patt.c1c: %pattern_type.50c = value_param_pattern [concrete]
// CHECK:STDOUT:   %p.patt.055: %pattern_type.50c = wrapper_binding_pattern p, %p.param_patt.c1c [concrete]
// CHECK:STDOUT:   %.2a0: Core.Form = init_form %ptr.49e [concrete]
// CHECK:STDOUT:   %return.param_patt.c11: %pattern_type.50c = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.9d1: %pattern_type.50c = return_slot_pattern %return.param_patt.c11, %ptr.49e [concrete]
// CHECK:STDOUT:   %Ptr.type: type = fn_type @Ptr [concrete]
// CHECK:STDOUT:   %Ptr: %Ptr.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.ff5: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ce9: %ptr.as.Copy.impl.Op.type.ff5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.dd4: <witness> = impl_witness imports.%Copy.impl_witness_table.fb0, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.d83: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.22e: %ptr.as.Copy.impl.Op.type.d83 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.759: %Copy.type = facet_value %ptr.49e, (%Copy.impl_witness.dd4) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.037: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.759) [concrete]
// CHECK:STDOUT:   %.29f: type = fn_type_with_self_type %Copy.WithSelf.Op.type.037, %Copy.facet.759 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.f8b: <specific function> = specific_function %ptr.as.Copy.impl.Op.22e, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %const.ed3: type = const_type %i32 [concrete]
// CHECK:STDOUT:   %ptr.e7c: type = ptr_type %const.ed3 [concrete]
// CHECK:STDOUT:   %pattern_type.aa2: type = pattern_type %ptr.e7c [concrete]
// CHECK:STDOUT:   %p.param_patt.f82: %pattern_type.aa2 = value_param_pattern [concrete]
// CHECK:STDOUT:   %p.patt.fd3: %pattern_type.aa2 = wrapper_binding_pattern p, %p.param_patt.f82 [concrete]
// CHECK:STDOUT:   %.43d: Core.Form = init_form %ptr.e7c [concrete]
// CHECK:STDOUT:   %return.param_patt.abf: %pattern_type.aa2 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.8a5: %pattern_type.aa2 = return_slot_pattern %return.param_patt.abf, %ptr.e7c [concrete]
// CHECK:STDOUT:   %ConstPtr.type: type = fn_type @ConstPtr [concrete]
// CHECK:STDOUT:   %ConstPtr: %ConstPtr.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.2c5: <witness> = impl_witness imports.%Copy.impl_witness_table.fb0, @ptr.as.Copy.impl(%const.ed3) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.a2f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%const.ed3) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.1ba: %ptr.as.Copy.impl.Op.type.a2f = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.4fa: %Copy.type = facet_value %ptr.e7c, (%Copy.impl_witness.2c5) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.137: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.4fa) [concrete]
// CHECK:STDOUT:   %.38b: type = fn_type_with_self_type %Copy.WithSelf.Op.type.137, %Copy.facet.4fa [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.75c: <specific function> = specific_function %ptr.as.Copy.impl.Op.1ba, @ptr.as.Copy.impl.Op(%const.ed3) [concrete]
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
// CHECK:STDOUT:   %Copy.impl_witness_table.fb0 = impl_witness_table (%Core.import_ref.e5f), @ptr.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %p.param_patt: %pattern_type.50c = value_param_pattern [concrete = constants.%p.param_patt.c1c]
// CHECK:STDOUT:     %p.patt: %pattern_type.50c = wrapper_binding_pattern p, %p.param_patt [concrete = constants.%p.patt.055]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.50c = out_param_pattern [concrete = constants.%return.param_patt.c11]
// CHECK:STDOUT:     %return.patt: %pattern_type.50c = return_slot_pattern %return.param_patt, %ptr.loc15_23 [concrete = constants.%return.patt.9d1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc15_20: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc15_23: type = ptr_type %i32.loc15_20 [concrete = constants.%ptr.49e]
// CHECK:STDOUT:     %.loc15_23: Core.Form = init_form %ptr.loc15_23 [concrete = constants.%.2a0]
// CHECK:STDOUT:     %p.param: %ptr.49e = value_param call_param0
// CHECK:STDOUT:     %.loc15_14: type = splice_block %ptr.loc15_14 [concrete = constants.%ptr.49e] {
// CHECK:STDOUT:       %i32.loc15_11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %ptr.loc15_14: type = ptr_type %i32.loc15_11 [concrete = constants.%ptr.49e]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.49e = wrapper_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.49e = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.49e = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConstPtr.decl: %ConstPtr.type = fn_decl @ConstPtr [concrete = constants.%ConstPtr] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.aa2 = value_param_pattern [concrete = constants.%p.param_patt.f82]
// CHECK:STDOUT:     %p.patt: %pattern_type.aa2 = wrapper_binding_pattern p, %p.param_patt [concrete = constants.%p.patt.fd3]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.aa2 = out_param_pattern [concrete = constants.%return.param_patt.abf]
// CHECK:STDOUT:     %return.patt: %pattern_type.aa2 = return_slot_pattern %return.param_patt, %ptr.loc19_42 [concrete = constants.%return.patt.8a5]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc19_38: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %const.loc19_32: type = const_type %i32.loc19_38 [concrete = constants.%const.ed3]
// CHECK:STDOUT:     %ptr.loc19_42: type = ptr_type %const.loc19_32 [concrete = constants.%ptr.e7c]
// CHECK:STDOUT:     %.loc19_42: Core.Form = init_form %ptr.loc19_42 [concrete = constants.%.43d]
// CHECK:STDOUT:     %p.param: %ptr.e7c = value_param call_param0
// CHECK:STDOUT:     %.loc19_25: type = splice_block %ptr.loc19_25 [concrete = constants.%ptr.e7c] {
// CHECK:STDOUT:       %i32.loc19_22: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %const.loc19_16: type = const_type %i32.loc19_22 [concrete = constants.%const.ed3]
// CHECK:STDOUT:       %ptr.loc19_25: type = ptr_type %const.loc19_16 [concrete = constants.%ptr.e7c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.e7c = wrapper_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.e7c = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.e7c = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Ptr(%p.param: %ptr.49e) -> out %return.param: %ptr.49e {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.49e = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.29f = impl_witness_access constants.%Copy.impl_witness.dd4, element0 [concrete = constants.%ptr.as.Copy.impl.Op.22e]
// CHECK:STDOUT:   %bound_method.loc16_10.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.f8b]
// CHECK:STDOUT:   %bound_method.loc16_10.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.49e = call %bound_method.loc16_10.2(%p.ref)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstPtr(%p.param: %ptr.e7c) -> out %return.param: %ptr.e7c {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.e7c = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.38b = impl_witness_access constants.%Copy.impl_witness.2c5, element0 [concrete = constants.%ptr.as.Copy.impl.Op.1ba]
// CHECK:STDOUT:   %bound_method.loc20_10.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%const.ed3) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.75c]
// CHECK:STDOUT:   %bound_method.loc20_10.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.e7c = call %bound_method.loc20_10.2(%p.ref)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
