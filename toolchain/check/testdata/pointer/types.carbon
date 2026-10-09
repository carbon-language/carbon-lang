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
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.1fe: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.55f: type = pattern_type %ptr.1fe [concrete]
// CHECK:STDOUT:   %p.param_patt.46f: %pattern_type.55f = value_param_pattern [concrete]
// CHECK:STDOUT:   %p.patt.289: %pattern_type.55f = wrapper_binding_pattern p, %p.param_patt.46f [concrete]
// CHECK:STDOUT:   %.857: Core.Form = init_form %ptr.1fe [concrete]
// CHECK:STDOUT:   %return.param_patt.10d: %pattern_type.55f = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.568: %pattern_type.55f = return_slot_pattern %return.param_patt.10d, %ptr.1fe [concrete]
// CHECK:STDOUT:   %Ptr.type: type = fn_type @Ptr [concrete]
// CHECK:STDOUT:   %Ptr: %Ptr.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.20f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.f12: %ptr.as.Copy.impl.Op.type.20f = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.1fc: <witness> = impl_witness imports.%Copy.impl_witness_table.fb0, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.194: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.e42: %ptr.as.Copy.impl.Op.type.194 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.b57: %Copy.type = facet_value %ptr.1fe, (%Copy.impl_witness.1fc) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.4c0: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.b57) [concrete]
// CHECK:STDOUT:   %.b1d: type = fn_type_with_self_type %Copy.WithSelf.Op.type.4c0, %Copy.facet.b57 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.ef7: <specific function> = specific_function %ptr.as.Copy.impl.Op.e42, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %const.d85: type = const_type %i32 [concrete]
// CHECK:STDOUT:   %ptr.875: type = ptr_type %const.d85 [concrete]
// CHECK:STDOUT:   %pattern_type.ba3: type = pattern_type %ptr.875 [concrete]
// CHECK:STDOUT:   %p.param_patt.f94: %pattern_type.ba3 = value_param_pattern [concrete]
// CHECK:STDOUT:   %p.patt.4f0: %pattern_type.ba3 = wrapper_binding_pattern p, %p.param_patt.f94 [concrete]
// CHECK:STDOUT:   %.564: Core.Form = init_form %ptr.875 [concrete]
// CHECK:STDOUT:   %return.param_patt.881: %pattern_type.ba3 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.8f4: %pattern_type.ba3 = return_slot_pattern %return.param_patt.881, %ptr.875 [concrete]
// CHECK:STDOUT:   %ConstPtr.type: type = fn_type @ConstPtr [concrete]
// CHECK:STDOUT:   %ConstPtr: %ConstPtr.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.b61: <witness> = impl_witness imports.%Copy.impl_witness_table.fb0, @ptr.as.Copy.impl(%const.d85) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.bb6: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%const.d85) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.0ee: %ptr.as.Copy.impl.Op.type.bb6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.fe5: %Copy.type = facet_value %ptr.875, (%Copy.impl_witness.b61) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.299: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.fe5) [concrete]
// CHECK:STDOUT:   %.46c: type = fn_type_with_self_type %Copy.WithSelf.Op.type.299, %Copy.facet.fe5 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.f13: <specific function> = specific_function %ptr.as.Copy.impl.Op.0ee, @ptr.as.Copy.impl.Op(%const.d85) [concrete]
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
// CHECK:STDOUT:     %p.param_patt: %pattern_type.55f = value_param_pattern [concrete = constants.%p.param_patt.46f]
// CHECK:STDOUT:     %p.patt: %pattern_type.55f = wrapper_binding_pattern p, %p.param_patt [concrete = constants.%p.patt.289]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.55f = out_param_pattern [concrete = constants.%return.param_patt.10d]
// CHECK:STDOUT:     %return.patt: %pattern_type.55f = return_slot_pattern %return.param_patt, %ptr.loc15_23 [concrete = constants.%return.patt.568]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc15_20: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc15_23: type = ptr_type %i32.loc15_20 [concrete = constants.%ptr.1fe]
// CHECK:STDOUT:     %.loc15_23: Core.Form = init_form %ptr.loc15_23 [concrete = constants.%.857]
// CHECK:STDOUT:     %p.param: %ptr.1fe = value_param call_param0
// CHECK:STDOUT:     %.loc15_14: type = splice_block %ptr.loc15_14 [concrete = constants.%ptr.1fe] {
// CHECK:STDOUT:       %i32.loc15_11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %ptr.loc15_14: type = ptr_type %i32.loc15_11 [concrete = constants.%ptr.1fe]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.1fe = wrapper_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.1fe = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.1fe = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConstPtr.decl: %ConstPtr.type = fn_decl @ConstPtr [concrete = constants.%ConstPtr] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.ba3 = value_param_pattern [concrete = constants.%p.param_patt.f94]
// CHECK:STDOUT:     %p.patt: %pattern_type.ba3 = wrapper_binding_pattern p, %p.param_patt [concrete = constants.%p.patt.4f0]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.ba3 = out_param_pattern [concrete = constants.%return.param_patt.881]
// CHECK:STDOUT:     %return.patt: %pattern_type.ba3 = return_slot_pattern %return.param_patt, %ptr.loc19_42 [concrete = constants.%return.patt.8f4]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc19_38: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %const.loc19_32: type = const_type %i32.loc19_38 [concrete = constants.%const.d85]
// CHECK:STDOUT:     %ptr.loc19_42: type = ptr_type %const.loc19_32 [concrete = constants.%ptr.875]
// CHECK:STDOUT:     %.loc19_42: Core.Form = init_form %ptr.loc19_42 [concrete = constants.%.564]
// CHECK:STDOUT:     %p.param: %ptr.875 = value_param call_param0
// CHECK:STDOUT:     %.loc19_25: type = splice_block %ptr.loc19_25 [concrete = constants.%ptr.875] {
// CHECK:STDOUT:       %i32.loc19_22: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %const.loc19_16: type = const_type %i32.loc19_22 [concrete = constants.%const.d85]
// CHECK:STDOUT:       %ptr.loc19_25: type = ptr_type %const.loc19_16 [concrete = constants.%ptr.875]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.875 = wrapper_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.875 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.875 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Ptr(%p.param: %ptr.1fe) -> out %return.param: %ptr.1fe {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.1fe = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.b1d = impl_witness_access constants.%Copy.impl_witness.1fc, element0 [concrete = constants.%ptr.as.Copy.impl.Op.e42]
// CHECK:STDOUT:   %bound_method.loc16_10.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.ef7]
// CHECK:STDOUT:   %bound_method.loc16_10.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.1fe = call %bound_method.loc16_10.2(%p.ref)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstPtr(%p.param: %ptr.875) -> out %return.param: %ptr.875 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.875 = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.46c = impl_witness_access constants.%Copy.impl_witness.b61, element0 [concrete = constants.%ptr.as.Copy.impl.Op.0ee]
// CHECK:STDOUT:   %bound_method.loc20_10.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%const.d85) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.f13]
// CHECK:STDOUT:   %bound_method.loc20_10.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.875 = call %bound_method.loc20_10.2(%p.ref)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
