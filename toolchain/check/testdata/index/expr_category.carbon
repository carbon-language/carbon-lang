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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/index/expr_category.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/index/expr_category.carbon

fn F() -> array(i32, 3);

fn G(unused b: array(i32, 3)) {
  var a: array(i32, 3) = (1, 2, 3);

  // Indexing a durable array reference gives a durable reference.
  var unused pa: i32* = &a[0];
  a[0] = 4;
}

fn ValueBinding(b: array(i32, 3)) {
  var a: array(i32, 3) = (1, 2, 3);

  // Index but don't do anything else so we can check that a value binding is
  // produced when appropriate.
  a[0];
  b[0];
  F()[0];
}

// CHECK:STDOUT: --- expr_category.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1ba, %i32 [concrete]
// CHECK:STDOUT:   %.2b2: Core.Form = init_form %array_type [concrete]
// CHECK:STDOUT:   %pattern_type.771: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %return.param_patt.8a2: %pattern_type.771 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.2f6: %pattern_type.771 = return_slot_pattern %return.param_patt.8a2, %array_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %b.param_patt: %pattern_type.771 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt.4a0e77.1: %pattern_type.771 = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.771 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.771 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.37f: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.2d5: %tuple.type.37f = tuple_value (%int_1.5b8, %int_2.ecc, %int_3.1ba) [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a23: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.740: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.1c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.740, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.953: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.3cb: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.295: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.485: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.763: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.410: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_1.0c6, %int_2.295, %int_3.410) [concrete]
// CHECK:STDOUT:   %ptr.d08: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.f00: type = pattern_type %ptr.d08 [concrete]
// CHECK:STDOUT:   %pa.patt: %pattern_type.f00 = ref_binding_pattern pa [concrete]
// CHECK:STDOUT:   %pa.var_patt: %pattern_type.f00 = var_pattern %pa.patt [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7b4: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.6e0: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.3c0: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.ff5: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ce9: %ptr.as.Copy.impl.Op.type.ff5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.ce1: <witness> = impl_witness imports.%Copy.impl_witness_table.4f1, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.c1c: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.d1c: %ptr.as.Copy.impl.Op.type.c1c = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.d08, (%Copy.impl_witness.ce1) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f9d: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.ac9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f9d, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.d1c, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aad: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.b7a: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.4eb: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.1: type = fn_type @Destroy.Op.loc21 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.1: %Destroy.Op.type.1d8f74.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc18_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
// CHECK:STDOUT:   %b.patt.4a0e77.2: %pattern_type.771 = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:   %ValueBinding.type: type = fn_type @ValueBinding [concrete]
// CHECK:STDOUT:   %ValueBinding: %ValueBinding.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.e5f: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.ff5) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.ce9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.4f1 = impl_witness_table (%Core.import_ref.e5f), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .ValueBinding = %ValueBinding.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.771 = out_param_pattern [concrete = constants.%return.param_patt.8a2]
// CHECK:STDOUT:     %return.patt: %pattern_type.771 = return_slot_pattern %return.param_patt, %array_type [concrete = constants.%return.patt.2f6]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %i32 [concrete = constants.%array_type]
// CHECK:STDOUT:     %.loc15: Core.Form = init_form %array_type [concrete = constants.%.2b2]
// CHECK:STDOUT:     %return.param: ref %array_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %array_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %b.param_patt: %pattern_type.771 = value_param_pattern [concrete = constants.%b.param_patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.771 = at_binding_pattern b, %b.param_patt [concrete = constants.%b.patt.4a0e77.1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: %array_type = value_param call_param0
// CHECK:STDOUT:     %.loc17: type = splice_block %array_type.loc17 [concrete = constants.%array_type] {
// CHECK:STDOUT:       %i32.loc17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %int_3.loc17: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:       %array_type.loc17: type = array_type %int_3.loc17, %i32.loc17 [concrete = constants.%array_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %array_type = value_binding b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ValueBinding.decl: %ValueBinding.type = fn_decl @ValueBinding [concrete = constants.%ValueBinding] {
// CHECK:STDOUT:     %b.param_patt: %pattern_type.771 = value_param_pattern [concrete = constants.%b.param_patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.771 = at_binding_pattern b, %b.param_patt [concrete = constants.%b.patt.4a0e77.2]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: %array_type = value_param call_param0
// CHECK:STDOUT:     %.loc25: type = splice_block %array_type.loc25 [concrete = constants.%array_type] {
// CHECK:STDOUT:       %i32.loc25: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %int_3.loc25: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:       %array_type.loc25: type = array_type %int_3.loc25, %i32.loc25 [concrete = constants.%array_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %array_type = value_binding b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %array_type;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%b.param: %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.771 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.771 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc18_27: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc18_30: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3.loc18_33: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc18_34.1: %tuple.type.37f = tuple_literal (%int_1.loc18_27, %int_2.loc18_30, %int_3.loc18_33) [concrete = constants.%tuple.2d5]
// CHECK:STDOUT:   %impl.elem0.loc18_34.1: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc18_34.1: <bound method> = bound_method %int_1.loc18_27, %impl.elem0.loc18_34.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:   %specific_fn.loc18_34.1: <specific function> = specific_function %impl.elem0.loc18_34.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_34.2: <bound method> = bound_method %int_1.loc18_27, %specific_fn.loc18_34.1 [concrete = constants.%bound_method.953]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_34.1: init %i32 = call %bound_method.loc18_34.2(%int_1.loc18_27) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc18_34.2: init %i32 = converted %int_1.loc18_27, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_34.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %int_0.loc18: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc18_34.3: ref %i32 = array_index %a.var, %int_0.loc18
// CHECK:STDOUT:   %.loc18_34.4: init %i32 to %.loc18_34.3 = in_place_init %.loc18_34.2 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %impl.elem0.loc18_34.2: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc18_34.3: <bound method> = bound_method %int_2.loc18_30, %impl.elem0.loc18_34.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3]
// CHECK:STDOUT:   %specific_fn.loc18_34.2: <specific function> = specific_function %impl.elem0.loc18_34.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_34.4: <bound method> = bound_method %int_2.loc18_30, %specific_fn.loc18_34.2 [concrete = constants.%bound_method.3cb]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_34.2: init %i32 = call %bound_method.loc18_34.4(%int_2.loc18_30) [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc18_34.5: init %i32 = converted %int_2.loc18_30, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_34.2 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %int_1.loc18_34: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc18_34.6: ref %i32 = array_index %a.var, %int_1.loc18_34
// CHECK:STDOUT:   %.loc18_34.7: init %i32 to %.loc18_34.6 = in_place_init %.loc18_34.5 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %impl.elem0.loc18_34.3: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc18_34.5: <bound method> = bound_method %int_3.loc18_33, %impl.elem0.loc18_34.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.485]
// CHECK:STDOUT:   %specific_fn.loc18_34.3: <specific function> = specific_function %impl.elem0.loc18_34.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_34.6: <bound method> = bound_method %int_3.loc18_33, %specific_fn.loc18_34.3 [concrete = constants.%bound_method.763]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_34.3: init %i32 = call %bound_method.loc18_34.6(%int_3.loc18_33) [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %.loc18_34.8: init %i32 = converted %int_3.loc18_33, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_34.3 [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %int_2.loc18_34: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc18_34.9: ref %i32 = array_index %a.var, %int_2.loc18_34
// CHECK:STDOUT:   %.loc18_34.10: init %i32 to %.loc18_34.9 = in_place_init %.loc18_34.8 [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %.loc18_34.11: init %array_type to %a.var = array_init (%.loc18_34.4, %.loc18_34.7, %.loc18_34.10) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc18_3: init %array_type = converted %.loc18_34.1, %.loc18_34.11 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc18_3
// CHECK:STDOUT:   %.loc18_22: type = splice_block %array_type.loc18 [concrete = constants.%array_type] {
// CHECK:STDOUT:     %i32.loc18: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_3.loc18_21: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type.loc18: type = array_type %int_3.loc18_21, %i32.loc18 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pa.patt: %pattern_type.f00 = ref_binding_pattern pa [concrete = constants.%pa.patt]
// CHECK:STDOUT:     %pa.var_patt: %pattern_type.f00 = var_pattern %pa.patt [concrete = constants.%pa.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pa.var: ref %ptr.d08 = var %pa.var_patt
// CHECK:STDOUT:   %a.ref.loc21: ref %array_type = name_ref a, %a
// CHECK:STDOUT:   %int_0.loc21: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc21_28: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc21_28.1: <bound method> = bound_method %int_0.loc21, %impl.elem0.loc21_28 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7b4]
// CHECK:STDOUT:   %specific_fn.loc21_28: <specific function> = specific_function %impl.elem0.loc21_28, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_28.2: <bound method> = bound_method %int_0.loc21, %specific_fn.loc21_28 [concrete = constants.%bound_method.6e0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %i32 = call %bound_method.loc21_28.2(%int_0.loc21) [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc21_28.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc21_28.2: %i32 = converted %int_0.loc21, %.loc21_28.1 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc21_29: ref %i32 = array_index %a.ref.loc21, %.loc21_28.2
// CHECK:STDOUT:   %addr: %ptr.d08 = addr_of %.loc21_29
// CHECK:STDOUT:   %impl.elem0.loc21_25: %.ac9 = impl_witness_access constants.%Copy.impl_witness.ce1, element0 [concrete = constants.%ptr.as.Copy.impl.Op.d1c]
// CHECK:STDOUT:   %bound_method.loc21_25.1: <bound method> = bound_method %addr, %impl.elem0.loc21_25
// CHECK:STDOUT:   %specific_fn.loc21_25: <specific function> = specific_function %impl.elem0.loc21_25, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_25.2: <bound method> = bound_method %addr, %specific_fn.loc21_25
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.d08 = call %bound_method.loc21_25.2(%addr)
// CHECK:STDOUT:   assign %pa.var, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc21_21: type = splice_block %ptr [concrete = constants.%ptr.d08] {
// CHECK:STDOUT:     %i32.loc21: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %i32.loc21 [concrete = constants.%ptr.d08]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pa: ref %ptr.d08 = ref_binding pa, %pa.var
// CHECK:STDOUT:   %a.ref.loc22: ref %array_type = name_ref a, %a
// CHECK:STDOUT:   %int_0.loc22: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc22_5: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %int_0.loc22, %impl.elem0.loc22_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7b4]
// CHECK:STDOUT:   %specific_fn.loc22_5: <specific function> = specific_function %impl.elem0.loc22_5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %int_0.loc22, %specific_fn.loc22_5 [concrete = constants.%bound_method.6e0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_5: init %i32 = call %bound_method.loc22_5.2(%int_0.loc22) [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc22_5.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_5 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc22_5.2: %i32 = converted %int_0.loc22, %.loc22_5.1 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc22_6: ref %i32 = array_index %a.ref.loc22, %.loc22_5.2
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc22_8: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc22_8.1: <bound method> = bound_method %int_4, %impl.elem0.loc22_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aad]
// CHECK:STDOUT:   %specific_fn.loc22_8: <specific function> = specific_function %impl.elem0.loc22_8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_8.2: <bound method> = bound_method %int_4, %specific_fn.loc22_8 [concrete = constants.%bound_method.b7a]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_8: init %i32 = call %bound_method.loc22_8.2(%int_4) [concrete = constants.%int_4.4eb]
// CHECK:STDOUT:   %.loc22_8: init %i32 = converted %int_4, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_8 [concrete = constants.%int_4.4eb]
// CHECK:STDOUT:   assign %.loc22_6, %.loc22_8
// CHECK:STDOUT:   %Destroy.Op.bound.loc21: <bound method> = bound_method %pa.var, constants.%Destroy.Op.1a2547.1
// CHECK:STDOUT:   %Destroy.Op.call.loc21: init %empty_tuple.type = call %Destroy.Op.bound.loc21(%pa.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc18: <bound method> = bound_method %a.var, constants.%Destroy.Op.1a2547.4
// CHECK:STDOUT:   %Destroy.Op.call.loc18: init %empty_tuple.type = call %Destroy.Op.bound.loc18(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc21(%self.param: ref %ptr.d08) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc18_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc18_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc18_3.3(%self.param: ref %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ValueBinding(%b.param: %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.771 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.771 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc26_27: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc26_30: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3.loc26_33: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc26_34.1: %tuple.type.37f = tuple_literal (%int_1.loc26_27, %int_2.loc26_30, %int_3.loc26_33) [concrete = constants.%tuple.2d5]
// CHECK:STDOUT:   %impl.elem0.loc26_34.1: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc26_34.1: <bound method> = bound_method %int_1.loc26_27, %impl.elem0.loc26_34.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:   %specific_fn.loc26_34.1: <specific function> = specific_function %impl.elem0.loc26_34.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26_34.2: <bound method> = bound_method %int_1.loc26_27, %specific_fn.loc26_34.1 [concrete = constants.%bound_method.953]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc26_34.1: init %i32 = call %bound_method.loc26_34.2(%int_1.loc26_27) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc26_34.2: init %i32 = converted %int_1.loc26_27, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc26_34.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %int_0.loc26: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc26_34.3: ref %i32 = array_index %a.var, %int_0.loc26
// CHECK:STDOUT:   %.loc26_34.4: init %i32 to %.loc26_34.3 = in_place_init %.loc26_34.2 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %impl.elem0.loc26_34.2: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc26_34.3: <bound method> = bound_method %int_2.loc26_30, %impl.elem0.loc26_34.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3]
// CHECK:STDOUT:   %specific_fn.loc26_34.2: <specific function> = specific_function %impl.elem0.loc26_34.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26_34.4: <bound method> = bound_method %int_2.loc26_30, %specific_fn.loc26_34.2 [concrete = constants.%bound_method.3cb]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc26_34.2: init %i32 = call %bound_method.loc26_34.4(%int_2.loc26_30) [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc26_34.5: init %i32 = converted %int_2.loc26_30, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc26_34.2 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %int_1.loc26_34: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc26_34.6: ref %i32 = array_index %a.var, %int_1.loc26_34
// CHECK:STDOUT:   %.loc26_34.7: init %i32 to %.loc26_34.6 = in_place_init %.loc26_34.5 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %impl.elem0.loc26_34.3: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc26_34.5: <bound method> = bound_method %int_3.loc26_33, %impl.elem0.loc26_34.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.485]
// CHECK:STDOUT:   %specific_fn.loc26_34.3: <specific function> = specific_function %impl.elem0.loc26_34.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26_34.6: <bound method> = bound_method %int_3.loc26_33, %specific_fn.loc26_34.3 [concrete = constants.%bound_method.763]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc26_34.3: init %i32 = call %bound_method.loc26_34.6(%int_3.loc26_33) [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %.loc26_34.8: init %i32 = converted %int_3.loc26_33, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc26_34.3 [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %int_2.loc26_34: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc26_34.9: ref %i32 = array_index %a.var, %int_2.loc26_34
// CHECK:STDOUT:   %.loc26_34.10: init %i32 to %.loc26_34.9 = in_place_init %.loc26_34.8 [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %.loc26_34.11: init %array_type to %a.var = array_init (%.loc26_34.4, %.loc26_34.7, %.loc26_34.10) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc26_3: init %array_type = converted %.loc26_34.1, %.loc26_34.11 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc26_3
// CHECK:STDOUT:   %.loc26_22: type = splice_block %array_type.loc26 [concrete = constants.%array_type] {
// CHECK:STDOUT:     %i32.loc26: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_3.loc26_21: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type.loc26: type = array_type %int_3.loc26_21, %i32.loc26 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref: ref %array_type = name_ref a, %a
// CHECK:STDOUT:   %int_0.loc30: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc30: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc30_5.1: <bound method> = bound_method %int_0.loc30, %impl.elem0.loc30 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7b4]
// CHECK:STDOUT:   %specific_fn.loc30: <specific function> = specific_function %impl.elem0.loc30, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc30_5.2: <bound method> = bound_method %int_0.loc30, %specific_fn.loc30 [concrete = constants.%bound_method.6e0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc30: init %i32 = call %bound_method.loc30_5.2(%int_0.loc30) [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc30_5.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc30 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc30_5.2: %i32 = converted %int_0.loc30, %.loc30_5.1 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc30_6: ref %i32 = array_index %a.ref, %.loc30_5.2
// CHECK:STDOUT:   %b.ref: %array_type = name_ref b, %b
// CHECK:STDOUT:   %int_0.loc31: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc31: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc31_5.1: <bound method> = bound_method %int_0.loc31, %impl.elem0.loc31 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7b4]
// CHECK:STDOUT:   %specific_fn.loc31: <specific function> = specific_function %impl.elem0.loc31, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc31_5.2: <bound method> = bound_method %int_0.loc31, %specific_fn.loc31 [concrete = constants.%bound_method.6e0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc31: init %i32 = call %bound_method.loc31_5.2(%int_0.loc31) [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc31_5.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc31 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc31_5.2: %i32 = converted %int_0.loc31, %.loc31_5.1 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc31_6.1: ref %array_type = value_as_ref %b.ref
// CHECK:STDOUT:   %.loc31_6.2: ref %i32 = array_index %.loc31_6.1, %.loc31_5.2
// CHECK:STDOUT:   %.loc31_6.3: %i32 = acquire_value %.loc31_6.2
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc32_5.1: ref %array_type = temporary_storage
// CHECK:STDOUT:   %F.call: init %array_type to %.loc32_5.1 = call %F.ref()
// CHECK:STDOUT:   %int_0.loc32: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc32_5.2: ref %array_type = temporary %.loc32_5.1, %F.call
// CHECK:STDOUT:   %impl.elem0.loc32: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc32_7.1: <bound method> = bound_method %int_0.loc32, %impl.elem0.loc32 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7b4]
// CHECK:STDOUT:   %specific_fn.loc32: <specific function> = specific_function %impl.elem0.loc32, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc32_7.2: <bound method> = bound_method %int_0.loc32, %specific_fn.loc32 [concrete = constants.%bound_method.6e0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32: init %i32 = call %bound_method.loc32_7.2(%int_0.loc32) [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc32_7.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc32_7.2: %i32 = converted %int_0.loc32, %.loc32_7.1 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc32_8.1: ref %i32 = array_index %.loc32_5.2, %.loc32_7.2
// CHECK:STDOUT:   %.loc32_8.2: %i32 = acquire_value %.loc32_8.1
// CHECK:STDOUT:   %Destroy.Op.bound.loc32: <bound method> = bound_method %.loc32_5.2, constants.%Destroy.Op.1a2547.4
// CHECK:STDOUT:   %Destroy.Op.call.loc32: init %empty_tuple.type = call %Destroy.Op.bound.loc32(%.loc32_5.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc26: <bound method> = bound_method %a.var, constants.%Destroy.Op.1a2547.4
// CHECK:STDOUT:   %Destroy.Op.call.loc26: init %empty_tuple.type = call %Destroy.Op.bound.loc26(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
