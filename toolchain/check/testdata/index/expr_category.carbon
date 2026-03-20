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
// CHECK:STDOUT:   %.b4b: Core.Form = init_form %array_type [concrete]
// CHECK:STDOUT:   %pattern_type.5d8: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.37f: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.2d5: %tuple.type.37f = tuple_value (%int_1.5b8, %int_2.ecc, %int_3.1ba) [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.38b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.646: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.fa7: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_1.5d2, %int_2.ef8, %int_3.822) [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.fe8: type = pattern_type %ptr.235 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.d2e: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.2d4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.74e: %ptr.as.Copy.impl.Op.type.2d4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.843: <witness> = impl_witness imports.%Copy.impl_witness_table.c3a, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.c3c: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.011: %ptr.as.Copy.impl.Op.type.c3c = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.235, (%Copy.impl_witness.843) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.e01: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.a62: type = fn_type_with_self_type %Copy.WithSelf.Op.type.e01, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.011, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f0c: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.6d7: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.1: type = fn_type @Destroy.Op.loc45 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.1: %Destroy.Op.type.bae255.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc27 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc119, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.203: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.2d4) = import_ref Core//prelude/parts/copy, loc217, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.74e)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c3a = impl_witness_table (%Core.import_ref.203), @ptr.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %return.param_patt: %pattern_type.5d8 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.5d8 = return_slot_pattern %return.param_patt, %array_type [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %i32 [concrete = constants.%array_type]
// CHECK:STDOUT:     %.loc11: Core.Form = init_form %array_type [concrete = constants.%.b4b]
// CHECK:STDOUT:     %return.param: ref %array_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %array_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %b.param_patt: %pattern_type.5d8 = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.5d8 = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: %array_type = value_param call_param0
// CHECK:STDOUT:     %.loc24: type = splice_block %array_type.loc24 [concrete = constants.%array_type] {
// CHECK:STDOUT:       %i32.loc21: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %int_3.loc23: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:       %array_type.loc24: type = array_type %int_3.loc23, %i32.loc21 [concrete = constants.%array_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %array_type = value_binding b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ValueBinding.decl: %ValueBinding.type = fn_decl @ValueBinding [concrete = constants.%ValueBinding] {
// CHECK:STDOUT:     %b.param_patt: %pattern_type.5d8 = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.5d8 = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: %array_type = value_param call_param0
// CHECK:STDOUT:     %.loc76: type = splice_block %array_type.loc76 [concrete = constants.%array_type] {
// CHECK:STDOUT:       %i32.loc73: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %int_3.loc75: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:       %array_type.loc76: type = array_type %int_3.loc75, %i32.loc73 [concrete = constants.%array_type]
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
// CHECK:STDOUT:     %a.patt: %pattern_type.5d8 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.5d8 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc38: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc40: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3.loc42: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc43.1: %tuple.type.37f = tuple_literal (%int_1.loc38, %int_2.loc40, %int_3.loc42) [concrete = constants.%tuple.2d5]
// CHECK:STDOUT:   %impl.elem0.loc43.1: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc43.1: <bound method> = bound_method %int_1.loc38, %impl.elem0.loc43.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc43.1: <specific function> = specific_function %impl.elem0.loc43.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc43.2: <bound method> = bound_method %int_1.loc38, %specific_fn.loc43.1 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc43.1: init %i32 = call %bound_method.loc43.2(%int_1.loc38) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc43.2: init %i32 = converted %int_1.loc38, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc43.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %int_0.loc43: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc43.3: ref %i32 = array_index %a.var, %int_0.loc43
// CHECK:STDOUT:   %.loc43.4: init %i32 to %.loc43.3 = in_place_init %.loc43.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc43.2: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc43.3: <bound method> = bound_method %int_2.loc40, %impl.elem0.loc43.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc43.2: <specific function> = specific_function %impl.elem0.loc43.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc43.4: <bound method> = bound_method %int_2.loc40, %specific_fn.loc43.2 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc43.2: init %i32 = call %bound_method.loc43.4(%int_2.loc40) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc43.5: init %i32 = converted %int_2.loc40, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc43.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %int_1.loc43: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc43.6: ref %i32 = array_index %a.var, %int_1.loc43
// CHECK:STDOUT:   %.loc43.7: init %i32 to %.loc43.6 = in_place_init %.loc43.5 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc43.3: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc43.5: <bound method> = bound_method %int_3.loc42, %impl.elem0.loc43.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:   %specific_fn.loc43.3: <specific function> = specific_function %impl.elem0.loc43.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc43.6: <bound method> = bound_method %int_3.loc42, %specific_fn.loc43.3 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc43.3: init %i32 = call %bound_method.loc43.6(%int_3.loc42) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc43.8: init %i32 = converted %int_3.loc42, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc43.3 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %int_2.loc43: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc43.9: ref %i32 = array_index %a.var, %int_2.loc43
// CHECK:STDOUT:   %.loc43.10: init %i32 to %.loc43.9 = in_place_init %.loc43.8 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc43.11: init %array_type to %a.var = array_init (%.loc43.4, %.loc43.7, %.loc43.10) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc27: init %array_type = converted %.loc43.1, %.loc43.11 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc27
// CHECK:STDOUT:   %.loc35: type = splice_block %array_type.loc35 [concrete = constants.%array_type] {
// CHECK:STDOUT:     %i32.loc32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_3.loc34: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type.loc35: type = array_type %int_3.loc34, %i32.loc32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pa.patt: %pattern_type.fe8 = ref_binding_pattern pa [concrete]
// CHECK:STDOUT:     %pa.var_patt: %pattern_type.fe8 = var_pattern %pa.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pa.var: ref %ptr.235 = var %pa.var_patt
// CHECK:STDOUT:   %a.ref.loc53: ref %array_type = name_ref a, %a
// CHECK:STDOUT:   %int_0.loc55: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc55: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc55.1: <bound method> = bound_method %int_0.loc55, %impl.elem0.loc55 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897]
// CHECK:STDOUT:   %specific_fn.loc55: <specific function> = specific_function %impl.elem0.loc55, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc55.2: <bound method> = bound_method %int_0.loc55, %specific_fn.loc55 [concrete = constants.%bound_method.d2e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc55: init %i32 = call %bound_method.loc55.2(%int_0.loc55) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc55.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc55 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc55.2: %i32 = converted %int_0.loc55, %.loc55.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc56: ref %i32 = array_index %a.ref.loc53, %.loc55.2
// CHECK:STDOUT:   %addr: %ptr.235 = addr_of %.loc56
// CHECK:STDOUT:   %impl.elem0.loc52: %.a62 = impl_witness_access constants.%Copy.impl_witness.843, element0 [concrete = constants.%ptr.as.Copy.impl.Op.011]
// CHECK:STDOUT:   %bound_method.loc52.1: <bound method> = bound_method %addr, %impl.elem0.loc52
// CHECK:STDOUT:   %specific_fn.loc52: <specific function> = specific_function %impl.elem0.loc52, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc52.2: <bound method> = bound_method %addr, %specific_fn.loc52
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.235 = call %bound_method.loc52.2(%addr)
// CHECK:STDOUT:   assign %pa.var, %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc50: type = splice_block %ptr [concrete = constants.%ptr.235] {
// CHECK:STDOUT:     %i32.loc49: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %i32.loc49 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pa: ref %ptr.235 = ref_binding pa, %pa.var
// CHECK:STDOUT:   %a.ref.loc58: ref %array_type = name_ref a, %a
// CHECK:STDOUT:   %int_0.loc60: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc60: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc60.1: <bound method> = bound_method %int_0.loc60, %impl.elem0.loc60 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897]
// CHECK:STDOUT:   %specific_fn.loc60: <specific function> = specific_function %impl.elem0.loc60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc60.2: <bound method> = bound_method %int_0.loc60, %specific_fn.loc60 [concrete = constants.%bound_method.d2e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc60: init %i32 = call %bound_method.loc60.2(%int_0.loc60) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc60.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc60 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc60.2: %i32 = converted %int_0.loc60, %.loc60.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc61: ref %i32 = array_index %a.ref.loc58, %.loc60.2
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc62: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc62.1: <bound method> = bound_method %int_4, %impl.elem0.loc62 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f0c]
// CHECK:STDOUT:   %specific_fn.loc62: <specific function> = specific_function %impl.elem0.loc62, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc62.2: <bound method> = bound_method %int_4, %specific_fn.loc62 [concrete = constants.%bound_method.6d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc62: init %i32 = call %bound_method.loc62.2(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc62: init %i32 = converted %int_4, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc62 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   assign %.loc61, %.loc62
// CHECK:STDOUT:   %Destroy.Op.bound.loc45: <bound method> = bound_method %pa.var, constants.%Destroy.Op.651ba6.1
// CHECK:STDOUT:   %Destroy.Op.call.loc45: init %empty_tuple.type = call %Destroy.Op.bound.loc45(%pa.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc27: <bound method> = bound_method %a.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc27: init %empty_tuple.type = call %Destroy.Op.bound.loc27(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc45(%self.param: ref %ptr.235) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc27(%self.param: ref %array_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ValueBinding(%b.param: %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.5d8 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.5d8 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc90: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc92: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3.loc94: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc95.1: %tuple.type.37f = tuple_literal (%int_1.loc90, %int_2.loc92, %int_3.loc94) [concrete = constants.%tuple.2d5]
// CHECK:STDOUT:   %impl.elem0.loc95.1: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc95.1: <bound method> = bound_method %int_1.loc90, %impl.elem0.loc95.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc95.1: <specific function> = specific_function %impl.elem0.loc95.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc95.2: <bound method> = bound_method %int_1.loc90, %specific_fn.loc95.1 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc95.1: init %i32 = call %bound_method.loc95.2(%int_1.loc90) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc95.2: init %i32 = converted %int_1.loc90, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc95.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %int_0.loc95: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc95.3: ref %i32 = array_index %a.var, %int_0.loc95
// CHECK:STDOUT:   %.loc95.4: init %i32 to %.loc95.3 = in_place_init %.loc95.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc95.2: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc95.3: <bound method> = bound_method %int_2.loc92, %impl.elem0.loc95.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc95.2: <specific function> = specific_function %impl.elem0.loc95.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc95.4: <bound method> = bound_method %int_2.loc92, %specific_fn.loc95.2 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc95.2: init %i32 = call %bound_method.loc95.4(%int_2.loc92) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc95.5: init %i32 = converted %int_2.loc92, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc95.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %int_1.loc95: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc95.6: ref %i32 = array_index %a.var, %int_1.loc95
// CHECK:STDOUT:   %.loc95.7: init %i32 to %.loc95.6 = in_place_init %.loc95.5 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc95.3: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc95.5: <bound method> = bound_method %int_3.loc94, %impl.elem0.loc95.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:   %specific_fn.loc95.3: <specific function> = specific_function %impl.elem0.loc95.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc95.6: <bound method> = bound_method %int_3.loc94, %specific_fn.loc95.3 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc95.3: init %i32 = call %bound_method.loc95.6(%int_3.loc94) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc95.8: init %i32 = converted %int_3.loc94, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc95.3 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %int_2.loc95: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc95.9: ref %i32 = array_index %a.var, %int_2.loc95
// CHECK:STDOUT:   %.loc95.10: init %i32 to %.loc95.9 = in_place_init %.loc95.8 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc95.11: init %array_type to %a.var = array_init (%.loc95.4, %.loc95.7, %.loc95.10) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc79: init %array_type = converted %.loc95.1, %.loc95.11 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc79
// CHECK:STDOUT:   %.loc87: type = splice_block %array_type.loc87 [concrete = constants.%array_type] {
// CHECK:STDOUT:     %i32.loc84: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_3.loc86: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type.loc87: type = array_type %int_3.loc86, %i32.loc84 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   %a.ref: ref %array_type = name_ref a, %a
// CHECK:STDOUT:   %int_0.loc99: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc99: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc99.1: <bound method> = bound_method %int_0.loc99, %impl.elem0.loc99 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897]
// CHECK:STDOUT:   %specific_fn.loc99: <specific function> = specific_function %impl.elem0.loc99, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc99.2: <bound method> = bound_method %int_0.loc99, %specific_fn.loc99 [concrete = constants.%bound_method.d2e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc99: init %i32 = call %bound_method.loc99.2(%int_0.loc99) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc99.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc99 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc99.2: %i32 = converted %int_0.loc99, %.loc99.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc100: ref %i32 = array_index %a.ref, %.loc99.2
// CHECK:STDOUT:   %b.ref: %array_type = name_ref b, %b
// CHECK:STDOUT:   %int_0.loc104: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc104: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc104.1: <bound method> = bound_method %int_0.loc104, %impl.elem0.loc104 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897]
// CHECK:STDOUT:   %specific_fn.loc104: <specific function> = specific_function %impl.elem0.loc104, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc104.2: <bound method> = bound_method %int_0.loc104, %specific_fn.loc104 [concrete = constants.%bound_method.d2e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc104: init %i32 = call %bound_method.loc104.2(%int_0.loc104) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc104.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc104 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc104.2: %i32 = converted %int_0.loc104, %.loc104.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc105.1: ref %array_type = value_as_ref %b.ref
// CHECK:STDOUT:   %.loc105.2: ref %i32 = array_index %.loc105.1, %.loc104.2
// CHECK:STDOUT:   %.loc105.3: %i32 = acquire_value %.loc105.2
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc109.1: ref %array_type = temporary_storage
// CHECK:STDOUT:   %F.call: init %array_type to %.loc109.1 = call %F.ref()
// CHECK:STDOUT:   %int_0.loc111: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc109.2: ref %array_type = temporary %.loc109.1, %F.call
// CHECK:STDOUT:   %impl.elem0.loc111: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc111.1: <bound method> = bound_method %int_0.loc111, %impl.elem0.loc111 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897]
// CHECK:STDOUT:   %specific_fn.loc111: <specific function> = specific_function %impl.elem0.loc111, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc111.2: <bound method> = bound_method %int_0.loc111, %specific_fn.loc111 [concrete = constants.%bound_method.d2e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc111: init %i32 = call %bound_method.loc111.2(%int_0.loc111) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc111.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc111 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc111.2: %i32 = converted %int_0.loc111, %.loc111.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc112.1: ref %i32 = array_index %.loc109.2, %.loc111.2
// CHECK:STDOUT:   %.loc112.2: %i32 = acquire_value %.loc112.1
// CHECK:STDOUT:   %Destroy.Op.bound.loc109: <bound method> = bound_method %.loc109.2, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc109: init %empty_tuple.type = call %Destroy.Op.bound.loc109(%.loc109.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc79: <bound method> = bound_method %a.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc79: init %empty_tuple.type = call %Destroy.Op.bound.loc79(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
