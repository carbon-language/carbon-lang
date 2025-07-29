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

fn G(b: array(i32, 3)) {
  var a: array(i32, 3) = (1, 2, 3);

  // Indexing a durable array reference gives a durable reference.
  var pa: i32* = &a[0];
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
// CHECK:STDOUT:   %pattern_type.5d8: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.f01: type = ptr_type %array_type [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f06: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.956: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.956, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b30: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.047: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_1.5d2, %int_2.ef8, %int_3.822) [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.fe8: type = pattern_type %ptr.235 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d04: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b6e: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ac3: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.1da: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.ba2: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%ptr.235) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.649: %T.as.Destroy.impl.Op.type.ba2 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5d5: type = ptr_type %ptr.235 [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn.829: <specific function> = specific_function %T.as.Destroy.impl.Op.649, @T.as.Destroy.impl.Op(%ptr.235) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.1e4: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%array_type) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.f0b: %T.as.Destroy.impl.Op.type.1e4 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn.c51: <specific function> = specific_function %T.as.Destroy.impl.Op.f0b, @T.as.Destroy.impl.Op(%array_type) [concrete]
// CHECK:STDOUT:   %ValueBinding.type: type = fn_type @ValueBinding [concrete]
// CHECK:STDOUT:   %ValueBinding: %ValueBinding.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %return.patt: %pattern_type.5d8 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.5d8 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %i32 [concrete = constants.%array_type]
// CHECK:STDOUT:     %return.param: ref %array_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %array_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %b.patt: %pattern_type.5d8 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.5d8 = value_param_pattern %b.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: %array_type = value_param call_param0
// CHECK:STDOUT:     %.loc17: type = splice_block %array_type.loc17 [concrete = constants.%array_type] {
// CHECK:STDOUT:       %int_32.loc17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %int_3.loc17: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:       %array_type.loc17: type = array_type %int_3.loc17, %i32.loc17 [concrete = constants.%array_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %array_type = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ValueBinding.decl: %ValueBinding.type = fn_decl @ValueBinding [concrete = constants.%ValueBinding] {
// CHECK:STDOUT:     %b.patt: %pattern_type.5d8 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.5d8 = value_param_pattern %b.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: %array_type = value_param call_param0
// CHECK:STDOUT:     %.loc25: type = splice_block %array_type.loc25 [concrete = constants.%array_type] {
// CHECK:STDOUT:       %int_32.loc25: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc25: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %int_3.loc25: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:       %array_type.loc25: type = array_type %int_3.loc25, %i32.loc25 [concrete = constants.%array_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %array_type = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return.param: %array_type;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%b.param: %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.5d8 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.5d8 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc18_27: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc18_30: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3.loc18_33: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc18_34.1: %tuple.type = tuple_literal (%int_1.loc18_27, %int_2.loc18_30, %int_3.loc18_33)
// CHECK:STDOUT:   %impl.elem0.loc18_34.1: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc18_34.1: <bound method> = bound_method %int_1.loc18_27, %impl.elem0.loc18_34.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc18_34.1: <specific function> = specific_function %impl.elem0.loc18_34.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_34.2: <bound method> = bound_method %int_1.loc18_27, %specific_fn.loc18_34.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_34.1: init %i32 = call %bound_method.loc18_34.2(%int_1.loc18_27) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc18_34.2: init %i32 = converted %int_1.loc18_27, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_34.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %int_0.loc18: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc18_34.3: ref %i32 = array_index %a.var, %int_0.loc18
// CHECK:STDOUT:   %.loc18_34.4: init %i32 = initialize_from %.loc18_34.2 to %.loc18_34.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc18_34.2: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc18_34.3: <bound method> = bound_method %int_2.loc18_30, %impl.elem0.loc18_34.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc18_34.2: <specific function> = specific_function %impl.elem0.loc18_34.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_34.4: <bound method> = bound_method %int_2.loc18_30, %specific_fn.loc18_34.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_34.2: init %i32 = call %bound_method.loc18_34.4(%int_2.loc18_30) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc18_34.5: init %i32 = converted %int_2.loc18_30, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_34.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %int_1.loc18_34: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc18_34.6: ref %i32 = array_index %a.var, %int_1.loc18_34
// CHECK:STDOUT:   %.loc18_34.7: init %i32 = initialize_from %.loc18_34.5 to %.loc18_34.6 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc18_34.3: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc18_34.5: <bound method> = bound_method %int_3.loc18_33, %impl.elem0.loc18_34.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc18_34.3: <specific function> = specific_function %impl.elem0.loc18_34.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_34.6: <bound method> = bound_method %int_3.loc18_33, %specific_fn.loc18_34.3 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_34.3: init %i32 = call %bound_method.loc18_34.6(%int_3.loc18_33) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc18_34.8: init %i32 = converted %int_3.loc18_33, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18_34.3 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %int_2.loc18_34: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc18_34.9: ref %i32 = array_index %a.var, %int_2.loc18_34
// CHECK:STDOUT:   %.loc18_34.10: init %i32 = initialize_from %.loc18_34.8 to %.loc18_34.9 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc18_34.11: init %array_type = array_init (%.loc18_34.4, %.loc18_34.7, %.loc18_34.10) to %a.var [concrete = constants.%array]
// CHECK:STDOUT:   %.loc18_3: init %array_type = converted %.loc18_34.1, %.loc18_34.11 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc18_3
// CHECK:STDOUT:   %.loc18_22: type = splice_block %array_type.loc18 [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_3.loc18_21: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type.loc18: type = array_type %int_3.loc18_21, %i32.loc18 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pa.patt: %pattern_type.fe8 = binding_pattern pa [concrete]
// CHECK:STDOUT:     %pa.var_patt: %pattern_type.fe8 = var_pattern %pa.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pa.var: ref %ptr.235 = var %pa.var_patt
// CHECK:STDOUT:   %a.ref.loc21: ref %array_type = name_ref a, %a
// CHECK:STDOUT:   %int_0.loc21: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_32.loc21_22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc21_22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc21: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc21_21.1: <bound method> = bound_method %int_0.loc21, %impl.elem0.loc21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc21: <specific function> = specific_function %impl.elem0.loc21, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_21.2: <bound method> = bound_method %int_0.loc21, %specific_fn.loc21 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21: init %i32 = call %bound_method.loc21_21.2(%int_0.loc21) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc21_21.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc21 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc21_21.2: %i32 = converted %int_0.loc21, %.loc21_21.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc21_22: ref %i32 = array_index %a.ref.loc21, %.loc21_21.2
// CHECK:STDOUT:   %addr.loc21_18: %ptr.235 = addr_of %.loc21_22
// CHECK:STDOUT:   assign %pa.var, %addr.loc21_18
// CHECK:STDOUT:   %.loc21_14: type = splice_block %ptr [concrete = constants.%ptr.235] {
// CHECK:STDOUT:     %int_32.loc21_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc21_11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %i32.loc21_11 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pa: ref %ptr.235 = bind_name pa, %pa.var
// CHECK:STDOUT:   %a.ref.loc22: ref %array_type = name_ref a, %a
// CHECK:STDOUT:   %int_0.loc22: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_32.loc22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc22_5: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc22_5.1: <bound method> = bound_method %int_0.loc22, %impl.elem0.loc22_5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc22_5: <specific function> = specific_function %impl.elem0.loc22_5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_5.2: <bound method> = bound_method %int_0.loc22, %specific_fn.loc22_5 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_5: init %i32 = call %bound_method.loc22_5.2(%int_0.loc22) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc22_5.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_5 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc22_5.2: %i32 = converted %int_0.loc22, %.loc22_5.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc22_6: ref %i32 = array_index %a.ref.loc22, %.loc22_5.2
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc22_8: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc22_8.1: <bound method> = bound_method %int_4, %impl.elem0.loc22_8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ac3]
// CHECK:STDOUT:   %specific_fn.loc22_8: <specific function> = specific_function %impl.elem0.loc22_8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_8.2: <bound method> = bound_method %int_4, %specific_fn.loc22_8 [concrete = constants.%bound_method.1da]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_8: init %i32 = call %bound_method.loc22_8.2(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc22_8: init %i32 = converted %int_4, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22_8 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   assign %.loc22_6, %.loc22_8
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc21: <bound method> = bound_method %pa.var, constants.%T.as.Destroy.impl.Op.649
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.649, @T.as.Destroy.impl.Op(constants.%ptr.235) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn.829]
// CHECK:STDOUT:   %bound_method.loc21_3: <bound method> = bound_method %pa.var, %T.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc21_3: %ptr.5d5 = addr_of %pa.var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc21: init %empty_tuple.type = call %bound_method.loc21_3(%addr.loc21_3)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc18: <bound method> = bound_method %a.var, constants.%T.as.Destroy.impl.Op.f0b
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.f0b, @T.as.Destroy.impl.Op(constants.%array_type) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn.c51]
// CHECK:STDOUT:   %bound_method.loc18_3: <bound method> = bound_method %a.var, %T.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc18: %ptr.f01 = addr_of %a.var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc18: init %empty_tuple.type = call %bound_method.loc18_3(%addr.loc18)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ValueBinding(%b.param: %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.5d8 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.5d8 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %int_1.loc26_27: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc26_30: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3.loc26_33: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc26_34.1: %tuple.type = tuple_literal (%int_1.loc26_27, %int_2.loc26_30, %int_3.loc26_33)
// CHECK:STDOUT:   %impl.elem0.loc26_34.1: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc26_34.1: <bound method> = bound_method %int_1.loc26_27, %impl.elem0.loc26_34.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc26_34.1: <specific function> = specific_function %impl.elem0.loc26_34.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26_34.2: <bound method> = bound_method %int_1.loc26_27, %specific_fn.loc26_34.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc26_34.1: init %i32 = call %bound_method.loc26_34.2(%int_1.loc26_27) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc26_34.2: init %i32 = converted %int_1.loc26_27, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc26_34.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %int_0.loc26: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc26_34.3: ref %i32 = array_index %a.var, %int_0.loc26
// CHECK:STDOUT:   %.loc26_34.4: init %i32 = initialize_from %.loc26_34.2 to %.loc26_34.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc26_34.2: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc26_34.3: <bound method> = bound_method %int_2.loc26_30, %impl.elem0.loc26_34.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc26_34.2: <specific function> = specific_function %impl.elem0.loc26_34.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26_34.4: <bound method> = bound_method %int_2.loc26_30, %specific_fn.loc26_34.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc26_34.2: init %i32 = call %bound_method.loc26_34.4(%int_2.loc26_30) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc26_34.5: init %i32 = converted %int_2.loc26_30, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc26_34.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %int_1.loc26_34: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc26_34.6: ref %i32 = array_index %a.var, %int_1.loc26_34
// CHECK:STDOUT:   %.loc26_34.7: init %i32 = initialize_from %.loc26_34.5 to %.loc26_34.6 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc26_34.3: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc26_34.5: <bound method> = bound_method %int_3.loc26_33, %impl.elem0.loc26_34.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc26_34.3: <specific function> = specific_function %impl.elem0.loc26_34.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26_34.6: <bound method> = bound_method %int_3.loc26_33, %specific_fn.loc26_34.3 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc26_34.3: init %i32 = call %bound_method.loc26_34.6(%int_3.loc26_33) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc26_34.8: init %i32 = converted %int_3.loc26_33, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc26_34.3 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %int_2.loc26_34: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc26_34.9: ref %i32 = array_index %a.var, %int_2.loc26_34
// CHECK:STDOUT:   %.loc26_34.10: init %i32 = initialize_from %.loc26_34.8 to %.loc26_34.9 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc26_34.11: init %array_type = array_init (%.loc26_34.4, %.loc26_34.7, %.loc26_34.10) to %a.var [concrete = constants.%array]
// CHECK:STDOUT:   %.loc26_3: init %array_type = converted %.loc26_34.1, %.loc26_34.11 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc26_3
// CHECK:STDOUT:   %.loc26_22: type = splice_block %array_type.loc26 [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc26: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc26: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_3.loc26_21: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type.loc26: type = array_type %int_3.loc26_21, %i32.loc26 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = bind_name a, %a.var
// CHECK:STDOUT:   %a.ref: ref %array_type = name_ref a, %a
// CHECK:STDOUT:   %int_0.loc30: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_32.loc30: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc30: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc30: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc30_5.1: <bound method> = bound_method %int_0.loc30, %impl.elem0.loc30 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc30: <specific function> = specific_function %impl.elem0.loc30, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc30_5.2: <bound method> = bound_method %int_0.loc30, %specific_fn.loc30 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc30: init %i32 = call %bound_method.loc30_5.2(%int_0.loc30) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc30_5.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc30 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc30_5.2: %i32 = converted %int_0.loc30, %.loc30_5.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc30_6: ref %i32 = array_index %a.ref, %.loc30_5.2
// CHECK:STDOUT:   %b.ref: %array_type = name_ref b, %b
// CHECK:STDOUT:   %int_0.loc31: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_32.loc31: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc31: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc31: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc31_5.1: <bound method> = bound_method %int_0.loc31, %impl.elem0.loc31 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc31: <specific function> = specific_function %impl.elem0.loc31, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc31_5.2: <bound method> = bound_method %int_0.loc31, %specific_fn.loc31 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc31: init %i32 = call %bound_method.loc31_5.2(%int_0.loc31) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc31_5.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc31 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc31_5.2: %i32 = converted %int_0.loc31, %.loc31_5.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc31_6.1: ref %array_type = value_as_ref %b.ref
// CHECK:STDOUT:   %.loc31_6.2: ref %i32 = array_index %.loc31_6.1, %.loc31_5.2
// CHECK:STDOUT:   %.loc31_6.3: %i32 = bind_value %.loc31_6.2
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc32_5.1: ref %array_type = temporary_storage
// CHECK:STDOUT:   %F.call: init %array_type = call %F.ref() to %.loc32_5.1
// CHECK:STDOUT:   %int_0.loc32: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc32_5.2: ref %array_type = temporary %.loc32_5.1, %F.call
// CHECK:STDOUT:   %int_32.loc32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc32: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc32_7.1: <bound method> = bound_method %int_0.loc32, %impl.elem0.loc32 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc32: <specific function> = specific_function %impl.elem0.loc32, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc32_7.2: <bound method> = bound_method %int_0.loc32, %specific_fn.loc32 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32: init %i32 = call %bound_method.loc32_7.2(%int_0.loc32) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc32_7.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc32 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc32_7.2: %i32 = converted %int_0.loc32, %.loc32_7.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc32_8.1: ref %i32 = array_index %.loc32_5.2, %.loc32_7.2
// CHECK:STDOUT:   %.loc32_8.2: %i32 = bind_value %.loc32_8.1
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc32: <bound method> = bound_method %.loc32_5.1, constants.%T.as.Destroy.impl.Op.f0b
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.f0b, @T.as.Destroy.impl.Op(constants.%array_type) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn.c51]
// CHECK:STDOUT:   %bound_method.loc32_5: <bound method> = bound_method %.loc32_5.1, %T.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc32: %ptr.f01 = addr_of %.loc32_5.1
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc32: init %empty_tuple.type = call %bound_method.loc32_5(%addr.loc32)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc26: <bound method> = bound_method %a.var, constants.%T.as.Destroy.impl.Op.f0b
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.f0b, @T.as.Destroy.impl.Op(constants.%array_type) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn.c51]
// CHECK:STDOUT:   %bound_method.loc26_3: <bound method> = bound_method %a.var, %T.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc26: %ptr.f01 = addr_of %a.var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc26: init %empty_tuple.type = call %bound_method.loc26_3(%addr.loc26)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
