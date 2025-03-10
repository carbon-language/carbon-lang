// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1ba, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Convert.bound.b30: <bound method> = bound_method %int_3.1ba, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.047: <bound method> = bound_method %int_3.1ba, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_1.5d2, %int_2.ef8, %int_3.822) [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Convert.bound.d04: <bound method> = bound_method %int_0.5c6, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b6e: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Convert.bound.ac3: <bound method> = bound_method %int_4.0c1, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.1da: <bound method> = bound_method %int_4.0c1, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %ValueBinding.type: type = fn_type @ValueBinding [concrete]
// CHECK:STDOUT:   %ValueBinding: %ValueBinding.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %return.patt: %array_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %array_type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %i32 [concrete = constants.%array_type]
// CHECK:STDOUT:     %return.param: ref %array_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %array_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %b.patt: %array_type = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %array_type = value_param_pattern %b.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: %array_type = value_param call_param0
// CHECK:STDOUT:     %.loc13: type = splice_block %array_type.loc13 [concrete = constants.%array_type] {
// CHECK:STDOUT:       %int_32.loc13: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc13: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %int_3.loc13: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:       %array_type.loc13: type = array_type %int_3.loc13, %i32 [concrete = constants.%array_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %array_type = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ValueBinding.decl: %ValueBinding.type = fn_decl @ValueBinding [concrete = constants.%ValueBinding] {
// CHECK:STDOUT:     %b.patt: %array_type = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %array_type = value_param_pattern %b.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: %array_type = value_param call_param0
// CHECK:STDOUT:     %.loc21: type = splice_block %array_type.loc21 [concrete = constants.%array_type] {
// CHECK:STDOUT:       %int_32.loc21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc21: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %int_3.loc21: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:       %array_type.loc21: type = array_type %int_3.loc21, %i32 [concrete = constants.%array_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %array_type = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %array_type;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%b.param_patt: %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %array_type = binding_pattern a
// CHECK:STDOUT:     %.loc14_3.1: %array_type = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var a
// CHECK:STDOUT:   %int_1.loc14_27: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc14_30: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3.loc14_33: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc14_34.1: %tuple.type = tuple_literal (%int_1.loc14_27, %int_2.loc14_30, %int_3.loc14_33)
// CHECK:STDOUT:   %impl.elem0.loc14_34.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc14_34.1: <bound method> = bound_method %int_1.loc14_27, %impl.elem0.loc14_34.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc14_34.1: <specific function> = specific_function %impl.elem0.loc14_34.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_34.2: <bound method> = bound_method %int_1.loc14_27, %specific_fn.loc14_34.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc14_34.1: init %i32 = call %bound_method.loc14_34.2(%int_1.loc14_27) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc14_34.2: init %i32 = converted %int_1.loc14_27, %int.convert_checked.loc14_34.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %int_0.loc14: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc14_34.3: ref %i32 = array_index %a.var, %int_0.loc14
// CHECK:STDOUT:   %.loc14_34.4: init %i32 = initialize_from %.loc14_34.2 to %.loc14_34.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc14_34.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc14_34.3: <bound method> = bound_method %int_2.loc14_30, %impl.elem0.loc14_34.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc14_34.2: <specific function> = specific_function %impl.elem0.loc14_34.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_34.4: <bound method> = bound_method %int_2.loc14_30, %specific_fn.loc14_34.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc14_34.2: init %i32 = call %bound_method.loc14_34.4(%int_2.loc14_30) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc14_34.5: init %i32 = converted %int_2.loc14_30, %int.convert_checked.loc14_34.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %int_1.loc14_34: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc14_34.6: ref %i32 = array_index %a.var, %int_1.loc14_34
// CHECK:STDOUT:   %.loc14_34.7: init %i32 = initialize_from %.loc14_34.5 to %.loc14_34.6 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc14_34.3: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc14_34.5: <bound method> = bound_method %int_3.loc14_33, %impl.elem0.loc14_34.3 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc14_34.3: <specific function> = specific_function %impl.elem0.loc14_34.3, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_34.6: <bound method> = bound_method %int_3.loc14_33, %specific_fn.loc14_34.3 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc14_34.3: init %i32 = call %bound_method.loc14_34.6(%int_3.loc14_33) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc14_34.8: init %i32 = converted %int_3.loc14_33, %int.convert_checked.loc14_34.3 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %int_2.loc14_34: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc14_34.9: ref %i32 = array_index %a.var, %int_2.loc14_34
// CHECK:STDOUT:   %.loc14_34.10: init %i32 = initialize_from %.loc14_34.8 to %.loc14_34.9 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc14_34.11: init %array_type = array_init (%.loc14_34.4, %.loc14_34.7, %.loc14_34.10) to %a.var [concrete = constants.%array]
// CHECK:STDOUT:   %.loc14_3.2: init %array_type = converted %.loc14_34.1, %.loc14_34.11 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc14_3.2
// CHECK:STDOUT:   %.loc14_22: type = splice_block %array_type.loc14 [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_3.loc14_21: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type.loc14: type = array_type %int_3.loc14_21, %i32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pa.patt: %ptr.235 = binding_pattern pa
// CHECK:STDOUT:     %.loc17_3: %ptr.235 = var_pattern %pa.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pa.var: ref %ptr.235 = var pa
// CHECK:STDOUT:   %a.ref.loc17: ref %array_type = name_ref a, %a
// CHECK:STDOUT:   %int_0.loc17: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_32.loc17_22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc17_22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc17: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc17_21.1: <bound method> = bound_method %int_0.loc17, %impl.elem0.loc17 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_21.2: <bound method> = bound_method %int_0.loc17, %specific_fn.loc17 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc17: init %i32 = call %bound_method.loc17_21.2(%int_0.loc17) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc17_21.1: %i32 = value_of_initializer %int.convert_checked.loc17 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc17_21.2: %i32 = converted %int_0.loc17, %.loc17_21.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc17_22: ref %i32 = array_index %a.ref.loc17, %.loc17_21.2
// CHECK:STDOUT:   %addr: %ptr.235 = addr_of %.loc17_22
// CHECK:STDOUT:   assign %pa.var, %addr
// CHECK:STDOUT:   %.loc17_14: type = splice_block %ptr [concrete = constants.%ptr.235] {
// CHECK:STDOUT:     %int_32.loc17_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc17_11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %i32 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pa: ref %ptr.235 = bind_name pa, %pa.var
// CHECK:STDOUT:   %a.ref.loc18: ref %array_type = name_ref a, %a
// CHECK:STDOUT:   %int_0.loc18: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_32.loc18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc18: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc18_5: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc18_5.1: <bound method> = bound_method %int_0.loc18, %impl.elem0.loc18_5 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc18_5: <specific function> = specific_function %impl.elem0.loc18_5, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_5.2: <bound method> = bound_method %int_0.loc18, %specific_fn.loc18_5 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc18_5: init %i32 = call %bound_method.loc18_5.2(%int_0.loc18) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc18_5.1: %i32 = value_of_initializer %int.convert_checked.loc18_5 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc18_5.2: %i32 = converted %int_0.loc18, %.loc18_5.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc18_6: ref %i32 = array_index %a.ref.loc18, %.loc18_5.2
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc18_8: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc18_8.1: <bound method> = bound_method %int_4, %impl.elem0.loc18_8 [concrete = constants.%Convert.bound.ac3]
// CHECK:STDOUT:   %specific_fn.loc18_8: <specific function> = specific_function %impl.elem0.loc18_8, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_8.2: <bound method> = bound_method %int_4, %specific_fn.loc18_8 [concrete = constants.%bound_method.1da]
// CHECK:STDOUT:   %int.convert_checked.loc18_8: init %i32 = call %bound_method.loc18_8.2(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc18_8: init %i32 = converted %int_4, %int.convert_checked.loc18_8 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   assign %.loc18_6, %.loc18_8
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ValueBinding(%b.param_patt: %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %array_type = binding_pattern a
// CHECK:STDOUT:     %.loc22_3.1: %array_type = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var a
// CHECK:STDOUT:   %int_1.loc22_27: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc22_30: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3.loc22_33: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc22_34.1: %tuple.type = tuple_literal (%int_1.loc22_27, %int_2.loc22_30, %int_3.loc22_33)
// CHECK:STDOUT:   %impl.elem0.loc22_34.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc22_34.1: <bound method> = bound_method %int_1.loc22_27, %impl.elem0.loc22_34.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc22_34.1: <specific function> = specific_function %impl.elem0.loc22_34.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_34.2: <bound method> = bound_method %int_1.loc22_27, %specific_fn.loc22_34.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc22_34.1: init %i32 = call %bound_method.loc22_34.2(%int_1.loc22_27) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc22_34.2: init %i32 = converted %int_1.loc22_27, %int.convert_checked.loc22_34.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %int_0.loc22: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc22_34.3: ref %i32 = array_index %a.var, %int_0.loc22
// CHECK:STDOUT:   %.loc22_34.4: init %i32 = initialize_from %.loc22_34.2 to %.loc22_34.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc22_34.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc22_34.3: <bound method> = bound_method %int_2.loc22_30, %impl.elem0.loc22_34.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc22_34.2: <specific function> = specific_function %impl.elem0.loc22_34.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_34.4: <bound method> = bound_method %int_2.loc22_30, %specific_fn.loc22_34.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc22_34.2: init %i32 = call %bound_method.loc22_34.4(%int_2.loc22_30) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc22_34.5: init %i32 = converted %int_2.loc22_30, %int.convert_checked.loc22_34.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %int_1.loc22_34: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc22_34.6: ref %i32 = array_index %a.var, %int_1.loc22_34
// CHECK:STDOUT:   %.loc22_34.7: init %i32 = initialize_from %.loc22_34.5 to %.loc22_34.6 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc22_34.3: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc22_34.5: <bound method> = bound_method %int_3.loc22_33, %impl.elem0.loc22_34.3 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc22_34.3: <specific function> = specific_function %impl.elem0.loc22_34.3, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_34.6: <bound method> = bound_method %int_3.loc22_33, %specific_fn.loc22_34.3 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc22_34.3: init %i32 = call %bound_method.loc22_34.6(%int_3.loc22_33) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc22_34.8: init %i32 = converted %int_3.loc22_33, %int.convert_checked.loc22_34.3 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %int_2.loc22_34: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc22_34.9: ref %i32 = array_index %a.var, %int_2.loc22_34
// CHECK:STDOUT:   %.loc22_34.10: init %i32 = initialize_from %.loc22_34.8 to %.loc22_34.9 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc22_34.11: init %array_type = array_init (%.loc22_34.4, %.loc22_34.7, %.loc22_34.10) to %a.var [concrete = constants.%array]
// CHECK:STDOUT:   %.loc22_3.2: init %array_type = converted %.loc22_34.1, %.loc22_34.11 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc22_3.2
// CHECK:STDOUT:   %.loc22_22: type = splice_block %array_type.loc22 [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_3.loc22_21: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type.loc22: type = array_type %int_3.loc22_21, %i32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = bind_name a, %a.var
// CHECK:STDOUT:   %a.ref: ref %array_type = name_ref a, %a
// CHECK:STDOUT:   %int_0.loc26: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_32.loc26: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc26: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc26: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc26_5.1: <bound method> = bound_method %int_0.loc26, %impl.elem0.loc26 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc26: <specific function> = specific_function %impl.elem0.loc26, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26_5.2: <bound method> = bound_method %int_0.loc26, %specific_fn.loc26 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc26: init %i32 = call %bound_method.loc26_5.2(%int_0.loc26) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc26_5.1: %i32 = value_of_initializer %int.convert_checked.loc26 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc26_5.2: %i32 = converted %int_0.loc26, %.loc26_5.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc26_6: ref %i32 = array_index %a.ref, %.loc26_5.2
// CHECK:STDOUT:   %b.ref: %array_type = name_ref b, %b
// CHECK:STDOUT:   %int_0.loc27: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_32.loc27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc27: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc27: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc27_5.1: <bound method> = bound_method %int_0.loc27, %impl.elem0.loc27 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc27: <specific function> = specific_function %impl.elem0.loc27, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_5.2: <bound method> = bound_method %int_0.loc27, %specific_fn.loc27 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc27: init %i32 = call %bound_method.loc27_5.2(%int_0.loc27) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc27_5.1: %i32 = value_of_initializer %int.convert_checked.loc27 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc27_5.2: %i32 = converted %int_0.loc27, %.loc27_5.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc27_6.1: ref %array_type = value_as_ref %b.ref
// CHECK:STDOUT:   %.loc27_6.2: ref %i32 = array_index %.loc27_6.1, %.loc27_5.2
// CHECK:STDOUT:   %.loc27_6.3: %i32 = bind_value %.loc27_6.2
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc28_5.1: ref %array_type = temporary_storage
// CHECK:STDOUT:   %F.call: init %array_type = call %F.ref() to %.loc28_5.1
// CHECK:STDOUT:   %int_0.loc28: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc28_5.2: ref %array_type = temporary %.loc28_5.1, %F.call
// CHECK:STDOUT:   %int_32.loc28: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc28: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc28: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc28_7.1: <bound method> = bound_method %int_0.loc28, %impl.elem0.loc28 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc28: <specific function> = specific_function %impl.elem0.loc28, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc28_7.2: <bound method> = bound_method %int_0.loc28, %specific_fn.loc28 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc28: init %i32 = call %bound_method.loc28_7.2(%int_0.loc28) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc28_7.1: %i32 = value_of_initializer %int.convert_checked.loc28 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc28_7.2: %i32 = converted %int_0.loc28, %.loc28_7.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc28_8.1: ref %i32 = array_index %.loc28_5.2, %.loc28_7.2
// CHECK:STDOUT:   %.loc28_8.2: %i32 = bind_value %.loc28_8.1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
