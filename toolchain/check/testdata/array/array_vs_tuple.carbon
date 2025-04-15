// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/array_vs_tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/array_vs_tuple.carbon

fn G() {
  // These should have two different constant values.
  var a: array(i32, 3) = (1, 2, 3);
  var b: (i32, i32, i32) = (1, 2, 3);
}

// CHECK:STDOUT: --- array_vs_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1ba, %i32 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (imports.%Core.import_ref.0b9), @impl.49c [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (imports.%Core.import_ref.a5b), @impl.4f9 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.14a: <witness> = impl_witness %Destroy.impl_witness_table, @impl.49c(%array_type) [concrete]
// CHECK:STDOUT:   %Op.type.1e4: type = fn_type @Op.2, @impl.49c(%array_type) [concrete]
// CHECK:STDOUT:   %Op.f0b: %Op.type.1e4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.75f: %Destroy.type = facet_value %array_type, (%Destroy.impl_witness.14a) [concrete]
// CHECK:STDOUT:   %.9c9: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.75f [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.37f: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness %ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
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
// CHECK:STDOUT:   %tuple.type.ff9: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.189: type = tuple_type (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.c0f: <witness> = impl_witness %Destroy.impl_witness_table, @impl.49c(%tuple.type.189) [concrete]
// CHECK:STDOUT:   %Op.type.19f: type = fn_type @Op.2, @impl.49c(%tuple.type.189) [concrete]
// CHECK:STDOUT:   %Op.672: %Op.type.19f = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.ba8: %Destroy.type = facet_value %tuple.type.189, (%Destroy.impl_witness.c0f) [concrete]
// CHECK:STDOUT:   %.cce: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.ba8 [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.189 = tuple_value (%int_1.5d2, %int_2.ef8, %int_3.822) [concrete]
// CHECK:STDOUT:   %Op.specific_fn.cf9: <specific function> = specific_function %Op.672, @Op.2(%tuple.type.189) [concrete]
// CHECK:STDOUT:   %Op.specific_fn.c51: <specific function> = specific_function %Op.f0b, @Op.2(%array_type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %array_type = binding_pattern a
// CHECK:STDOUT:     %.loc13_3.1: %array_type = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var a
// CHECK:STDOUT:   %impl.elem0.loc13_3: %.9c9 = impl_witness_access constants.%Destroy.impl_witness.14a, element0 [concrete = constants.%Op.f0b]
// CHECK:STDOUT:   %bound_method.loc13_3: <bound method> = bound_method %a.var, %impl.elem0.loc13_3
// CHECK:STDOUT:   %int_1.loc13_27: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc13_30: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3.loc13_33: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc13_34.1: %tuple.type.37f = tuple_literal (%int_1.loc13_27, %int_2.loc13_30, %int_3.loc13_33)
// CHECK:STDOUT:   %impl.elem0.loc13_34.1: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc13_34.1: <bound method> = bound_method %int_1.loc13_27, %impl.elem0.loc13_34.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc13_34.1: <specific function> = specific_function %impl.elem0.loc13_34.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_34.2: <bound method> = bound_method %int_1.loc13_27, %specific_fn.loc13_34.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc13_34.1: init %i32 = call %bound_method.loc13_34.2(%int_1.loc13_27) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc13_34.2: init %i32 = converted %int_1.loc13_27, %int.convert_checked.loc13_34.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc13_34.3: ref %i32 = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc13_34.4: init %i32 = initialize_from %.loc13_34.2 to %.loc13_34.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc13_34.2: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc13_34.3: <bound method> = bound_method %int_2.loc13_30, %impl.elem0.loc13_34.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc13_34.2: <specific function> = specific_function %impl.elem0.loc13_34.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_34.4: <bound method> = bound_method %int_2.loc13_30, %specific_fn.loc13_34.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc13_34.2: init %i32 = call %bound_method.loc13_34.4(%int_2.loc13_30) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc13_34.5: init %i32 = converted %int_2.loc13_30, %int.convert_checked.loc13_34.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %int_1.loc13_34: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc13_34.6: ref %i32 = array_index %a.var, %int_1.loc13_34
// CHECK:STDOUT:   %.loc13_34.7: init %i32 = initialize_from %.loc13_34.5 to %.loc13_34.6 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc13_34.3: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc13_34.5: <bound method> = bound_method %int_3.loc13_33, %impl.elem0.loc13_34.3 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc13_34.3: <specific function> = specific_function %impl.elem0.loc13_34.3, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_34.6: <bound method> = bound_method %int_3.loc13_33, %specific_fn.loc13_34.3 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc13_34.3: init %i32 = call %bound_method.loc13_34.6(%int_3.loc13_33) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc13_34.8: init %i32 = converted %int_3.loc13_33, %int.convert_checked.loc13_34.3 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %int_2.loc13_34: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc13_34.9: ref %i32 = array_index %a.var, %int_2.loc13_34
// CHECK:STDOUT:   %.loc13_34.10: init %i32 = initialize_from %.loc13_34.8 to %.loc13_34.9 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc13_34.11: init %array_type = array_init (%.loc13_34.4, %.loc13_34.7, %.loc13_34.10) to %a.var [concrete = constants.%array]
// CHECK:STDOUT:   %.loc13_3.2: init %array_type = converted %.loc13_34.1, %.loc13_34.11 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc13_3.2
// CHECK:STDOUT:   %.loc13_22: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc13: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_3.loc13_21: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type: type = array_type %int_3.loc13_21, %i32.loc13 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %tuple.type.189 = binding_pattern b
// CHECK:STDOUT:     %.loc14_3.1: %tuple.type.189 = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %tuple.type.189 = var b
// CHECK:STDOUT:   %impl.elem0.loc14_3: %.cce = impl_witness_access constants.%Destroy.impl_witness.c0f, element0 [concrete = constants.%Op.672]
// CHECK:STDOUT:   %bound_method.loc14_3: <bound method> = bound_method %b.var, %impl.elem0.loc14_3
// CHECK:STDOUT:   %int_1.loc14: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc14: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3.loc14: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc14_36.1: %tuple.type.37f = tuple_literal (%int_1.loc14, %int_2.loc14, %int_3.loc14)
// CHECK:STDOUT:   %impl.elem0.loc14_36.1: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc14_36.1: <bound method> = bound_method %int_1.loc14, %impl.elem0.loc14_36.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc14_36.1: <specific function> = specific_function %impl.elem0.loc14_36.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_36.2: <bound method> = bound_method %int_1.loc14, %specific_fn.loc14_36.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc14_36.1: init %i32 = call %bound_method.loc14_36.2(%int_1.loc14) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc14_36.2: init %i32 = converted %int_1.loc14, %int.convert_checked.loc14_36.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %tuple.elem0.loc14_36: ref %i32 = tuple_access %b.var, element0
// CHECK:STDOUT:   %.loc14_36.3: init %i32 = initialize_from %.loc14_36.2 to %tuple.elem0.loc14_36 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc14_36.2: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc14_36.3: <bound method> = bound_method %int_2.loc14, %impl.elem0.loc14_36.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc14_36.2: <specific function> = specific_function %impl.elem0.loc14_36.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_36.4: <bound method> = bound_method %int_2.loc14, %specific_fn.loc14_36.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc14_36.2: init %i32 = call %bound_method.loc14_36.4(%int_2.loc14) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc14_36.4: init %i32 = converted %int_2.loc14, %int.convert_checked.loc14_36.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %tuple.elem1.loc14_36: ref %i32 = tuple_access %b.var, element1
// CHECK:STDOUT:   %.loc14_36.5: init %i32 = initialize_from %.loc14_36.4 to %tuple.elem1.loc14_36 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc14_36.3: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc14_36.5: <bound method> = bound_method %int_3.loc14, %impl.elem0.loc14_36.3 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc14_36.3: <specific function> = specific_function %impl.elem0.loc14_36.3, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_36.6: <bound method> = bound_method %int_3.loc14, %specific_fn.loc14_36.3 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc14_36.3: init %i32 = call %bound_method.loc14_36.6(%int_3.loc14) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc14_36.6: init %i32 = converted %int_3.loc14, %int.convert_checked.loc14_36.3 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %tuple.elem2.loc14_36: ref %i32 = tuple_access %b.var, element2
// CHECK:STDOUT:   %.loc14_36.7: init %i32 = initialize_from %.loc14_36.6 to %tuple.elem2.loc14_36 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc14_36.8: init %tuple.type.189 = tuple_init (%.loc14_36.3, %.loc14_36.5, %.loc14_36.7) to %b.var [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc14_3.2: init %tuple.type.189 = converted %.loc14_36.1, %.loc14_36.8 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign %b.var, %.loc14_3.2
// CHECK:STDOUT:   %.loc14_24.1: type = splice_block %.loc14_24.3 [concrete = constants.%tuple.type.189] {
// CHECK:STDOUT:     %int_32.loc14_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc14_16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc14_21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_21: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc14_24.2: %tuple.type.ff9 = tuple_literal (%i32.loc14_11, %i32.loc14_16, %i32.loc14_21)
// CHECK:STDOUT:     %.loc14_24.3: type = converted %.loc14_24.2, constants.%tuple.type.189 [concrete = constants.%tuple.type.189]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %tuple.type.189 = bind_name b, %b.var
// CHECK:STDOUT:   %specific_fn.loc14_3: <specific function> = specific_function %impl.elem0.loc14_3, @Op.2(constants.%tuple.type.189) [concrete = constants.%Op.specific_fn.cf9]
// CHECK:STDOUT:   %bound_method.1: <bound method> = bound_method %b.var, %specific_fn.loc14_3
// CHECK:STDOUT:   %tuple.elem0.loc14_3: ref %i32 = tuple_access %b.var, element0
// CHECK:STDOUT:   %.loc14_3.3: %i32 = bind_value %tuple.elem0.loc14_3
// CHECK:STDOUT:   %tuple.elem1.loc14_3: ref %i32 = tuple_access %b.var, element1
// CHECK:STDOUT:   %.loc14_3.4: %i32 = bind_value %tuple.elem1.loc14_3
// CHECK:STDOUT:   %tuple.elem2.loc14_3: ref %i32 = tuple_access %b.var, element2
// CHECK:STDOUT:   %.loc14_3.5: %i32 = bind_value %tuple.elem2.loc14_3
// CHECK:STDOUT:   %tuple: %tuple.type.189 = tuple_value (%.loc14_3.3, %.loc14_3.4, %.loc14_3.5)
// CHECK:STDOUT:   %.loc14_3.6: %tuple.type.189 = converted %b.var, %tuple
// CHECK:STDOUT:   %no_op.1: init %empty_tuple.type = call %bound_method.1(%.loc14_3.6)
// CHECK:STDOUT:   %specific_fn.loc13_3: <specific function> = specific_function %impl.elem0.loc13_3, @Op.2(constants.%array_type) [concrete = constants.%Op.specific_fn.c51]
// CHECK:STDOUT:   %bound_method.2: <bound method> = bound_method %a.var, %specific_fn.loc13_3
// CHECK:STDOUT:   %.loc13_3.3: %array_type = bind_value %a.var
// CHECK:STDOUT:   %no_op.2: init %empty_tuple.type = call %bound_method.2(%.loc13_3.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
