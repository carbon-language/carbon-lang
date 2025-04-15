// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/array_in_place.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/array_in_place.carbon

fn F() -> (i32, i32, i32);

fn G() {
  var v: array((i32, i32, i32), 2) = (F(), F());
}

// CHECK:STDOUT: --- array_in_place.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.ff9: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.189: type = tuple_type (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_2, %tuple.type.189 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (imports.%Core.import_ref.0b9), @impl.49c [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.2d6: <witness> = impl_witness %Destroy.impl_witness_table, @impl.49c(%array_type) [concrete]
// CHECK:STDOUT:   %Op.type.43e: type = fn_type @Op.2, @impl.49c(%array_type) [concrete]
// CHECK:STDOUT:   %Op.353: %Op.type.43e = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.437: %Destroy.type = facet_value %array_type, (%Destroy.impl_witness.2d6) [concrete]
// CHECK:STDOUT:   %.f66: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.437 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.c0f: <witness> = impl_witness %Destroy.impl_witness_table, @impl.49c(%tuple.type.189) [concrete]
// CHECK:STDOUT:   %Op.type.19f: type = fn_type @Op.2, @impl.49c(%tuple.type.189) [concrete]
// CHECK:STDOUT:   %Op.672: %Op.type.19f = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.ba8: %Destroy.type = facet_value %tuple.type.189, (%Destroy.impl_witness.c0f) [concrete]
// CHECK:STDOUT:   %.cce: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.ba8 [concrete]
// CHECK:STDOUT:   %tuple.type.99b: type = tuple_type (%tuple.type.189, %tuple.type.189) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.cf9: <specific function> = specific_function %Op.672, @Op.2(%tuple.type.189) [concrete]
// CHECK:STDOUT:   %Op.specific_fn.130: <specific function> = specific_function %Op.353, @Op.2(%array_type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
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
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %tuple.type.189 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type.189 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc11_12: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_12: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc11_25.1: %tuple.type.ff9 = tuple_literal (%i32.loc11_12, %i32.loc11_17, %i32.loc11_22)
// CHECK:STDOUT:     %.loc11_25.2: type = converted %.loc11_25.1, constants.%tuple.type.189 [concrete = constants.%tuple.type.189]
// CHECK:STDOUT:     %return.param: ref %tuple.type.189 = out_param call_param0
// CHECK:STDOUT:     %return: ref %tuple.type.189 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return.param: %tuple.type.189;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %array_type = binding_pattern v
// CHECK:STDOUT:     %.loc14_3.1: %array_type = var_pattern %v.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %array_type = var v
// CHECK:STDOUT:   %impl.elem0.loc14_3: %.f66 = impl_witness_access constants.%Destroy.impl_witness.2d6, element0 [concrete = constants.%Op.353]
// CHECK:STDOUT:   %bound_method.loc14_3: <bound method> = bound_method %v.var, %impl.elem0.loc14_3
// CHECK:STDOUT:   %F.ref.loc14_39: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc14_47.1: ref %tuple.type.189 = splice_block %.loc14_47.2 {
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:     %.loc14_47.2: ref %tuple.type.189 = array_index %v.var, %int_0
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc14_41: %.cce = impl_witness_access constants.%Destroy.impl_witness.c0f, element0 [concrete = constants.%Op.672]
// CHECK:STDOUT:   %bound_method.loc14_41: <bound method> = bound_method %.loc14_47.1, %impl.elem0.loc14_41
// CHECK:STDOUT:   %F.call.loc14_41: init %tuple.type.189 = call %F.ref.loc14_39() to %.loc14_47.1
// CHECK:STDOUT:   %F.ref.loc14_44: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc14_47.3: ref %tuple.type.189 = splice_block %.loc14_47.4 {
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:     %.loc14_47.4: ref %tuple.type.189 = array_index %v.var, %int_1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc14_46: %.cce = impl_witness_access constants.%Destroy.impl_witness.c0f, element0 [concrete = constants.%Op.672]
// CHECK:STDOUT:   %bound_method.loc14_46: <bound method> = bound_method %.loc14_47.3, %impl.elem0.loc14_46
// CHECK:STDOUT:   %F.call.loc14_46: init %tuple.type.189 = call %F.ref.loc14_44() to %.loc14_47.3
// CHECK:STDOUT:   %.loc14_47.5: %tuple.type.99b = tuple_literal (%F.call.loc14_41, %F.call.loc14_46)
// CHECK:STDOUT:   %.loc14_47.6: init %array_type = array_init (%F.call.loc14_41, %F.call.loc14_46) to %v.var
// CHECK:STDOUT:   %.loc14_3.2: init %array_type = converted %.loc14_47.5, %.loc14_47.6
// CHECK:STDOUT:   assign %v.var, %.loc14_3.2
// CHECK:STDOUT:   %.loc14_34: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc14_17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc14_22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc14_27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_27: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc14_30.1: %tuple.type.ff9 = tuple_literal (%i32.loc14_17, %i32.loc14_22, %i32.loc14_27)
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:     %.loc14_30.2: type = converted %.loc14_30.1, constants.%tuple.type.189 [concrete = constants.%tuple.type.189]
// CHECK:STDOUT:     %array_type: type = array_type %int_2, %.loc14_30.2 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %array_type = bind_name v, %v.var
// CHECK:STDOUT:   %specific_fn.loc14_46: <specific function> = specific_function %impl.elem0.loc14_46, @Op.2(constants.%tuple.type.189) [concrete = constants.%Op.specific_fn.cf9]
// CHECK:STDOUT:   %bound_method.1: <bound method> = bound_method %.loc14_47.3, %specific_fn.loc14_46
// CHECK:STDOUT:   %tuple.elem0.loc14_47.1: ref %i32 = tuple_access %.loc14_47.3, element0
// CHECK:STDOUT:   %.loc14_47.7: %i32 = bind_value %tuple.elem0.loc14_47.1
// CHECK:STDOUT:   %tuple.elem1.loc14_47.1: ref %i32 = tuple_access %.loc14_47.3, element1
// CHECK:STDOUT:   %.loc14_47.8: %i32 = bind_value %tuple.elem1.loc14_47.1
// CHECK:STDOUT:   %tuple.elem2.loc14_47.1: ref %i32 = tuple_access %.loc14_47.3, element2
// CHECK:STDOUT:   %.loc14_47.9: %i32 = bind_value %tuple.elem2.loc14_47.1
// CHECK:STDOUT:   %tuple.loc14_47.1: %tuple.type.189 = tuple_value (%.loc14_47.7, %.loc14_47.8, %.loc14_47.9)
// CHECK:STDOUT:   %.loc14_47.10: %tuple.type.189 = converted %.loc14_47.3, %tuple.loc14_47.1
// CHECK:STDOUT:   %no_op.1: init %empty_tuple.type = call %bound_method.1(%.loc14_47.10)
// CHECK:STDOUT:   %specific_fn.loc14_41: <specific function> = specific_function %impl.elem0.loc14_41, @Op.2(constants.%tuple.type.189) [concrete = constants.%Op.specific_fn.cf9]
// CHECK:STDOUT:   %bound_method.2: <bound method> = bound_method %.loc14_47.1, %specific_fn.loc14_41
// CHECK:STDOUT:   %tuple.elem0.loc14_47.2: ref %i32 = tuple_access %.loc14_47.1, element0
// CHECK:STDOUT:   %.loc14_47.11: %i32 = bind_value %tuple.elem0.loc14_47.2
// CHECK:STDOUT:   %tuple.elem1.loc14_47.2: ref %i32 = tuple_access %.loc14_47.1, element1
// CHECK:STDOUT:   %.loc14_47.12: %i32 = bind_value %tuple.elem1.loc14_47.2
// CHECK:STDOUT:   %tuple.elem2.loc14_47.2: ref %i32 = tuple_access %.loc14_47.1, element2
// CHECK:STDOUT:   %.loc14_47.13: %i32 = bind_value %tuple.elem2.loc14_47.2
// CHECK:STDOUT:   %tuple.loc14_47.2: %tuple.type.189 = tuple_value (%.loc14_47.11, %.loc14_47.12, %.loc14_47.13)
// CHECK:STDOUT:   %.loc14_47.14: %tuple.type.189 = converted %.loc14_47.1, %tuple.loc14_47.2
// CHECK:STDOUT:   %no_op.2: init %empty_tuple.type = call %bound_method.2(%.loc14_47.14)
// CHECK:STDOUT:   %specific_fn.loc14_3: <specific function> = specific_function %impl.elem0.loc14_3, @Op.2(constants.%array_type) [concrete = constants.%Op.specific_fn.130]
// CHECK:STDOUT:   %bound_method.3: <bound method> = bound_method %v.var, %specific_fn.loc14_3
// CHECK:STDOUT:   %.loc14_3.3: %array_type = bind_value %v.var
// CHECK:STDOUT:   %no_op.3: init %empty_tuple.type = call %bound_method.3(%.loc14_3.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
