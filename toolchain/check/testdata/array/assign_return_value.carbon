// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/assign_return_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/assign_return_value.carbon

fn F() -> (i32,) { return (0,); }

fn Run() {
  var t: array(i32, 1) = F();
}

// CHECK:STDOUT: --- assign_return_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.85c: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.type.a1c: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (imports.%Core.import_ref.a5b), @impl.4f9 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness %ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0.5c6, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.a1c = tuple_value (%int_0.6a9) [concrete]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [concrete]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1, %i32 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.23 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (imports.%Core.import_ref.0b9), @impl.49c [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.bc9: <witness> = impl_witness %Destroy.impl_witness_table, @impl.49c(%array_type) [concrete]
// CHECK:STDOUT:   %Op.type.fa9: type = fn_type @Op.24, @impl.49c(%array_type) [concrete]
// CHECK:STDOUT:   %Op.606: %Op.type.fa9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.3c4: %Destroy.type = facet_value %array_type, (%Destroy.impl_witness.bc9) [concrete]
// CHECK:STDOUT:   %.050: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.3c4 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.088: <witness> = impl_witness %Destroy.impl_witness_table, @impl.49c(%tuple.type.a1c) [concrete]
// CHECK:STDOUT:   %Op.type.438: type = fn_type @Op.24, @impl.49c(%tuple.type.a1c) [concrete]
// CHECK:STDOUT:   %Op.c8e: %Op.type.438 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.935: %Destroy.type = facet_value %tuple.type.a1c, (%Destroy.impl_witness.088) [concrete]
// CHECK:STDOUT:   %.6a9: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.935 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.f94: <specific function> = specific_function %Op.c8e, @Op.24(%tuple.type.a1c) [concrete]
// CHECK:STDOUT:   %Op.specific_fn.7c2: <specific function> = specific_function %Op.606, @Op.24(%array_type) [concrete]
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %tuple.type.a1c = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type.a1c = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc11_16.1: %tuple.type.85c = tuple_literal (%i32)
// CHECK:STDOUT:     %.loc11_16.2: type = converted %.loc11_16.1, constants.%tuple.type.a1c [concrete = constants.%tuple.type.a1c]
// CHECK:STDOUT:     %return.param: ref %tuple.type.a1c = out_param call_param0
// CHECK:STDOUT:     %return: ref %tuple.type.a1c = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [concrete = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %tuple.type.a1c {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc11_30.1: %tuple.type.985 = tuple_literal (%int_0)
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_30.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_30.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc11_30.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc11_30.2: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc11_30.3: %i32 = converted %int_0, %.loc11_30.2 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %tuple: %tuple.type.a1c = tuple_value (%.loc11_30.3) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc11_31: %tuple.type.a1c = converted %.loc11_30.1, %tuple [concrete = constants.%tuple]
// CHECK:STDOUT:   return %.loc11_31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t.patt: %array_type = binding_pattern t
// CHECK:STDOUT:     %.loc14_3.1: %array_type = var_pattern %t.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t.var: ref %array_type = var t
// CHECK:STDOUT:   %impl.elem0.loc14_3: %.050 = impl_witness_access constants.%Destroy.impl_witness.bc9, element0 [concrete = constants.%Op.606]
// CHECK:STDOUT:   %bound_method.loc14_3: <bound method> = bound_method %t.var, %impl.elem0.loc14_3
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %tuple.type.a1c = call %F.ref()
// CHECK:STDOUT:   %.loc14_28.1: ref %tuple.type.a1c = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc14_28: %.6a9 = impl_witness_access constants.%Destroy.impl_witness.088, element0 [concrete = constants.%Op.c8e]
// CHECK:STDOUT:   %bound_method.loc14_28: <bound method> = bound_method %.loc14_28.1, %impl.elem0.loc14_28
// CHECK:STDOUT:   %.loc14_28.2: ref %tuple.type.a1c = temporary %.loc14_28.1, %F.call
// CHECK:STDOUT:   %tuple.elem0.loc14_28.1: ref %i32 = tuple_access %.loc14_28.2, element0
// CHECK:STDOUT:   %.loc14_28.3: %i32 = bind_value %tuple.elem0.loc14_28.1
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc14_28.4: ref %i32 = array_index %t.var, %int_0
// CHECK:STDOUT:   %.loc14_28.5: init %i32 = initialize_from %.loc14_28.3 to %.loc14_28.4
// CHECK:STDOUT:   %.loc14_28.6: init %array_type = array_init (%.loc14_28.5) to %t.var
// CHECK:STDOUT:   %.loc14_3.2: init %array_type = converted %F.call, %.loc14_28.6
// CHECK:STDOUT:   assign %t.var, %.loc14_3.2
// CHECK:STDOUT:   %.loc14_22: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:     %array_type: type = array_type %int_1, %i32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t: ref %array_type = bind_name t, %t.var
// CHECK:STDOUT:   %specific_fn.loc14_28: <specific function> = specific_function %impl.elem0.loc14_28, @Op.24(constants.%tuple.type.a1c) [concrete = constants.%Op.specific_fn.f94]
// CHECK:STDOUT:   %bound_method.1: <bound method> = bound_method %.loc14_28.1, %specific_fn.loc14_28
// CHECK:STDOUT:   %tuple.elem0.loc14_28.2: ref %i32 = tuple_access %.loc14_28.1, element0
// CHECK:STDOUT:   %.loc14_28.7: %i32 = bind_value %tuple.elem0.loc14_28.2
// CHECK:STDOUT:   %tuple: %tuple.type.a1c = tuple_value (%.loc14_28.7)
// CHECK:STDOUT:   %.loc14_28.8: %tuple.type.a1c = converted %.loc14_28.1, %tuple
// CHECK:STDOUT:   %no_op.1: init %empty_tuple.type = call %bound_method.1(%.loc14_28.8)
// CHECK:STDOUT:   %specific_fn.loc14_3: <specific function> = specific_function %impl.elem0.loc14_3, @Op.24(constants.%array_type) [concrete = constants.%Op.specific_fn.7c2]
// CHECK:STDOUT:   %bound_method.2: <bound method> = bound_method %t.var, %specific_fn.loc14_3
// CHECK:STDOUT:   %.loc14_3.3: %array_type = bind_value %t.var
// CHECK:STDOUT:   %no_op.2: init %empty_tuple.type = call %bound_method.2(%.loc14_3.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
