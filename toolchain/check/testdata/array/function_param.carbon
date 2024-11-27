// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/function_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/function_param.carbon

fn F(arr: array(i32, 3), i: i32) -> i32 {
  return arr[i];
}

fn G() -> i32 {
  return F((1, 2, 3), 1);
}

// CHECK:STDOUT: --- function_param.carbon
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
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.1(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.1(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, %impl_witness.d39 [concrete]
// CHECK:STDOUT:   %.a0b: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.70c: <specific function> = specific_function %Convert.bound.ab5, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.787: <specific function> = specific_function %Convert.bound.ef9, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Convert.bound.b30: <bound method> = bound_method %int_3.1ba, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.b42: <specific function> = specific_function %Convert.bound.b30, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_1.5d2, %int_2.ef8, %int_3.822) [concrete]
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
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %arr.patt: %array_type = binding_pattern arr
// CHECK:STDOUT:     %arr.param_patt: %array_type = value_param_pattern %arr.patt, runtime_param0
// CHECK:STDOUT:     %i.patt: %i32 = binding_pattern i
// CHECK:STDOUT:     %i.param_patt: %i32 = value_param_pattern %i.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc11_37: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_37: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %arr.param: %array_type = value_param runtime_param0
// CHECK:STDOUT:     %.loc11_23: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:       %int_32.loc11_17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc11_17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:       %array_type: type = array_type %int_3, %i32 [concrete = constants.%array_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %arr: %array_type = bind_name arr, %arr.param
// CHECK:STDOUT:     %i.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc11_29: type = splice_block %i32.loc11_29 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc11_29: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc11_29: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %i: %i32 = bind_name i, %i.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%arr.param_patt: %array_type, %i.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr.ref: %array_type = name_ref arr, %arr
// CHECK:STDOUT:   %i.ref: %i32 = name_ref i, %i
// CHECK:STDOUT:   %int_32.loc12: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc12: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc12_15.1: ref %array_type = value_as_ref %arr.ref
// CHECK:STDOUT:   %.loc12_15.2: ref %i32 = array_index %.loc12_15.1, %i.ref
// CHECK:STDOUT:   %.loc12_15.3: %i32 = bind_value %.loc12_15.2
// CHECK:STDOUT:   return %.loc12_15.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %int_1.loc16_13: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc16_16: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc16_20.1: %tuple.type = tuple_literal (%int_1.loc16_13, %int_2.loc16_16, %int_3)
// CHECK:STDOUT:   %int_1.loc16_23: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc16_20.1: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc16_20.1: <bound method> = bound_method %int_1.loc16_13, %impl.elem0.loc16_20.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc16_20.1: <specific function> = specific_function %bound_method.loc16_20.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.70c]
// CHECK:STDOUT:   %int.convert_checked.loc16_20.1: init %i32 = call %specific_fn.loc16_20.1(%int_1.loc16_13) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc16_20.2: init %i32 = converted %int_1.loc16_13, %int.convert_checked.loc16_20.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc16_20.3: ref %array_type = temporary_storage
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc16_20.4: ref %i32 = array_index %.loc16_20.3, %int_0
// CHECK:STDOUT:   %.loc16_20.5: init %i32 = initialize_from %.loc16_20.2 to %.loc16_20.4 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc16_20.2: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc16_20.2: <bound method> = bound_method %int_2.loc16_16, %impl.elem0.loc16_20.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc16_20.2: <specific function> = specific_function %bound_method.loc16_20.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.787]
// CHECK:STDOUT:   %int.convert_checked.loc16_20.2: init %i32 = call %specific_fn.loc16_20.2(%int_2.loc16_16) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc16_20.6: init %i32 = converted %int_2.loc16_16, %int.convert_checked.loc16_20.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %int_1.loc16_20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc16_20.7: ref %i32 = array_index %.loc16_20.3, %int_1.loc16_20
// CHECK:STDOUT:   %.loc16_20.8: init %i32 = initialize_from %.loc16_20.6 to %.loc16_20.7 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc16_20.3: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc16_20.3: <bound method> = bound_method %int_3, %impl.elem0.loc16_20.3 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc16_20.3: <specific function> = specific_function %bound_method.loc16_20.3, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.b42]
// CHECK:STDOUT:   %int.convert_checked.loc16_20.3: init %i32 = call %specific_fn.loc16_20.3(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc16_20.9: init %i32 = converted %int_3, %int.convert_checked.loc16_20.3 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %int_2.loc16_20: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc16_20.10: ref %i32 = array_index %.loc16_20.3, %int_2.loc16_20
// CHECK:STDOUT:   %.loc16_20.11: init %i32 = initialize_from %.loc16_20.9 to %.loc16_20.10 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc16_20.12: init %array_type = array_init (%.loc16_20.5, %.loc16_20.8, %.loc16_20.11) to %.loc16_20.3 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc16_20.13: init %array_type = converted %.loc16_20.1, %.loc16_20.12 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc16_20.14: ref %array_type = temporary %.loc16_20.3, %.loc16_20.13
// CHECK:STDOUT:   %.loc16_20.15: %array_type = bind_value %.loc16_20.14
// CHECK:STDOUT:   %impl.elem0.loc16_23: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc16_23: <bound method> = bound_method %int_1.loc16_23, %impl.elem0.loc16_23 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc16_23: <specific function> = specific_function %bound_method.loc16_23, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.70c]
// CHECK:STDOUT:   %int.convert_checked.loc16_23: init %i32 = call %specific_fn.loc16_23(%int_1.loc16_23) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc16_23.1: %i32 = value_of_initializer %int.convert_checked.loc16_23 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc16_23.2: %i32 = converted %int_1.loc16_23, %.loc16_23.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.ref(%.loc16_20.15, %.loc16_23.2)
// CHECK:STDOUT:   %.loc16_25.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc16_25.2: %i32 = converted %F.call, %.loc16_25.1
// CHECK:STDOUT:   return %.loc16_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
