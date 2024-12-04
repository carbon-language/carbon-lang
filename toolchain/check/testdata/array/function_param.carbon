// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/function_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/function_param.carbon

fn F(arr: [i32; 3], i: i32) -> i32 {
  return arr[i];
}

fn G() -> i32 {
  return F((1, 2, 3), 1);
}

// CHECK:STDOUT: --- function_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %int_3.1: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1, %i32 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_3.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_3.2: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_1.2, %int_2.2, %int_3.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %arr.patt: %array_type = binding_pattern arr
// CHECK:STDOUT:     %arr.param_patt: %array_type = value_param_pattern %arr.patt, runtime_param0
// CHECK:STDOUT:     %i.patt: %i32 = binding_pattern i
// CHECK:STDOUT:     %i.param_patt: %i32 = value_param_pattern %i.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc11_12: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11_12: init type = call constants.%Int(%int_32.loc11_12) [template = constants.%i32]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3.1]
// CHECK:STDOUT:     %.loc11_12.1: type = value_of_initializer %int.make_type_signed.loc11_12 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_12.2: type = converted %int.make_type_signed.loc11_12, %.loc11_12.1 [template = constants.%i32]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %i32 [template = constants.%array_type]
// CHECK:STDOUT:     %int_32.loc11_24: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11_24: init type = call constants.%Int(%int_32.loc11_24) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_24.1: type = value_of_initializer %int.make_type_signed.loc11_24 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_24.2: type = converted %int.make_type_signed.loc11_24, %.loc11_24.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11_32: init type = call constants.%Int(%int_32.loc11_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_32.1: type = value_of_initializer %int.make_type_signed.loc11_32 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_32.2: type = converted %int.make_type_signed.loc11_32, %.loc11_32.1 [template = constants.%i32]
// CHECK:STDOUT:     %arr.param: %array_type = value_param runtime_param0
// CHECK:STDOUT:     %arr: %array_type = bind_name arr, %arr.param
// CHECK:STDOUT:     %i.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %i: %i32 = bind_name i, %i.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_11.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_11.2: type = converted %int.make_type_signed, %.loc15_11.1 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%arr.param_patt: %array_type, %i.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr.ref: %array_type = name_ref arr, %arr
// CHECK:STDOUT:   %i.ref: %i32 = name_ref i, %i
// CHECK:STDOUT:   %int_32.loc12: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%int_32.loc12) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_15.1: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_15.2: type = converted %int.make_type_signed.loc12, %.loc12_15.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_15.3: ref %array_type = value_as_ref %arr.ref
// CHECK:STDOUT:   %.loc12_15.4: ref %i32 = array_index %.loc12_15.3, %i.ref
// CHECK:STDOUT:   %.loc12_15.5: %i32 = bind_value %.loc12_15.4
// CHECK:STDOUT:   return %.loc12_15.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %int_1.loc16_13: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2.loc16_16: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3.1]
// CHECK:STDOUT:   %.loc16_20.1: %tuple.type = tuple_literal (%int_1.loc16_13, %int_2.loc16_16, %int_3)
// CHECK:STDOUT:   %int_1.loc16_23: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc16_20.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc16_20.1: <bound method> = bound_method %int_1.loc16_13, %impl.elem0.loc16_20.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_20.1: <specific function> = specific_function %Convert.bound.loc16_20.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc16_20.1: init %i32 = call %Convert.specific_fn.loc16_20.1(%int_1.loc16_13) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc16_20.2: init %i32 = converted %int_1.loc16_13, %int.convert_checked.loc16_20.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc16_20.3: ref %array_type = temporary_storage
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc16_20.4: ref %i32 = array_index %.loc16_20.3, %int_0
// CHECK:STDOUT:   %.loc16_20.5: init %i32 = initialize_from %.loc16_20.2 to %.loc16_20.4 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc16_20.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc16_20.2: <bound method> = bound_method %int_2.loc16_16, %impl.elem0.loc16_20.2 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_20.2: <specific function> = specific_function %Convert.bound.loc16_20.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc16_20.2: init %i32 = call %Convert.specific_fn.loc16_20.2(%int_2.loc16_16) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc16_20.6: init %i32 = converted %int_2.loc16_16, %int.convert_checked.loc16_20.2 [template = constants.%int_2.2]
// CHECK:STDOUT:   %int_1.loc16_20: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc16_20.7: ref %i32 = array_index %.loc16_20.3, %int_1.loc16_20
// CHECK:STDOUT:   %.loc16_20.8: init %i32 = initialize_from %.loc16_20.6 to %.loc16_20.7 [template = constants.%int_2.2]
// CHECK:STDOUT:   %impl.elem0.loc16_20.3: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc16_20.3: <bound method> = bound_method %int_3, %impl.elem0.loc16_20.3 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_20.3: <specific function> = specific_function %Convert.bound.loc16_20.3, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc16_20.3: init %i32 = call %Convert.specific_fn.loc16_20.3(%int_3) [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc16_20.9: init %i32 = converted %int_3, %int.convert_checked.loc16_20.3 [template = constants.%int_3.2]
// CHECK:STDOUT:   %int_2.loc16_20: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %.loc16_20.10: ref %i32 = array_index %.loc16_20.3, %int_2.loc16_20
// CHECK:STDOUT:   %.loc16_20.11: init %i32 = initialize_from %.loc16_20.9 to %.loc16_20.10 [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc16_20.12: init %array_type = array_init (%.loc16_20.5, %.loc16_20.8, %.loc16_20.11) to %.loc16_20.3 [template = constants.%array]
// CHECK:STDOUT:   %.loc16_20.13: init %array_type = converted %.loc16_20.1, %.loc16_20.12 [template = constants.%array]
// CHECK:STDOUT:   %.loc16_20.14: ref %array_type = temporary %.loc16_20.3, %.loc16_20.13
// CHECK:STDOUT:   %.loc16_20.15: %array_type = bind_value %.loc16_20.14
// CHECK:STDOUT:   %impl.elem0.loc16_23: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc16_23: <bound method> = bound_method %int_1.loc16_23, %impl.elem0.loc16_23 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_23: <specific function> = specific_function %Convert.bound.loc16_23, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc16_23: init %i32 = call %Convert.specific_fn.loc16_23(%int_1.loc16_23) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc16_23.1: %i32 = value_of_initializer %int.convert_checked.loc16_23 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc16_23.2: %i32 = converted %int_1.loc16_23, %.loc16_23.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.ref(%.loc16_20.15, %.loc16_23.2)
// CHECK:STDOUT:   %.loc16_25.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc16_25.2: %i32 = converted %F.call, %.loc16_25.1
// CHECK:STDOUT:   return %.loc16_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
