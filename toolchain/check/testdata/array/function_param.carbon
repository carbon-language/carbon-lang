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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: i32 = int_value 3 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.27: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.28: type = array_type %.27, i32 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.30: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.31: i32 = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.32: i32 = int_value 0 [template]
// CHECK:STDOUT:   %array: %.28 = tuple_value (%.30, %.31, %.1) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
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
// CHECK:STDOUT:     %arr.patt: %.28 = binding_pattern arr
// CHECK:STDOUT:     %arr.param_patt: %.28 = value_param_pattern %arr.patt, runtime_param0
// CHECK:STDOUT:     %i.patt: i32 = binding_pattern i
// CHECK:STDOUT:     %i.param_patt: i32 = value_param_pattern %i.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc11_12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_17.1: i32 = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:     %.loc11_12.1: type = value_of_initializer %int.make_type_32.loc11_12 [template = i32]
// CHECK:STDOUT:     %.loc11_12.2: type = converted %int.make_type_32.loc11_12, %.loc11_12.1 [template = i32]
// CHECK:STDOUT:     %.loc11_17.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:     %.loc11_17.3: <bound method> = bound_method %.loc11_17.1, %.loc11_17.2 [template = constants.%.26]
// CHECK:STDOUT:     %int.convert_checked: init Core.IntLiteral = call %.loc11_17.3(%.loc11_17.1) [template = constants.%.27]
// CHECK:STDOUT:     %.loc11_17.4: Core.IntLiteral = value_of_initializer %int.convert_checked [template = constants.%.27]
// CHECK:STDOUT:     %.loc11_17.5: Core.IntLiteral = converted %.loc11_17.1, %.loc11_17.4 [template = constants.%.27]
// CHECK:STDOUT:     %.loc11_18: type = array_type %.loc11_17.5, i32 [template = constants.%.28]
// CHECK:STDOUT:     %int.make_type_32.loc11_24: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_24.1: type = value_of_initializer %int.make_type_32.loc11_24 [template = i32]
// CHECK:STDOUT:     %.loc11_24.2: type = converted %int.make_type_32.loc11_24, %.loc11_24.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc11_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_32.1: type = value_of_initializer %int.make_type_32.loc11_32 [template = i32]
// CHECK:STDOUT:     %.loc11_32.2: type = converted %int.make_type_32.loc11_32, %.loc11_32.1 [template = i32]
// CHECK:STDOUT:     %arr.param: %.28 = value_param runtime_param0
// CHECK:STDOUT:     %arr: %.28 = bind_name arr, %arr.param
// CHECK:STDOUT:     %i.param: i32 = value_param runtime_param1
// CHECK:STDOUT:     %i: i32 = bind_name i, %i.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc15_11.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc15_11.2: type = converted %int.make_type_32, %.loc15_11.1 [template = i32]
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%arr.param_patt: %.28, %i.param_patt: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr.ref: %.28 = name_ref arr, %arr
// CHECK:STDOUT:   %i.ref: i32 = name_ref i, %i
// CHECK:STDOUT:   %.loc12_15.1: ref %.28 = value_as_ref %arr.ref
// CHECK:STDOUT:   %.loc12_15.2: ref i32 = array_index %.loc12_15.1, %i.ref
// CHECK:STDOUT:   %.loc12_15.3: i32 = bind_value %.loc12_15.2
// CHECK:STDOUT:   return %.loc12_15.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc16_13: i32 = int_value 1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc16_16: i32 = int_value 2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc16_19: i32 = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc16_20.1: %tuple.type = tuple_literal (%.loc16_13, %.loc16_16, %.loc16_19)
// CHECK:STDOUT:   %.loc16_23: i32 = int_value 1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc16_20.2: ref %.28 = temporary_storage
// CHECK:STDOUT:   %.loc16_20.3: i32 = int_value 0 [template = constants.%.32]
// CHECK:STDOUT:   %.loc16_20.4: ref i32 = array_index %.loc16_20.2, %.loc16_20.3
// CHECK:STDOUT:   %.loc16_20.5: init i32 = initialize_from %.loc16_13 to %.loc16_20.4 [template = constants.%.30]
// CHECK:STDOUT:   %.loc16_20.6: i32 = int_value 1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc16_20.7: ref i32 = array_index %.loc16_20.2, %.loc16_20.6
// CHECK:STDOUT:   %.loc16_20.8: init i32 = initialize_from %.loc16_16 to %.loc16_20.7 [template = constants.%.31]
// CHECK:STDOUT:   %.loc16_20.9: i32 = int_value 2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc16_20.10: ref i32 = array_index %.loc16_20.2, %.loc16_20.9
// CHECK:STDOUT:   %.loc16_20.11: init i32 = initialize_from %.loc16_19 to %.loc16_20.10 [template = constants.%.1]
// CHECK:STDOUT:   %.loc16_20.12: init %.28 = array_init (%.loc16_20.5, %.loc16_20.8, %.loc16_20.11) to %.loc16_20.2 [template = constants.%array]
// CHECK:STDOUT:   %.loc16_20.13: init %.28 = converted %.loc16_20.1, %.loc16_20.12 [template = constants.%array]
// CHECK:STDOUT:   %.loc16_20.14: ref %.28 = temporary %.loc16_20.2, %.loc16_20.13
// CHECK:STDOUT:   %.loc16_20.15: %.28 = bind_value %.loc16_20.14
// CHECK:STDOUT:   %F.call: init i32 = call %F.ref(%.loc16_20.15, %.loc16_23)
// CHECK:STDOUT:   %.loc16_25.1: i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc16_25.2: i32 = converted %F.call, %.loc16_25.1
// CHECK:STDOUT:   return %.loc16_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
