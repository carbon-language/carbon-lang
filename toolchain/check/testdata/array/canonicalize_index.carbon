// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/canonicalize_index.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/canonicalize_index.carbon

fn Add(a: i32, b: i32) -> i32 = "int.sadd";
fn ConvertToU32(a: i32) -> u32 = "int.convert_checked";

var a: [i32; Add(1, 2)] = (1, 2, 3);
let b: [i32; 3]* = &a;
let c: [i32; ConvertToU32(3)]* = &a;

// CHECK:STDOUT: --- canonicalize_index.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Add.type: type = fn_type @Add [template]
// CHECK:STDOUT:   %Add: %Add.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type unsigned, %.1 [template]
// CHECK:STDOUT:   %ConvertToU32.type: type = fn_type @ConvertToU32 [template]
// CHECK:STDOUT:   %ConvertToU32: %ConvertToU32.type = struct_value () [template]
// CHECK:STDOUT:   %.3: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.4: i32 = int_value 2 [template]
// CHECK:STDOUT:   %.5: i32 = int_value 3 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.29: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.30: <bound method> = bound_method %.5, %Convert.15 [template]
// CHECK:STDOUT:   %.31: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.32: type = array_type %.31, i32 [template]
// CHECK:STDOUT:   %.33: type = ptr_type %.32 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.34: i32 = int_value 0 [template]
// CHECK:STDOUT:   %array: %.32 = tuple_value (%.3, %.4, %.5) [template]
// CHECK:STDOUT:   %.35: %.2 = int_value 3 [template]
// CHECK:STDOUT:   %Convert.type.16: type = fn_type @Convert.5, @impl.6(%.1) [template]
// CHECK:STDOUT:   %Convert.16: %Convert.type.16 = struct_value () [template]
// CHECK:STDOUT:   %.37: <witness> = interface_witness (%Convert.16) [template]
// CHECK:STDOUT:   %.38: <bound method> = bound_method %.35, %Convert.16 [template]
// CHECK:STDOUT:   %.39: <specific function> = specific_function %.38, @Convert.5(%.1) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .UInt = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.3
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Add = %Add.decl
// CHECK:STDOUT:     .ConvertToU32 = %ConvertToU32.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:     .c = @__global_init.%c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Add.decl: %Add.type = fn_decl @Add [template = constants.%Add] {
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc11_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_11.1: type = value_of_initializer %int.make_type_32.loc11_11 [template = i32]
// CHECK:STDOUT:     %.loc11_11.2: type = converted %int.make_type_32.loc11_11, %.loc11_11.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc11_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_19.1: type = value_of_initializer %int.make_type_32.loc11_19 [template = i32]
// CHECK:STDOUT:     %.loc11_19.2: type = converted %int.make_type_32.loc11_19, %.loc11_19.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc11_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_27.1: type = value_of_initializer %int.make_type_32.loc11_27 [template = i32]
// CHECK:STDOUT:     %.loc11_27.2: type = converted %int.make_type_32.loc11_27, %.loc11_27.1 [template = i32]
// CHECK:STDOUT:     %a.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertToU32.decl: %ConvertToU32.type = fn_decl @ConvertToU32 [template = constants.%ConvertToU32] {
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.2 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc12_20.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc12_20.2: type = converted %int.make_type_32, %.loc12_20.1 [template = i32]
// CHECK:STDOUT:     %.loc12_28.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_unsigned: init type = call constants.%UInt(%.loc12_28.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc12_28.2: type = value_of_initializer %int.make_type_unsigned [template = constants.%.2]
// CHECK:STDOUT:     %.loc12_28.3: type = converted %int.make_type_unsigned, %.loc12_28.2 [template = constants.%.2]
// CHECK:STDOUT:     %a.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %.2 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %Add.ref: %Add.type = name_ref Add, %Add.decl [template = constants.%Add]
// CHECK:STDOUT:   %.loc14_18: i32 = int_value 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc14_21: i32 = int_value 2 [template = constants.%.4]
// CHECK:STDOUT:   %int.sadd: init i32 = call %Add.ref(%.loc14_18, %.loc14_21) [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_9.1: type = value_of_initializer %int.make_type_32.loc14 [template = i32]
// CHECK:STDOUT:   %.loc14_9.2: type = converted %int.make_type_32.loc14, %.loc14_9.1 [template = i32]
// CHECK:STDOUT:   %.loc14_17.1: %Convert.type.2 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc14_17.2: <bound method> = bound_method %int.sadd, %.loc14_17.1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc14_17.3: i32 = value_of_initializer %int.sadd [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_17.4: i32 = converted %int.sadd, %.loc14_17.3 [template = constants.%.5]
// CHECK:STDOUT:   %int.convert_checked.loc14: init Core.IntLiteral = call %.loc14_17.2(%.loc14_17.4) [template = constants.%.31]
// CHECK:STDOUT:   %.loc14_17.5: Core.IntLiteral = value_of_initializer %int.convert_checked.loc14 [template = constants.%.31]
// CHECK:STDOUT:   %.loc14_17.6: Core.IntLiteral = converted %int.sadd, %.loc14_17.5 [template = constants.%.31]
// CHECK:STDOUT:   %.loc14_23: type = array_type %.loc14_17.6, i32 [template = constants.%.32]
// CHECK:STDOUT:   %a.var: ref %.32 = var a
// CHECK:STDOUT:   %a: ref %.32 = bind_name a, %a.var
// CHECK:STDOUT:   %int.make_type_32.loc15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc15_14.1: i32 = int_value 3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc15_9.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:   %.loc15_9.2: type = converted %int.make_type_32.loc15, %.loc15_9.1 [template = i32]
// CHECK:STDOUT:   %.loc15_14.2: %Convert.type.2 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc15_14.3: <bound method> = bound_method %.loc15_14.1, %.loc15_14.2 [template = constants.%.30]
// CHECK:STDOUT:   %int.convert_checked.loc15: init Core.IntLiteral = call %.loc15_14.3(%.loc15_14.1) [template = constants.%.31]
// CHECK:STDOUT:   %.loc15_14.4: Core.IntLiteral = value_of_initializer %int.convert_checked.loc15 [template = constants.%.31]
// CHECK:STDOUT:   %.loc15_14.5: Core.IntLiteral = converted %.loc15_14.1, %.loc15_14.4 [template = constants.%.31]
// CHECK:STDOUT:   %.loc15_15: type = array_type %.loc15_14.5, i32 [template = constants.%.32]
// CHECK:STDOUT:   %.loc15_16: type = ptr_type %.32 [template = constants.%.33]
// CHECK:STDOUT:   %int.make_type_32.loc16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %ConvertToU32.ref: %ConvertToU32.type = name_ref ConvertToU32, %ConvertToU32.decl [template = constants.%ConvertToU32]
// CHECK:STDOUT:   %.loc16_27: i32 = int_value 3 [template = constants.%.5]
// CHECK:STDOUT:   %int.convert_checked.loc16_26.1: init %.2 = call %ConvertToU32.ref(%.loc16_27) [template = constants.%.35]
// CHECK:STDOUT:   %.loc16_9.1: type = value_of_initializer %int.make_type_32.loc16 [template = i32]
// CHECK:STDOUT:   %.loc16_9.2: type = converted %int.make_type_32.loc16, %.loc16_9.1 [template = i32]
// CHECK:STDOUT:   %.loc16_26.1: %Convert.type.2 = interface_witness_access constants.%.37, element0 [template = constants.%Convert.16]
// CHECK:STDOUT:   %.loc16_26.2: <bound method> = bound_method %int.convert_checked.loc16_26.1, %.loc16_26.1 [template = constants.%.38]
// CHECK:STDOUT:   %.loc16_26.3: <specific function> = specific_function %.loc16_26.2, @Convert.5(constants.%.1) [template = constants.%.39]
// CHECK:STDOUT:   %.loc16_26.4: %.2 = value_of_initializer %int.convert_checked.loc16_26.1 [template = constants.%.35]
// CHECK:STDOUT:   %.loc16_26.5: %.2 = converted %int.convert_checked.loc16_26.1, %.loc16_26.4 [template = constants.%.35]
// CHECK:STDOUT:   %int.convert_checked.loc16_26.2: init Core.IntLiteral = call %.loc16_26.3(%.loc16_26.5) [template = constants.%.31]
// CHECK:STDOUT:   %.loc16_26.6: Core.IntLiteral = value_of_initializer %int.convert_checked.loc16_26.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc16_26.7: Core.IntLiteral = converted %int.convert_checked.loc16_26.1, %.loc16_26.6 [template = constants.%.31]
// CHECK:STDOUT:   %.loc16_29: type = array_type %.loc16_26.7, i32 [template = constants.%.32]
// CHECK:STDOUT:   %.loc16_30: type = ptr_type %.32 [template = constants.%.33]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Add(%a.param_patt: i32, %b.param_patt: i32) -> i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertToU32(%a.param_patt: i32) -> %.2 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_28: i32 = int_value 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc14_31: i32 = int_value 2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_34: i32 = int_value 3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_35.1: %tuple.type = tuple_literal (%.loc14_28, %.loc14_31, %.loc14_34)
// CHECK:STDOUT:   %.loc14_35.2: i32 = int_value 0 [template = constants.%.34]
// CHECK:STDOUT:   %.loc14_35.3: ref i32 = array_index file.%a.var, %.loc14_35.2
// CHECK:STDOUT:   %.loc14_35.4: init i32 = initialize_from %.loc14_28 to %.loc14_35.3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc14_35.5: i32 = int_value 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc14_35.6: ref i32 = array_index file.%a.var, %.loc14_35.5
// CHECK:STDOUT:   %.loc14_35.7: init i32 = initialize_from %.loc14_31 to %.loc14_35.6 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_35.8: i32 = int_value 2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_35.9: ref i32 = array_index file.%a.var, %.loc14_35.8
// CHECK:STDOUT:   %.loc14_35.10: init i32 = initialize_from %.loc14_34 to %.loc14_35.9 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_35.11: init %.32 = array_init (%.loc14_35.4, %.loc14_35.7, %.loc14_35.10) to file.%a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc14_36: init %.32 = converted %.loc14_35.1, %.loc14_35.11 [template = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc14_36
// CHECK:STDOUT:   %a.ref.loc15: ref %.32 = name_ref a, file.%a
// CHECK:STDOUT:   %.loc15: %.33 = addr_of %a.ref.loc15
// CHECK:STDOUT:   %b: %.33 = bind_name b, %.loc15
// CHECK:STDOUT:   %a.ref.loc16: ref %.32 = name_ref a, file.%a
// CHECK:STDOUT:   %.loc16: %.33 = addr_of %a.ref.loc16
// CHECK:STDOUT:   %c: %.33 = bind_name c, %.loc16
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
