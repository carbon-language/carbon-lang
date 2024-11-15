// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/smul.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/smul.carbon

// --- int_mul.carbon

fn Mul(a: i32, b: i32) -> i32 = "int.smul";

var arr: [i32; Mul(3, 2)];
let arr_p: [i32; 6]* = &arr;

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return Mul(a, b);
}

// --- fail_overflow.carbon

package FailOverflow;

fn Mul(a: i32, b: i32) -> i32 = "int.smul";

let a: i32 = Mul(0x7FFF, 0x10000);
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+3]]:14: error: integer overflow in calculation 32768 * 65536 [CompileTimeIntegerOverflow]
// CHECK:STDERR: let b: i32 = Mul(0x8000, 0x10000);
// CHECK:STDERR:              ^~~~
let b: i32 = Mul(0x8000, 0x10000);

// CHECK:STDOUT: --- int_mul.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Mul.type: type = fn_type @Mul [template]
// CHECK:STDOUT:   %Mul: %Mul.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.28: i32 = int_value 3 [template]
// CHECK:STDOUT:   %.29: <bound method> = bound_method %.2, %Convert.15 [template]
// CHECK:STDOUT:   %.30: i32 = int_value 2 [template]
// CHECK:STDOUT:   %.31: i32 = int_value 6 [template]
// CHECK:STDOUT:   %Convert.type.16: type = fn_type @Convert.12 [template]
// CHECK:STDOUT:   %Convert.16: %Convert.type.16 = struct_value () [template]
// CHECK:STDOUT:   %.32: <witness> = interface_witness (%Convert.16) [template]
// CHECK:STDOUT:   %.33: <bound method> = bound_method %.31, %Convert.16 [template]
// CHECK:STDOUT:   %.34: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %.35: type = array_type %.34, i32 [template]
// CHECK:STDOUT:   %.36: type = ptr_type %.35 [template]
// CHECK:STDOUT:   %RuntimeCall.type: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %RuntimeCall: %RuntimeCall.type = struct_value () [template]
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
// CHECK:STDOUT:     .Mul = %Mul.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .arr_p = @__global_init.%arr_p
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Mul.decl: %Mul.type = fn_decl @Mul [template = constants.%Mul] {
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc2_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc2_11.1: type = value_of_initializer %int.make_type_32.loc2_11 [template = i32]
// CHECK:STDOUT:     %.loc2_11.2: type = converted %int.make_type_32.loc2_11, %.loc2_11.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc2_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc2_19.1: type = value_of_initializer %int.make_type_32.loc2_19 [template = i32]
// CHECK:STDOUT:     %.loc2_19.2: type = converted %int.make_type_32.loc2_19, %.loc2_19.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc2_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc2_27.1: type = value_of_initializer %int.make_type_32.loc2_27 [template = i32]
// CHECK:STDOUT:     %.loc2_27.2: type = converted %int.make_type_32.loc2_27, %.loc2_27.1 [template = i32]
// CHECK:STDOUT:     %a.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc4: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %Mul.ref: %Mul.type = name_ref Mul, %Mul.decl [template = constants.%Mul]
// CHECK:STDOUT:   %.loc4_20.1: Core.IntLiteral = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc4_23.1: Core.IntLiteral = int_value 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_20.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc4_20.3: <bound method> = bound_method %.loc4_20.1, %.loc4_20.2 [template = constants.%.27]
// CHECK:STDOUT:   %int.convert_checked.loc4_20: init i32 = call %.loc4_20.3(%.loc4_20.1) [template = constants.%.28]
// CHECK:STDOUT:   %.loc4_20.4: i32 = value_of_initializer %int.convert_checked.loc4_20 [template = constants.%.28]
// CHECK:STDOUT:   %.loc4_20.5: i32 = converted %.loc4_20.1, %.loc4_20.4 [template = constants.%.28]
// CHECK:STDOUT:   %.loc4_23.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc4_23.3: <bound method> = bound_method %.loc4_23.1, %.loc4_23.2 [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc4_23: init i32 = call %.loc4_23.3(%.loc4_23.1) [template = constants.%.30]
// CHECK:STDOUT:   %.loc4_23.4: i32 = value_of_initializer %int.convert_checked.loc4_23 [template = constants.%.30]
// CHECK:STDOUT:   %.loc4_23.5: i32 = converted %.loc4_23.1, %.loc4_23.4 [template = constants.%.30]
// CHECK:STDOUT:   %int.smul: init i32 = call %Mul.ref(%.loc4_20.5, %.loc4_23.5) [template = constants.%.31]
// CHECK:STDOUT:   %.loc4_11.1: type = value_of_initializer %int.make_type_32.loc4 [template = i32]
// CHECK:STDOUT:   %.loc4_11.2: type = converted %int.make_type_32.loc4, %.loc4_11.1 [template = i32]
// CHECK:STDOUT:   %.loc4_19.1: %Convert.type.5 = interface_witness_access constants.%.32, element0 [template = constants.%Convert.16]
// CHECK:STDOUT:   %.loc4_19.2: <bound method> = bound_method %int.smul, %.loc4_19.1 [template = constants.%.33]
// CHECK:STDOUT:   %.loc4_19.3: i32 = value_of_initializer %int.smul [template = constants.%.31]
// CHECK:STDOUT:   %.loc4_19.4: i32 = converted %int.smul, %.loc4_19.3 [template = constants.%.31]
// CHECK:STDOUT:   %int.convert_checked.loc4_19: init Core.IntLiteral = call %.loc4_19.2(%.loc4_19.4) [template = constants.%.34]
// CHECK:STDOUT:   %.loc4_19.5: Core.IntLiteral = value_of_initializer %int.convert_checked.loc4_19 [template = constants.%.34]
// CHECK:STDOUT:   %.loc4_19.6: Core.IntLiteral = converted %int.smul, %.loc4_19.5 [template = constants.%.34]
// CHECK:STDOUT:   %.loc4_25: type = array_type %.loc4_19.6, i32 [template = constants.%.35]
// CHECK:STDOUT:   %arr.var: ref %.35 = var arr
// CHECK:STDOUT:   %arr: ref %.35 = bind_name arr, %arr.var
// CHECK:STDOUT:   %int.make_type_32.loc5: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_18: Core.IntLiteral = int_value 6 [template = constants.%.34]
// CHECK:STDOUT:   %.loc5_13.1: type = value_of_initializer %int.make_type_32.loc5 [template = i32]
// CHECK:STDOUT:   %.loc5_13.2: type = converted %int.make_type_32.loc5, %.loc5_13.1 [template = i32]
// CHECK:STDOUT:   %.loc5_19: type = array_type %.loc5_18, i32 [template = constants.%.35]
// CHECK:STDOUT:   %.loc5_20: type = ptr_type %.35 [template = constants.%.36]
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc7_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc7_19.1: type = value_of_initializer %int.make_type_32.loc7_19 [template = i32]
// CHECK:STDOUT:     %.loc7_19.2: type = converted %int.make_type_32.loc7_19, %.loc7_19.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc7_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc7_27.1: type = value_of_initializer %int.make_type_32.loc7_27 [template = i32]
// CHECK:STDOUT:     %.loc7_27.2: type = converted %int.make_type_32.loc7_27, %.loc7_27.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc7_35: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc7_35.1: type = value_of_initializer %int.make_type_32.loc7_35 [template = i32]
// CHECK:STDOUT:     %.loc7_35.2: type = converted %int.make_type_32.loc7_35, %.loc7_35.1 [template = i32]
// CHECK:STDOUT:     %a.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Mul(%a.param_patt: i32, %b.param_patt: i32) -> i32 = "int.smul";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: i32, %b.param_patt: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Mul.ref: %Mul.type = name_ref Mul, file.%Mul.decl [template = constants.%Mul]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %int.smul: init i32 = call %Mul.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc8_19.1: i32 = value_of_initializer %int.smul
// CHECK:STDOUT:   %.loc8_19.2: i32 = converted %int.smul, %.loc8_19.1
// CHECK:STDOUT:   return %.loc8_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr.ref: ref %.35 = name_ref arr, file.%arr
// CHECK:STDOUT:   %.loc5: %.36 = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: %.36 = bind_name arr_p, %.loc5
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Mul.type: type = fn_type @Mul [template]
// CHECK:STDOUT:   %Mul: %Mul.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32767 [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 65536 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.28: i32 = int_value 32767 [template]
// CHECK:STDOUT:   %.29: <bound method> = bound_method %.2, %Convert.15 [template]
// CHECK:STDOUT:   %.30: i32 = int_value 65536 [template]
// CHECK:STDOUT:   %.31: i32 = int_value 2147418112 [template]
// CHECK:STDOUT:   %.32: Core.IntLiteral = int_value 32768 [template]
// CHECK:STDOUT:   %.33: <bound method> = bound_method %.32, %Convert.15 [template]
// CHECK:STDOUT:   %.34: i32 = int_value 32768 [template]
// CHECK:STDOUT:   %.35: i32 = int_value -2147483648 [template]
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
// CHECK:STDOUT:     .Mul = %Mul.decl
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Mul.decl: %Mul.type = fn_decl @Mul [template = constants.%Mul] {
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc4_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_11.1: type = value_of_initializer %int.make_type_32.loc4_11 [template = i32]
// CHECK:STDOUT:     %.loc4_11.2: type = converted %int.make_type_32.loc4_11, %.loc4_11.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc4_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_19.1: type = value_of_initializer %int.make_type_32.loc4_19 [template = i32]
// CHECK:STDOUT:     %.loc4_19.2: type = converted %int.make_type_32.loc4_19, %.loc4_19.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc4_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_27.1: type = value_of_initializer %int.make_type_32.loc4_27 [template = i32]
// CHECK:STDOUT:     %.loc4_27.2: type = converted %int.make_type_32.loc4_27, %.loc4_27.1 [template = i32]
// CHECK:STDOUT:     %a.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc6: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc6_8.1: type = value_of_initializer %int.make_type_32.loc6 [template = i32]
// CHECK:STDOUT:   %.loc6_8.2: type = converted %int.make_type_32.loc6, %.loc6_8.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc10: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc10_8.1: type = value_of_initializer %int.make_type_32.loc10 [template = i32]
// CHECK:STDOUT:   %.loc10_8.2: type = converted %int.make_type_32.loc10, %.loc10_8.1 [template = i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Mul(%a.param_patt: i32, %b.param_patt: i32) -> i32 = "int.smul";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Mul.ref.loc6: %Mul.type = name_ref Mul, file.%Mul.decl [template = constants.%Mul]
// CHECK:STDOUT:   %.loc6_18.1: Core.IntLiteral = int_value 32767 [template = constants.%.1]
// CHECK:STDOUT:   %.loc6_26.1: Core.IntLiteral = int_value 65536 [template = constants.%.2]
// CHECK:STDOUT:   %.loc6_18.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc6_18.3: <bound method> = bound_method %.loc6_18.1, %.loc6_18.2 [template = constants.%.27]
// CHECK:STDOUT:   %int.convert_checked.loc6_18: init i32 = call %.loc6_18.3(%.loc6_18.1) [template = constants.%.28]
// CHECK:STDOUT:   %.loc6_18.4: i32 = value_of_initializer %int.convert_checked.loc6_18 [template = constants.%.28]
// CHECK:STDOUT:   %.loc6_18.5: i32 = converted %.loc6_18.1, %.loc6_18.4 [template = constants.%.28]
// CHECK:STDOUT:   %.loc6_26.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc6_26.3: <bound method> = bound_method %.loc6_26.1, %.loc6_26.2 [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc6_26: init i32 = call %.loc6_26.3(%.loc6_26.1) [template = constants.%.30]
// CHECK:STDOUT:   %.loc6_26.4: i32 = value_of_initializer %int.convert_checked.loc6_26 [template = constants.%.30]
// CHECK:STDOUT:   %.loc6_26.5: i32 = converted %.loc6_26.1, %.loc6_26.4 [template = constants.%.30]
// CHECK:STDOUT:   %int.smul.loc6: init i32 = call %Mul.ref.loc6(%.loc6_18.5, %.loc6_26.5) [template = constants.%.31]
// CHECK:STDOUT:   %.loc6_34.1: i32 = value_of_initializer %int.smul.loc6 [template = constants.%.31]
// CHECK:STDOUT:   %.loc6_34.2: i32 = converted %int.smul.loc6, %.loc6_34.1 [template = constants.%.31]
// CHECK:STDOUT:   %a: i32 = bind_name a, %.loc6_34.2
// CHECK:STDOUT:   %Mul.ref.loc10: %Mul.type = name_ref Mul, file.%Mul.decl [template = constants.%Mul]
// CHECK:STDOUT:   %.loc10_18.1: Core.IntLiteral = int_value 32768 [template = constants.%.32]
// CHECK:STDOUT:   %.loc10_26.1: Core.IntLiteral = int_value 65536 [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_18.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc10_18.3: <bound method> = bound_method %.loc10_18.1, %.loc10_18.2 [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc10_18: init i32 = call %.loc10_18.3(%.loc10_18.1) [template = constants.%.34]
// CHECK:STDOUT:   %.loc10_18.4: i32 = value_of_initializer %int.convert_checked.loc10_18 [template = constants.%.34]
// CHECK:STDOUT:   %.loc10_18.5: i32 = converted %.loc10_18.1, %.loc10_18.4 [template = constants.%.34]
// CHECK:STDOUT:   %.loc10_26.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc10_26.3: <bound method> = bound_method %.loc10_26.1, %.loc10_26.2 [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc10_26: init i32 = call %.loc10_26.3(%.loc10_26.1) [template = constants.%.30]
// CHECK:STDOUT:   %.loc10_26.4: i32 = value_of_initializer %int.convert_checked.loc10_26 [template = constants.%.30]
// CHECK:STDOUT:   %.loc10_26.5: i32 = converted %.loc10_26.1, %.loc10_26.4 [template = constants.%.30]
// CHECK:STDOUT:   %int.smul.loc10: init i32 = call %Mul.ref.loc10(%.loc10_18.5, %.loc10_26.5) [template = constants.%.35]
// CHECK:STDOUT:   %.loc10_34.1: i32 = value_of_initializer %int.smul.loc10 [template = constants.%.35]
// CHECK:STDOUT:   %.loc10_34.2: i32 = converted %int.smul.loc10, %.loc10_34.1 [template = constants.%.35]
// CHECK:STDOUT:   %b: i32 = bind_name b, %.loc10_34.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
