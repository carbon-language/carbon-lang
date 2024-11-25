// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/sdiv.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/sdiv.carbon

// --- int_div.carbon

fn Div(a: i32, b: i32) -> i32 = "int.sdiv";

var arr: [i32; Div(3, 2)];
let arr_p: [i32; 1]* = &arr;

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return Div(a, b);
}

// --- fail_overflow.carbon

package FailOverflow;

fn Div(a: i32, b: i32) -> i32 = "int.sdiv";
fn Sub(a: i32, b: i32) -> i32 = "int.ssub";
fn Negate(a: i32) -> i32 = "int.snegate";

// -0x7FFF_FFFF / -1 is OK.
let a: i32 = Div(Negate(0x7FFF_FFFF), Negate(1));

// -0x8000_0000 / 1 is OK.
let b: i32 = Div(Sub(Negate(0x7FFF_FFFF), 1), 1);

// -0x8000_0000 / -1 overflows.
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:14: error: integer overflow in calculation -2147483648 / -1 [CompileTimeIntegerOverflow]
// CHECK:STDERR: let c: i32 = Div(Sub(Negate(0x7FFF_FFFF), 1), Negate(1));
// CHECK:STDERR:              ^~~~
// CHECK:STDERR:
let c: i32 = Div(Sub(Negate(0x7FFF_FFFF), 1), Negate(1));

// --- fail_div_by_zero.carbon

package FailDivByZero;

fn Div(a: i32, b: i32) -> i32 = "int.sdiv";

// CHECK:STDERR: fail_div_by_zero.carbon:[[@LINE+4]]:14: error: division by zero [CompileTimeDivisionByZero]
// CHECK:STDERR: let a: i32 = Div(1, 0);
// CHECK:STDERR:              ^~~~
// CHECK:STDERR:
let a: i32 = Div(1, 0);

// CHECK:STDERR: fail_div_by_zero.carbon:[[@LINE+3]]:14: error: division by zero [CompileTimeDivisionByZero]
// CHECK:STDERR: let b: i32 = Div(0, 0);
// CHECK:STDERR:              ^~~~
let b: i32 = Div(0, 0);

// CHECK:STDOUT: --- int_div.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %Div.type: type = fn_type @Div [template]
// CHECK:STDOUT:   %Div: %Div.type = struct_value () [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.28: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.29: <specific function> = specific_function %.28, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.30: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.3, %Convert.14 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.33: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %.34: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.4, @impl.3(%.1) [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.35: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.36: <bound method> = bound_method %.34, %Convert.15 [template]
// CHECK:STDOUT:   %.37: <specific function> = specific_function %.36, @Convert.4(%.1) [template]
// CHECK:STDOUT:   %.38: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.39: type = array_type %.38, %i32 [template]
// CHECK:STDOUT:   %.40: type = ptr_type %.39 [template]
// CHECK:STDOUT:   %RuntimeCall.type: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %RuntimeCall: %RuntimeCall.type = struct_value () [template]
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
// CHECK:STDOUT:     .Div = %Div.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .arr_p = @__global_init.%arr_p
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Div.decl: %Div.type = fn_decl @Div [template = constants.%Div] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc2_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc2_11: init type = call constants.%Int(%.loc2_11.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_11.2: type = value_of_initializer %int.make_type_signed.loc2_11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_11.3: type = converted %int.make_type_signed.loc2_11, %.loc2_11.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc2_19: init type = call constants.%Int(%.loc2_19.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_19.2: type = value_of_initializer %int.make_type_signed.loc2_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_19.3: type = converted %int.make_type_signed.loc2_19, %.loc2_19.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_27.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc2_27: init type = call constants.%Int(%.loc2_27.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_27.2: type = value_of_initializer %int.make_type_signed.loc2_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_27.3: type = converted %int.make_type_signed.loc2_27, %.loc2_27.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc4_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc4: init type = call constants.%Int(%.loc4_11.1) [template = constants.%i32]
// CHECK:STDOUT:   %Div.ref: %Div.type = name_ref Div, %Div.decl [template = constants.%Div]
// CHECK:STDOUT:   %.loc4_20.1: Core.IntLiteral = int_value 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_23.1: Core.IntLiteral = int_value 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_20.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc4_20.3: <bound method> = bound_method %.loc4_20.1, %.loc4_20.2 [template = constants.%.28]
// CHECK:STDOUT:   %.loc4_20.4: <specific function> = specific_function %.loc4_20.3, @Convert.2(constants.%.1) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc4_20: init %i32 = call %.loc4_20.4(%.loc4_20.1) [template = constants.%.30]
// CHECK:STDOUT:   %.loc4_20.5: %i32 = value_of_initializer %int.convert_checked.loc4_20 [template = constants.%.30]
// CHECK:STDOUT:   %.loc4_20.6: %i32 = converted %.loc4_20.1, %.loc4_20.5 [template = constants.%.30]
// CHECK:STDOUT:   %.loc4_23.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc4_23.3: <bound method> = bound_method %.loc4_23.1, %.loc4_23.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc4_23.4: <specific function> = specific_function %.loc4_23.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc4_23: init %i32 = call %.loc4_23.4(%.loc4_23.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc4_23.5: %i32 = value_of_initializer %int.convert_checked.loc4_23 [template = constants.%.33]
// CHECK:STDOUT:   %.loc4_23.6: %i32 = converted %.loc4_23.1, %.loc4_23.5 [template = constants.%.33]
// CHECK:STDOUT:   %int.sdiv: init %i32 = call %Div.ref(%.loc4_20.6, %.loc4_23.6) [template = constants.%.34]
// CHECK:STDOUT:   %.loc4_11.2: type = value_of_initializer %int.make_type_signed.loc4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_11.3: type = converted %int.make_type_signed.loc4, %.loc4_11.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_19.1: %Convert.type.6 = interface_witness_access constants.%.35, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc4_19.2: <bound method> = bound_method %int.sdiv, %.loc4_19.1 [template = constants.%.36]
// CHECK:STDOUT:   %.loc4_19.3: <specific function> = specific_function %.loc4_19.2, @Convert.4(constants.%.1) [template = constants.%.37]
// CHECK:STDOUT:   %.loc4_19.4: %i32 = value_of_initializer %int.sdiv [template = constants.%.34]
// CHECK:STDOUT:   %.loc4_19.5: %i32 = converted %int.sdiv, %.loc4_19.4 [template = constants.%.34]
// CHECK:STDOUT:   %int.convert_checked.loc4_19: init Core.IntLiteral = call %.loc4_19.3(%.loc4_19.5) [template = constants.%.38]
// CHECK:STDOUT:   %.loc4_19.6: Core.IntLiteral = value_of_initializer %int.convert_checked.loc4_19 [template = constants.%.38]
// CHECK:STDOUT:   %.loc4_19.7: Core.IntLiteral = converted %int.sdiv, %.loc4_19.6 [template = constants.%.38]
// CHECK:STDOUT:   %.loc4_25: type = array_type %.loc4_19.7, %i32 [template = constants.%.39]
// CHECK:STDOUT:   %arr.var: ref %.39 = var arr
// CHECK:STDOUT:   %arr: ref %.39 = bind_name arr, %arr.var
// CHECK:STDOUT:   %.loc5_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc5: init type = call constants.%Int(%.loc5_13.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_18: Core.IntLiteral = int_value 1 [template = constants.%.38]
// CHECK:STDOUT:   %.loc5_13.2: type = value_of_initializer %int.make_type_signed.loc5 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_13.3: type = converted %int.make_type_signed.loc5, %.loc5_13.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_19: type = array_type %.loc5_18, %i32 [template = constants.%.39]
// CHECK:STDOUT:   %.loc5_20: type = ptr_type %.39 [template = constants.%.40]
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc7_19: init type = call constants.%Int(%.loc7_19.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_19.2: type = value_of_initializer %int.make_type_signed.loc7_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_19.3: type = converted %int.make_type_signed.loc7_19, %.loc7_19.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_27.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc7_27: init type = call constants.%Int(%.loc7_27.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_27.2: type = value_of_initializer %int.make_type_signed.loc7_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_27.3: type = converted %int.make_type_signed.loc7_27, %.loc7_27.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_35.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc7_35: init type = call constants.%Int(%.loc7_35.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_35.2: type = value_of_initializer %int.make_type_signed.loc7_35 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_35.3: type = converted %int.make_type_signed.loc7_35, %.loc7_35.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Div(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.sdiv";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Div.ref: %Div.type = name_ref Div, file.%Div.decl [template = constants.%Div]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.sdiv: init %i32 = call %Div.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc8_19.1: %i32 = value_of_initializer %int.sdiv
// CHECK:STDOUT:   %.loc8_19.2: %i32 = converted %int.sdiv, %.loc8_19.1
// CHECK:STDOUT:   return %.loc8_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr.ref: ref %.39 = name_ref arr, file.%arr
// CHECK:STDOUT:   %.loc5: %.40 = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: %.40 = bind_name arr_p, %.loc5
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %Div.type: type = fn_type @Div [template]
// CHECK:STDOUT:   %Div: %Div.type = struct_value () [template]
// CHECK:STDOUT:   %Sub.type: type = fn_type @Sub [template]
// CHECK:STDOUT:   %Sub: %Sub.type = struct_value () [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.28: <specific function> = specific_function %.27, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.29: %i32 = int_value 2147483647 [template]
// CHECK:STDOUT:   %.30: %i32 = int_value -2147483647 [template]
// CHECK:STDOUT:   %.31: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.32: <bound method> = bound_method %.31, %Convert.14 [template]
// CHECK:STDOUT:   %.33: <specific function> = specific_function %.32, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.34: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.35: %i32 = int_value -1 [template]
// CHECK:STDOUT:   %.36: %i32 = int_value -2147483648 [template]
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
// CHECK:STDOUT:     .Div = %Div.decl
// CHECK:STDOUT:     .Sub = %Sub.decl
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:     .c = @__global_init.%c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Div.decl: %Div.type = fn_decl @Div [template = constants.%Div] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_11: init type = call constants.%Int(%.loc4_11.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_11.2: type = value_of_initializer %int.make_type_signed.loc4_11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_11.3: type = converted %int.make_type_signed.loc4_11, %.loc4_11.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_19: init type = call constants.%Int(%.loc4_19.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_19.2: type = value_of_initializer %int.make_type_signed.loc4_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_19.3: type = converted %int.make_type_signed.loc4_19, %.loc4_19.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_27.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_27: init type = call constants.%Int(%.loc4_27.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_27.2: type = value_of_initializer %int.make_type_signed.loc4_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_27.3: type = converted %int.make_type_signed.loc4_27, %.loc4_27.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Sub.decl: %Sub.type = fn_decl @Sub [template = constants.%Sub] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc5_11: init type = call constants.%Int(%.loc5_11.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_11.2: type = value_of_initializer %int.make_type_signed.loc5_11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_11.3: type = converted %int.make_type_signed.loc5_11, %.loc5_11.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc5_19: init type = call constants.%Int(%.loc5_19.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_19.2: type = value_of_initializer %int.make_type_signed.loc5_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_19.3: type = converted %int.make_type_signed.loc5_19, %.loc5_19.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_27.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc5_27: init type = call constants.%Int(%.loc5_27.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_27.2: type = value_of_initializer %int.make_type_signed.loc5_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_27.3: type = converted %int.make_type_signed.loc5_27, %.loc5_27.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate.decl: %Negate.type = fn_decl @Negate [template = constants.%Negate] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_14.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc6_14: init type = call constants.%Int(%.loc6_14.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_14.2: type = value_of_initializer %int.make_type_signed.loc6_14 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_14.3: type = converted %int.make_type_signed.loc6_14, %.loc6_14.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_22.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc6_22: init type = call constants.%Int(%.loc6_22.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_22.2: type = value_of_initializer %int.make_type_signed.loc6_22 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_22.3: type = converted %int.make_type_signed.loc6_22, %.loc6_22.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_8.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc9: init type = call constants.%Int(%.loc9_8.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_8.2: type = value_of_initializer %int.make_type_signed.loc9 [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_8.3: type = converted %int.make_type_signed.loc9, %.loc9_8.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%.loc12_8.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8.2: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8.3: type = converted %int.make_type_signed.loc12, %.loc12_8.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_8.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc19: init type = call constants.%Int(%.loc19_8.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_8.2: type = value_of_initializer %int.make_type_signed.loc19 [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_8.3: type = converted %int.make_type_signed.loc19, %.loc19_8.2 [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Div(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.sdiv";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Sub(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.ssub";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a.param_patt: %i32) -> %i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Div.ref.loc9: %Div.type = name_ref Div, file.%Div.decl [template = constants.%Div]
// CHECK:STDOUT:   %Negate.ref.loc9_18: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc9_25.1: Core.IntLiteral = int_value 2147483647 [template = constants.%.2]
// CHECK:STDOUT:   %.loc9_25.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc9_25.3: <bound method> = bound_method %.loc9_25.1, %.loc9_25.2 [template = constants.%.27]
// CHECK:STDOUT:   %.loc9_25.4: <specific function> = specific_function %.loc9_25.3, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc9_25: init %i32 = call %.loc9_25.4(%.loc9_25.1) [template = constants.%.29]
// CHECK:STDOUT:   %.loc9_25.5: %i32 = value_of_initializer %int.convert_checked.loc9_25 [template = constants.%.29]
// CHECK:STDOUT:   %.loc9_25.6: %i32 = converted %.loc9_25.1, %.loc9_25.5 [template = constants.%.29]
// CHECK:STDOUT:   %int.snegate.loc9_24: init %i32 = call %Negate.ref.loc9_18(%.loc9_25.6) [template = constants.%.30]
// CHECK:STDOUT:   %Negate.ref.loc9_39: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc9_46.1: Core.IntLiteral = int_value 1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc9_46.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc9_46.3: <bound method> = bound_method %.loc9_46.1, %.loc9_46.2 [template = constants.%.32]
// CHECK:STDOUT:   %.loc9_46.4: <specific function> = specific_function %.loc9_46.3, @Convert.2(constants.%.1) [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc9_46: init %i32 = call %.loc9_46.4(%.loc9_46.1) [template = constants.%.34]
// CHECK:STDOUT:   %.loc9_46.5: %i32 = value_of_initializer %int.convert_checked.loc9_46 [template = constants.%.34]
// CHECK:STDOUT:   %.loc9_46.6: %i32 = converted %.loc9_46.1, %.loc9_46.5 [template = constants.%.34]
// CHECK:STDOUT:   %int.snegate.loc9_45: init %i32 = call %Negate.ref.loc9_39(%.loc9_46.6) [template = constants.%.35]
// CHECK:STDOUT:   %.loc9_24.1: %i32 = value_of_initializer %int.snegate.loc9_24 [template = constants.%.30]
// CHECK:STDOUT:   %.loc9_24.2: %i32 = converted %int.snegate.loc9_24, %.loc9_24.1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc9_45.1: %i32 = value_of_initializer %int.snegate.loc9_45 [template = constants.%.35]
// CHECK:STDOUT:   %.loc9_45.2: %i32 = converted %int.snegate.loc9_45, %.loc9_45.1 [template = constants.%.35]
// CHECK:STDOUT:   %int.sdiv.loc9: init %i32 = call %Div.ref.loc9(%.loc9_24.2, %.loc9_45.2) [template = constants.%.29]
// CHECK:STDOUT:   %.loc9_49.1: %i32 = value_of_initializer %int.sdiv.loc9 [template = constants.%.29]
// CHECK:STDOUT:   %.loc9_49.2: %i32 = converted %int.sdiv.loc9, %.loc9_49.1 [template = constants.%.29]
// CHECK:STDOUT:   %a: %i32 = bind_name a, %.loc9_49.2
// CHECK:STDOUT:   %Div.ref.loc12: %Div.type = name_ref Div, file.%Div.decl [template = constants.%Div]
// CHECK:STDOUT:   %Sub.ref.loc12: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %Negate.ref.loc12: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc12_29.1: Core.IntLiteral = int_value 2147483647 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_29.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_29.3: <bound method> = bound_method %.loc12_29.1, %.loc12_29.2 [template = constants.%.27]
// CHECK:STDOUT:   %.loc12_29.4: <specific function> = specific_function %.loc12_29.3, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc12_29: init %i32 = call %.loc12_29.4(%.loc12_29.1) [template = constants.%.29]
// CHECK:STDOUT:   %.loc12_29.5: %i32 = value_of_initializer %int.convert_checked.loc12_29 [template = constants.%.29]
// CHECK:STDOUT:   %.loc12_29.6: %i32 = converted %.loc12_29.1, %.loc12_29.5 [template = constants.%.29]
// CHECK:STDOUT:   %int.snegate.loc12: init %i32 = call %Negate.ref.loc12(%.loc12_29.6) [template = constants.%.30]
// CHECK:STDOUT:   %.loc12_43.1: Core.IntLiteral = int_value 1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc12_28.1: %i32 = value_of_initializer %int.snegate.loc12 [template = constants.%.30]
// CHECK:STDOUT:   %.loc12_28.2: %i32 = converted %int.snegate.loc12, %.loc12_28.1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc12_43.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_43.3: <bound method> = bound_method %.loc12_43.1, %.loc12_43.2 [template = constants.%.32]
// CHECK:STDOUT:   %.loc12_43.4: <specific function> = specific_function %.loc12_43.3, @Convert.2(constants.%.1) [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc12_43: init %i32 = call %.loc12_43.4(%.loc12_43.1) [template = constants.%.34]
// CHECK:STDOUT:   %.loc12_43.5: %i32 = value_of_initializer %int.convert_checked.loc12_43 [template = constants.%.34]
// CHECK:STDOUT:   %.loc12_43.6: %i32 = converted %.loc12_43.1, %.loc12_43.5 [template = constants.%.34]
// CHECK:STDOUT:   %int.ssub.loc12: init %i32 = call %Sub.ref.loc12(%.loc12_28.2, %.loc12_43.6) [template = constants.%.36]
// CHECK:STDOUT:   %.loc12_47.1: Core.IntLiteral = int_value 1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc12_21.1: %i32 = value_of_initializer %int.ssub.loc12 [template = constants.%.36]
// CHECK:STDOUT:   %.loc12_21.2: %i32 = converted %int.ssub.loc12, %.loc12_21.1 [template = constants.%.36]
// CHECK:STDOUT:   %.loc12_47.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_47.3: <bound method> = bound_method %.loc12_47.1, %.loc12_47.2 [template = constants.%.32]
// CHECK:STDOUT:   %.loc12_47.4: <specific function> = specific_function %.loc12_47.3, @Convert.2(constants.%.1) [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc12_47: init %i32 = call %.loc12_47.4(%.loc12_47.1) [template = constants.%.34]
// CHECK:STDOUT:   %.loc12_47.5: %i32 = value_of_initializer %int.convert_checked.loc12_47 [template = constants.%.34]
// CHECK:STDOUT:   %.loc12_47.6: %i32 = converted %.loc12_47.1, %.loc12_47.5 [template = constants.%.34]
// CHECK:STDOUT:   %int.sdiv.loc12: init %i32 = call %Div.ref.loc12(%.loc12_21.2, %.loc12_47.6) [template = constants.%.36]
// CHECK:STDOUT:   %.loc12_49.1: %i32 = value_of_initializer %int.sdiv.loc12 [template = constants.%.36]
// CHECK:STDOUT:   %.loc12_49.2: %i32 = converted %int.sdiv.loc12, %.loc12_49.1 [template = constants.%.36]
// CHECK:STDOUT:   %b: %i32 = bind_name b, %.loc12_49.2
// CHECK:STDOUT:   %Div.ref.loc19: %Div.type = name_ref Div, file.%Div.decl [template = constants.%Div]
// CHECK:STDOUT:   %Sub.ref.loc19: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %Negate.ref.loc19_22: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc19_29.1: Core.IntLiteral = int_value 2147483647 [template = constants.%.2]
// CHECK:STDOUT:   %.loc19_29.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc19_29.3: <bound method> = bound_method %.loc19_29.1, %.loc19_29.2 [template = constants.%.27]
// CHECK:STDOUT:   %.loc19_29.4: <specific function> = specific_function %.loc19_29.3, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc19_29: init %i32 = call %.loc19_29.4(%.loc19_29.1) [template = constants.%.29]
// CHECK:STDOUT:   %.loc19_29.5: %i32 = value_of_initializer %int.convert_checked.loc19_29 [template = constants.%.29]
// CHECK:STDOUT:   %.loc19_29.6: %i32 = converted %.loc19_29.1, %.loc19_29.5 [template = constants.%.29]
// CHECK:STDOUT:   %int.snegate.loc19_28: init %i32 = call %Negate.ref.loc19_22(%.loc19_29.6) [template = constants.%.30]
// CHECK:STDOUT:   %.loc19_43.1: Core.IntLiteral = int_value 1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc19_28.1: %i32 = value_of_initializer %int.snegate.loc19_28 [template = constants.%.30]
// CHECK:STDOUT:   %.loc19_28.2: %i32 = converted %int.snegate.loc19_28, %.loc19_28.1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc19_43.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc19_43.3: <bound method> = bound_method %.loc19_43.1, %.loc19_43.2 [template = constants.%.32]
// CHECK:STDOUT:   %.loc19_43.4: <specific function> = specific_function %.loc19_43.3, @Convert.2(constants.%.1) [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc19_43: init %i32 = call %.loc19_43.4(%.loc19_43.1) [template = constants.%.34]
// CHECK:STDOUT:   %.loc19_43.5: %i32 = value_of_initializer %int.convert_checked.loc19_43 [template = constants.%.34]
// CHECK:STDOUT:   %.loc19_43.6: %i32 = converted %.loc19_43.1, %.loc19_43.5 [template = constants.%.34]
// CHECK:STDOUT:   %int.ssub.loc19: init %i32 = call %Sub.ref.loc19(%.loc19_28.2, %.loc19_43.6) [template = constants.%.36]
// CHECK:STDOUT:   %Negate.ref.loc19_47: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc19_54.1: Core.IntLiteral = int_value 1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc19_54.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc19_54.3: <bound method> = bound_method %.loc19_54.1, %.loc19_54.2 [template = constants.%.32]
// CHECK:STDOUT:   %.loc19_54.4: <specific function> = specific_function %.loc19_54.3, @Convert.2(constants.%.1) [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc19_54: init %i32 = call %.loc19_54.4(%.loc19_54.1) [template = constants.%.34]
// CHECK:STDOUT:   %.loc19_54.5: %i32 = value_of_initializer %int.convert_checked.loc19_54 [template = constants.%.34]
// CHECK:STDOUT:   %.loc19_54.6: %i32 = converted %.loc19_54.1, %.loc19_54.5 [template = constants.%.34]
// CHECK:STDOUT:   %int.snegate.loc19_53: init %i32 = call %Negate.ref.loc19_47(%.loc19_54.6) [template = constants.%.35]
// CHECK:STDOUT:   %.loc19_21.1: %i32 = value_of_initializer %int.ssub.loc19 [template = constants.%.36]
// CHECK:STDOUT:   %.loc19_21.2: %i32 = converted %int.ssub.loc19, %.loc19_21.1 [template = constants.%.36]
// CHECK:STDOUT:   %.loc19_53.1: %i32 = value_of_initializer %int.snegate.loc19_53 [template = constants.%.35]
// CHECK:STDOUT:   %.loc19_53.2: %i32 = converted %int.snegate.loc19_53, %.loc19_53.1 [template = constants.%.35]
// CHECK:STDOUT:   %int.sdiv.loc19: init %i32 = call %Div.ref.loc19(%.loc19_21.2, %.loc19_53.2) [template = constants.%.36]
// CHECK:STDOUT:   %.loc19_57.1: %i32 = value_of_initializer %int.sdiv.loc19 [template = constants.%.36]
// CHECK:STDOUT:   %.loc19_57.2: %i32 = converted %int.sdiv.loc19, %.loc19_57.1 [template = constants.%.36]
// CHECK:STDOUT:   %c: %i32 = bind_name c, %.loc19_57.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_div_by_zero.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %Div.type: type = fn_type @Div [template]
// CHECK:STDOUT:   %Div: %Div.type = struct_value () [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.28: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.29: <specific function> = specific_function %.28, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.30: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.3, %Convert.14 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.33: %i32 = int_value 0 [template]
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
// CHECK:STDOUT:     .Div = %Div.decl
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Div.decl: %Div.type = fn_decl @Div [template = constants.%Div] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_11: init type = call constants.%Int(%.loc4_11.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_11.2: type = value_of_initializer %int.make_type_signed.loc4_11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_11.3: type = converted %int.make_type_signed.loc4_11, %.loc4_11.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_19: init type = call constants.%Int(%.loc4_19.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_19.2: type = value_of_initializer %int.make_type_signed.loc4_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_19.3: type = converted %int.make_type_signed.loc4_19, %.loc4_19.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_27.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_27: init type = call constants.%Int(%.loc4_27.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_27.2: type = value_of_initializer %int.make_type_signed.loc4_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_27.3: type = converted %int.make_type_signed.loc4_27, %.loc4_27.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_8.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc10: init type = call constants.%Int(%.loc10_8.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc10_8.2: type = value_of_initializer %int.make_type_signed.loc10 [template = constants.%i32]
// CHECK:STDOUT:   %.loc10_8.3: type = converted %int.make_type_signed.loc10, %.loc10_8.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_8.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc15: init type = call constants.%Int(%.loc15_8.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_8.2: type = value_of_initializer %int.make_type_signed.loc15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_8.3: type = converted %int.make_type_signed.loc15, %.loc15_8.2 [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Div(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.sdiv";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Div.ref.loc10: %Div.type = name_ref Div, file.%Div.decl [template = constants.%Div]
// CHECK:STDOUT:   %.loc10_18.1: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_21.1: Core.IntLiteral = int_value 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc10_18.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc10_18.3: <bound method> = bound_method %.loc10_18.1, %.loc10_18.2 [template = constants.%.28]
// CHECK:STDOUT:   %.loc10_18.4: <specific function> = specific_function %.loc10_18.3, @Convert.2(constants.%.1) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc10_18: init %i32 = call %.loc10_18.4(%.loc10_18.1) [template = constants.%.30]
// CHECK:STDOUT:   %.loc10_18.5: %i32 = value_of_initializer %int.convert_checked.loc10_18 [template = constants.%.30]
// CHECK:STDOUT:   %.loc10_18.6: %i32 = converted %.loc10_18.1, %.loc10_18.5 [template = constants.%.30]
// CHECK:STDOUT:   %.loc10_21.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc10_21.3: <bound method> = bound_method %.loc10_21.1, %.loc10_21.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc10_21.4: <specific function> = specific_function %.loc10_21.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc10_21: init %i32 = call %.loc10_21.4(%.loc10_21.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc10_21.5: %i32 = value_of_initializer %int.convert_checked.loc10_21 [template = constants.%.33]
// CHECK:STDOUT:   %.loc10_21.6: %i32 = converted %.loc10_21.1, %.loc10_21.5 [template = constants.%.33]
// CHECK:STDOUT:   %int.sdiv.loc10: init %i32 = call %Div.ref.loc10(%.loc10_18.6, %.loc10_21.6) [template = <error>]
// CHECK:STDOUT:   %.loc10_23.1: %i32 = value_of_initializer %int.sdiv.loc10 [template = <error>]
// CHECK:STDOUT:   %.loc10_23.2: %i32 = converted %int.sdiv.loc10, %.loc10_23.1 [template = <error>]
// CHECK:STDOUT:   %a: %i32 = bind_name a, %.loc10_23.2
// CHECK:STDOUT:   %Div.ref.loc15: %Div.type = name_ref Div, file.%Div.decl [template = constants.%Div]
// CHECK:STDOUT:   %.loc15_18.1: Core.IntLiteral = int_value 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc15_21.1: Core.IntLiteral = int_value 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc15_18.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc15_18.3: <bound method> = bound_method %.loc15_18.1, %.loc15_18.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc15_18.4: <specific function> = specific_function %.loc15_18.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc15_18: init %i32 = call %.loc15_18.4(%.loc15_18.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc15_18.5: %i32 = value_of_initializer %int.convert_checked.loc15_18 [template = constants.%.33]
// CHECK:STDOUT:   %.loc15_18.6: %i32 = converted %.loc15_18.1, %.loc15_18.5 [template = constants.%.33]
// CHECK:STDOUT:   %.loc15_21.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc15_21.3: <bound method> = bound_method %.loc15_21.1, %.loc15_21.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc15_21.4: <specific function> = specific_function %.loc15_21.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc15_21: init %i32 = call %.loc15_21.4(%.loc15_21.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc15_21.5: %i32 = value_of_initializer %int.convert_checked.loc15_21 [template = constants.%.33]
// CHECK:STDOUT:   %.loc15_21.6: %i32 = converted %.loc15_21.1, %.loc15_21.5 [template = constants.%.33]
// CHECK:STDOUT:   %int.sdiv.loc15: init %i32 = call %Div.ref.loc15(%.loc15_18.6, %.loc15_21.6) [template = <error>]
// CHECK:STDOUT:   %.loc15_23.1: %i32 = value_of_initializer %int.sdiv.loc15 [template = <error>]
// CHECK:STDOUT:   %.loc15_23.2: %i32 = converted %int.sdiv.loc15, %.loc15_23.1 [template = <error>]
// CHECK:STDOUT:   %b: %i32 = bind_name b, %.loc15_23.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
