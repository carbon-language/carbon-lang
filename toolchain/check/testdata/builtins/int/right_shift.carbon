// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/right_shift.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/right_shift.carbon

// --- int_right_shift.carbon

fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";

var arr: [i32; RightShift(22, 2)];
let arr_p: [i32; 5]* = &arr;

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return RightShift(a, b);
}

// TODO: Test mixed types for LHS and RHS.

// --- arith_shift.carbon

// TODO: Also test unsigned / logical right shift.

package ArithShift;

fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
fn Negate(a: i32) -> i32 = "int.snegate";

// -1 >> 1 is -1.
var arr1: [i32; Negate(RightShift(Negate(1), 1))];
let arr1_p: [i32; 1]* = &arr1;

// -10 >> 2 is -3.
var arr2: [i32; Negate(RightShift(Negate(10), 2))];
let arr2_p: [i32; 3]* = &arr2;

// --- fail_bad_shift.carbon

package BadShift;

fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
fn Negate(a: i32) -> i32 = "int.snegate";

// Shift greater than size is disallowed.
let size_1: i32 = RightShift(1, 31);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:19: error: shift distance not in range [0, 32) in 1 >> 32 [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let size_2: i32 = RightShift(1, 32);
// CHECK:STDERR:                   ^~~~~~~~~~~
// CHECK:STDERR:
let size_2: i32 = RightShift(1, 32);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:19: error: shift distance not in range [0, 32) in 1 >> 33 [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let size_3: i32 = RightShift(1, 33);
// CHECK:STDERR:                   ^~~~~~~~~~~
// CHECK:STDERR:
let size_3: i32 = RightShift(1, 33);

// Negative shifts aren't allowed either.
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+3]]:21: error: shift distance not in range [0, 32) in 1 >> -1 [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let negative: i32 = RightShift(1, Negate(1));
// CHECK:STDERR:                     ^~~~~~~~~~~
let negative: i32 = RightShift(1, Negate(1));

// CHECK:STDOUT: --- int_right_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %RightShift.type: type = fn_type @RightShift [template]
// CHECK:STDOUT:   %RightShift: %RightShift.type = struct_value () [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 22 [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.28: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.29: <specific function> = specific_function %.28, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.30: %i32 = int_value 22 [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.3, %Convert.14 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.33: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %.34: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.4, @impl.3(%.1) [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.35: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.36: <bound method> = bound_method %.34, %Convert.15 [template]
// CHECK:STDOUT:   %.37: <specific function> = specific_function %.36, @Convert.4(%.1) [template]
// CHECK:STDOUT:   %.38: Core.IntLiteral = int_value 5 [template]
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
// CHECK:STDOUT:     .RightShift = %RightShift.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .arr_p = @__global_init.%arr_p
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %RightShift.decl: %RightShift.type = fn_decl @RightShift [template = constants.%RightShift] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc2_18.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc2_18: init type = call constants.%Int(%.loc2_18.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_18.2: type = value_of_initializer %int.make_type_signed.loc2_18 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_18.3: type = converted %int.make_type_signed.loc2_18, %.loc2_18.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_26.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc2_26: init type = call constants.%Int(%.loc2_26.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_26.2: type = value_of_initializer %int.make_type_signed.loc2_26 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_26.3: type = converted %int.make_type_signed.loc2_26, %.loc2_26.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_34.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc2_34: init type = call constants.%Int(%.loc2_34.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_34.2: type = value_of_initializer %int.make_type_signed.loc2_34 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_34.3: type = converted %int.make_type_signed.loc2_34, %.loc2_34.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc4_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc4: init type = call constants.%Int(%.loc4_11.1) [template = constants.%i32]
// CHECK:STDOUT:   %RightShift.ref: %RightShift.type = name_ref RightShift, %RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %.loc4_27.1: Core.IntLiteral = int_value 22 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_31.1: Core.IntLiteral = int_value 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_27.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc4_27.3: <bound method> = bound_method %.loc4_27.1, %.loc4_27.2 [template = constants.%.28]
// CHECK:STDOUT:   %.loc4_27.4: <specific function> = specific_function %.loc4_27.3, @Convert.2(constants.%.1) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc4_27: init %i32 = call %.loc4_27.4(%.loc4_27.1) [template = constants.%.30]
// CHECK:STDOUT:   %.loc4_27.5: %i32 = value_of_initializer %int.convert_checked.loc4_27 [template = constants.%.30]
// CHECK:STDOUT:   %.loc4_27.6: %i32 = converted %.loc4_27.1, %.loc4_27.5 [template = constants.%.30]
// CHECK:STDOUT:   %.loc4_31.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc4_31.3: <bound method> = bound_method %.loc4_31.1, %.loc4_31.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc4_31.4: <specific function> = specific_function %.loc4_31.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc4_31: init %i32 = call %.loc4_31.4(%.loc4_31.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc4_31.5: %i32 = value_of_initializer %int.convert_checked.loc4_31 [template = constants.%.33]
// CHECK:STDOUT:   %.loc4_31.6: %i32 = converted %.loc4_31.1, %.loc4_31.5 [template = constants.%.33]
// CHECK:STDOUT:   %int.right_shift: init %i32 = call %RightShift.ref(%.loc4_27.6, %.loc4_31.6) [template = constants.%.34]
// CHECK:STDOUT:   %.loc4_11.2: type = value_of_initializer %int.make_type_signed.loc4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_11.3: type = converted %int.make_type_signed.loc4, %.loc4_11.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_26.1: %Convert.type.6 = interface_witness_access constants.%.35, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc4_26.2: <bound method> = bound_method %int.right_shift, %.loc4_26.1 [template = constants.%.36]
// CHECK:STDOUT:   %.loc4_26.3: <specific function> = specific_function %.loc4_26.2, @Convert.4(constants.%.1) [template = constants.%.37]
// CHECK:STDOUT:   %.loc4_26.4: %i32 = value_of_initializer %int.right_shift [template = constants.%.34]
// CHECK:STDOUT:   %.loc4_26.5: %i32 = converted %int.right_shift, %.loc4_26.4 [template = constants.%.34]
// CHECK:STDOUT:   %int.convert_checked.loc4_26: init Core.IntLiteral = call %.loc4_26.3(%.loc4_26.5) [template = constants.%.38]
// CHECK:STDOUT:   %.loc4_26.6: Core.IntLiteral = value_of_initializer %int.convert_checked.loc4_26 [template = constants.%.38]
// CHECK:STDOUT:   %.loc4_26.7: Core.IntLiteral = converted %int.right_shift, %.loc4_26.6 [template = constants.%.38]
// CHECK:STDOUT:   %.loc4_33: type = array_type %.loc4_26.7, %i32 [template = constants.%.39]
// CHECK:STDOUT:   %arr.var: ref %.39 = var arr
// CHECK:STDOUT:   %arr: ref %.39 = bind_name arr, %arr.var
// CHECK:STDOUT:   %.loc5_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc5: init type = call constants.%Int(%.loc5_13.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_18: Core.IntLiteral = int_value 5 [template = constants.%.38]
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
// CHECK:STDOUT: fn @RightShift(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.right_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %RightShift.ref: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.right_shift: init %i32 = call %RightShift.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc8_26.1: %i32 = value_of_initializer %int.right_shift
// CHECK:STDOUT:   %.loc8_26.2: %i32 = converted %int.right_shift, %.loc8_26.1
// CHECK:STDOUT:   return %.loc8_26.2
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
// CHECK:STDOUT: --- arith_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %RightShift.type: type = fn_type @RightShift [template]
// CHECK:STDOUT:   %RightShift: %RightShift.type = struct_value () [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.28: <specific function> = specific_function %.27, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.29: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.30: %i32 = int_value -1 [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.4, @impl.3(%.1) [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.31: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.32: <bound method> = bound_method %.29, %Convert.15 [template]
// CHECK:STDOUT:   %.33: <specific function> = specific_function %.32, @Convert.4(%.1) [template]
// CHECK:STDOUT:   %.34: type = array_type %.2, %i32 [template]
// CHECK:STDOUT:   %.35: type = ptr_type %.34 [template]
// CHECK:STDOUT:   %.36: Core.IntLiteral = int_value 10 [template]
// CHECK:STDOUT:   %.37: <bound method> = bound_method %.36, %Convert.14 [template]
// CHECK:STDOUT:   %.38: <specific function> = specific_function %.37, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.39: %i32 = int_value 10 [template]
// CHECK:STDOUT:   %.40: %i32 = int_value -10 [template]
// CHECK:STDOUT:   %.41: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.42: <bound method> = bound_method %.41, %Convert.14 [template]
// CHECK:STDOUT:   %.43: <specific function> = specific_function %.42, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.44: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %.45: %i32 = int_value -3 [template]
// CHECK:STDOUT:   %.46: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %.47: <bound method> = bound_method %.46, %Convert.15 [template]
// CHECK:STDOUT:   %.48: <specific function> = specific_function %.47, @Convert.4(%.1) [template]
// CHECK:STDOUT:   %.49: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.50: type = array_type %.49, %i32 [template]
// CHECK:STDOUT:   %.51: type = ptr_type %.50 [template]
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
// CHECK:STDOUT:     .RightShift = %RightShift.decl
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .arr1 = %arr1
// CHECK:STDOUT:     .arr1_p = @__global_init.%arr1_p
// CHECK:STDOUT:     .arr2 = %arr2
// CHECK:STDOUT:     .arr2_p = @__global_init.%arr2_p
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %RightShift.decl: %RightShift.type = fn_decl @RightShift [template = constants.%RightShift] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_18.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc6_18: init type = call constants.%Int(%.loc6_18.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_18.2: type = value_of_initializer %int.make_type_signed.loc6_18 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_18.3: type = converted %int.make_type_signed.loc6_18, %.loc6_18.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_26.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc6_26: init type = call constants.%Int(%.loc6_26.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_26.2: type = value_of_initializer %int.make_type_signed.loc6_26 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_26.3: type = converted %int.make_type_signed.loc6_26, %.loc6_26.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_34.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc6_34: init type = call constants.%Int(%.loc6_34.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_34.2: type = value_of_initializer %int.make_type_signed.loc6_34 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_34.3: type = converted %int.make_type_signed.loc6_34, %.loc6_34.2 [template = constants.%i32]
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
// CHECK:STDOUT:     %.loc7_14.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc7_14: init type = call constants.%Int(%.loc7_14.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_14.2: type = value_of_initializer %int.make_type_signed.loc7_14 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_14.3: type = converted %int.make_type_signed.loc7_14, %.loc7_14.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_22.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc7_22: init type = call constants.%Int(%.loc7_22.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_22.2: type = value_of_initializer %int.make_type_signed.loc7_22 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_22.3: type = converted %int.make_type_signed.loc7_22, %.loc7_22.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_12.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc10: init type = call constants.%Int(%.loc10_12.1) [template = constants.%i32]
// CHECK:STDOUT:   %Negate.ref.loc10_17: %Negate.type = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %RightShift.ref.loc10: %RightShift.type = name_ref RightShift, %RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %Negate.ref.loc10_35: %Negate.type = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc10_42.1: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_42.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc10_42.3: <bound method> = bound_method %.loc10_42.1, %.loc10_42.2 [template = constants.%.27]
// CHECK:STDOUT:   %.loc10_42.4: <specific function> = specific_function %.loc10_42.3, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc10_42: init %i32 = call %.loc10_42.4(%.loc10_42.1) [template = constants.%.29]
// CHECK:STDOUT:   %.loc10_42.5: %i32 = value_of_initializer %int.convert_checked.loc10_42 [template = constants.%.29]
// CHECK:STDOUT:   %.loc10_42.6: %i32 = converted %.loc10_42.1, %.loc10_42.5 [template = constants.%.29]
// CHECK:STDOUT:   %int.snegate.loc10_41: init %i32 = call %Negate.ref.loc10_35(%.loc10_42.6) [template = constants.%.30]
// CHECK:STDOUT:   %.loc10_46.1: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_41.1: %i32 = value_of_initializer %int.snegate.loc10_41 [template = constants.%.30]
// CHECK:STDOUT:   %.loc10_41.2: %i32 = converted %int.snegate.loc10_41, %.loc10_41.1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc10_46.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc10_46.3: <bound method> = bound_method %.loc10_46.1, %.loc10_46.2 [template = constants.%.27]
// CHECK:STDOUT:   %.loc10_46.4: <specific function> = specific_function %.loc10_46.3, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc10_46: init %i32 = call %.loc10_46.4(%.loc10_46.1) [template = constants.%.29]
// CHECK:STDOUT:   %.loc10_46.5: %i32 = value_of_initializer %int.convert_checked.loc10_46 [template = constants.%.29]
// CHECK:STDOUT:   %.loc10_46.6: %i32 = converted %.loc10_46.1, %.loc10_46.5 [template = constants.%.29]
// CHECK:STDOUT:   %int.right_shift.loc10: init %i32 = call %RightShift.ref.loc10(%.loc10_41.2, %.loc10_46.6) [template = constants.%.30]
// CHECK:STDOUT:   %.loc10_34.1: %i32 = value_of_initializer %int.right_shift.loc10 [template = constants.%.30]
// CHECK:STDOUT:   %.loc10_34.2: %i32 = converted %int.right_shift.loc10, %.loc10_34.1 [template = constants.%.30]
// CHECK:STDOUT:   %int.snegate.loc10_23: init %i32 = call %Negate.ref.loc10_17(%.loc10_34.2) [template = constants.%.29]
// CHECK:STDOUT:   %.loc10_12.2: type = value_of_initializer %int.make_type_signed.loc10 [template = constants.%i32]
// CHECK:STDOUT:   %.loc10_12.3: type = converted %int.make_type_signed.loc10, %.loc10_12.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc10_23.1: %Convert.type.6 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc10_23.2: <bound method> = bound_method %int.snegate.loc10_23, %.loc10_23.1 [template = constants.%.32]
// CHECK:STDOUT:   %.loc10_23.3: <specific function> = specific_function %.loc10_23.2, @Convert.4(constants.%.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc10_23.4: %i32 = value_of_initializer %int.snegate.loc10_23 [template = constants.%.29]
// CHECK:STDOUT:   %.loc10_23.5: %i32 = converted %int.snegate.loc10_23, %.loc10_23.4 [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc10_23: init Core.IntLiteral = call %.loc10_23.3(%.loc10_23.5) [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_23.6: Core.IntLiteral = value_of_initializer %int.convert_checked.loc10_23 [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_23.7: Core.IntLiteral = converted %int.snegate.loc10_23, %.loc10_23.6 [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_49: type = array_type %.loc10_23.7, %i32 [template = constants.%.34]
// CHECK:STDOUT:   %arr1.var: ref %.34 = var arr1
// CHECK:STDOUT:   %arr1: ref %.34 = bind_name arr1, %arr1.var
// CHECK:STDOUT:   %.loc11_14.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc11: init type = call constants.%Int(%.loc11_14.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_19: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_14.2: type = value_of_initializer %int.make_type_signed.loc11 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_14.3: type = converted %int.make_type_signed.loc11, %.loc11_14.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_20: type = array_type %.loc11_19, %i32 [template = constants.%.34]
// CHECK:STDOUT:   %.loc11_21: type = ptr_type %.34 [template = constants.%.35]
// CHECK:STDOUT:   %.loc14_12.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc14: init type = call constants.%Int(%.loc14_12.1) [template = constants.%i32]
// CHECK:STDOUT:   %Negate.ref.loc14_17: %Negate.type = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %RightShift.ref.loc14: %RightShift.type = name_ref RightShift, %RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %Negate.ref.loc14_35: %Negate.type = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc14_42.1: Core.IntLiteral = int_value 10 [template = constants.%.36]
// CHECK:STDOUT:   %.loc14_42.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc14_42.3: <bound method> = bound_method %.loc14_42.1, %.loc14_42.2 [template = constants.%.37]
// CHECK:STDOUT:   %.loc14_42.4: <specific function> = specific_function %.loc14_42.3, @Convert.2(constants.%.1) [template = constants.%.38]
// CHECK:STDOUT:   %int.convert_checked.loc14_42: init %i32 = call %.loc14_42.4(%.loc14_42.1) [template = constants.%.39]
// CHECK:STDOUT:   %.loc14_42.5: %i32 = value_of_initializer %int.convert_checked.loc14_42 [template = constants.%.39]
// CHECK:STDOUT:   %.loc14_42.6: %i32 = converted %.loc14_42.1, %.loc14_42.5 [template = constants.%.39]
// CHECK:STDOUT:   %int.snegate.loc14_41: init %i32 = call %Negate.ref.loc14_35(%.loc14_42.6) [template = constants.%.40]
// CHECK:STDOUT:   %.loc14_47.1: Core.IntLiteral = int_value 2 [template = constants.%.41]
// CHECK:STDOUT:   %.loc14_41.1: %i32 = value_of_initializer %int.snegate.loc14_41 [template = constants.%.40]
// CHECK:STDOUT:   %.loc14_41.2: %i32 = converted %int.snegate.loc14_41, %.loc14_41.1 [template = constants.%.40]
// CHECK:STDOUT:   %.loc14_47.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc14_47.3: <bound method> = bound_method %.loc14_47.1, %.loc14_47.2 [template = constants.%.42]
// CHECK:STDOUT:   %.loc14_47.4: <specific function> = specific_function %.loc14_47.3, @Convert.2(constants.%.1) [template = constants.%.43]
// CHECK:STDOUT:   %int.convert_checked.loc14_47: init %i32 = call %.loc14_47.4(%.loc14_47.1) [template = constants.%.44]
// CHECK:STDOUT:   %.loc14_47.5: %i32 = value_of_initializer %int.convert_checked.loc14_47 [template = constants.%.44]
// CHECK:STDOUT:   %.loc14_47.6: %i32 = converted %.loc14_47.1, %.loc14_47.5 [template = constants.%.44]
// CHECK:STDOUT:   %int.right_shift.loc14: init %i32 = call %RightShift.ref.loc14(%.loc14_41.2, %.loc14_47.6) [template = constants.%.45]
// CHECK:STDOUT:   %.loc14_34.1: %i32 = value_of_initializer %int.right_shift.loc14 [template = constants.%.45]
// CHECK:STDOUT:   %.loc14_34.2: %i32 = converted %int.right_shift.loc14, %.loc14_34.1 [template = constants.%.45]
// CHECK:STDOUT:   %int.snegate.loc14_23: init %i32 = call %Negate.ref.loc14_17(%.loc14_34.2) [template = constants.%.46]
// CHECK:STDOUT:   %.loc14_12.2: type = value_of_initializer %int.make_type_signed.loc14 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_12.3: type = converted %int.make_type_signed.loc14, %.loc14_12.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_23.1: %Convert.type.6 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc14_23.2: <bound method> = bound_method %int.snegate.loc14_23, %.loc14_23.1 [template = constants.%.47]
// CHECK:STDOUT:   %.loc14_23.3: <specific function> = specific_function %.loc14_23.2, @Convert.4(constants.%.1) [template = constants.%.48]
// CHECK:STDOUT:   %.loc14_23.4: %i32 = value_of_initializer %int.snegate.loc14_23 [template = constants.%.46]
// CHECK:STDOUT:   %.loc14_23.5: %i32 = converted %int.snegate.loc14_23, %.loc14_23.4 [template = constants.%.46]
// CHECK:STDOUT:   %int.convert_checked.loc14_23: init Core.IntLiteral = call %.loc14_23.3(%.loc14_23.5) [template = constants.%.49]
// CHECK:STDOUT:   %.loc14_23.6: Core.IntLiteral = value_of_initializer %int.convert_checked.loc14_23 [template = constants.%.49]
// CHECK:STDOUT:   %.loc14_23.7: Core.IntLiteral = converted %int.snegate.loc14_23, %.loc14_23.6 [template = constants.%.49]
// CHECK:STDOUT:   %.loc14_50: type = array_type %.loc14_23.7, %i32 [template = constants.%.50]
// CHECK:STDOUT:   %arr2.var: ref %.50 = var arr2
// CHECK:STDOUT:   %arr2: ref %.50 = bind_name arr2, %arr2.var
// CHECK:STDOUT:   %.loc15_14.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc15: init type = call constants.%Int(%.loc15_14.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_19: Core.IntLiteral = int_value 3 [template = constants.%.49]
// CHECK:STDOUT:   %.loc15_14.2: type = value_of_initializer %int.make_type_signed.loc15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_14.3: type = converted %int.make_type_signed.loc15, %.loc15_14.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_20: type = array_type %.loc15_19, %i32 [template = constants.%.50]
// CHECK:STDOUT:   %.loc15_21: type = ptr_type %.50 [template = constants.%.51]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RightShift(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.right_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a.param_patt: %i32) -> %i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr1.ref: ref %.34 = name_ref arr1, file.%arr1
// CHECK:STDOUT:   %.loc11: %.35 = addr_of %arr1.ref
// CHECK:STDOUT:   %arr1_p: %.35 = bind_name arr1_p, %.loc11
// CHECK:STDOUT:   %arr2.ref: ref %.50 = name_ref arr2, file.%arr2
// CHECK:STDOUT:   %.loc15: %.51 = addr_of %arr2.ref
// CHECK:STDOUT:   %arr2_p: %.51 = bind_name arr2_p, %.loc15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %RightShift.type: type = fn_type @RightShift [template]
// CHECK:STDOUT:   %RightShift: %RightShift.type = struct_value () [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 31 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.28: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.29: <specific function> = specific_function %.28, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.30: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.3, %Convert.14 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.33: %i32 = int_value 31 [template]
// CHECK:STDOUT:   %.34: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %.35: <bound method> = bound_method %.1, %Convert.14 [template]
// CHECK:STDOUT:   %.36: <specific function> = specific_function %.35, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.37: %i32 = int_value 32 [template]
// CHECK:STDOUT:   %.38: Core.IntLiteral = int_value 33 [template]
// CHECK:STDOUT:   %.39: <bound method> = bound_method %.38, %Convert.14 [template]
// CHECK:STDOUT:   %.40: <specific function> = specific_function %.39, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.41: %i32 = int_value 33 [template]
// CHECK:STDOUT:   %.42: %i32 = int_value -1 [template]
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
// CHECK:STDOUT:     .RightShift = %RightShift.decl
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .size_1 = @__global_init.%size_1
// CHECK:STDOUT:     .size_2 = @__global_init.%size_2
// CHECK:STDOUT:     .size_3 = @__global_init.%size_3
// CHECK:STDOUT:     .negative = @__global_init.%negative
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %RightShift.decl: %RightShift.type = fn_decl @RightShift [template = constants.%RightShift] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_18.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_18: init type = call constants.%Int(%.loc4_18.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_18.2: type = value_of_initializer %int.make_type_signed.loc4_18 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_18.3: type = converted %int.make_type_signed.loc4_18, %.loc4_18.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_26.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_26: init type = call constants.%Int(%.loc4_26.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_26.2: type = value_of_initializer %int.make_type_signed.loc4_26 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_26.3: type = converted %int.make_type_signed.loc4_26, %.loc4_26.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_34.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_34: init type = call constants.%Int(%.loc4_34.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_34.2: type = value_of_initializer %int.make_type_signed.loc4_34 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_34.3: type = converted %int.make_type_signed.loc4_34, %.loc4_34.2 [template = constants.%i32]
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
// CHECK:STDOUT:     %.loc5_14.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc5_14: init type = call constants.%Int(%.loc5_14.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_14.2: type = value_of_initializer %int.make_type_signed.loc5_14 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_14.3: type = converted %int.make_type_signed.loc5_14, %.loc5_14.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_22.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc5_22: init type = call constants.%Int(%.loc5_22.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_22.2: type = value_of_initializer %int.make_type_signed.loc5_22 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_22.3: type = converted %int.make_type_signed.loc5_22, %.loc5_22.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc8: init type = call constants.%Int(%.loc8_13.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_13.2: type = value_of_initializer %int.make_type_signed.loc8 [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_13.3: type = converted %int.make_type_signed.loc8, %.loc8_13.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc13: init type = call constants.%Int(%.loc13_13.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_13.2: type = value_of_initializer %int.make_type_signed.loc13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_13.3: type = converted %int.make_type_signed.loc13, %.loc13_13.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc18: init type = call constants.%Int(%.loc18_13.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_13.2: type = value_of_initializer %int.make_type_signed.loc18 [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_13.3: type = converted %int.make_type_signed.loc18, %.loc18_13.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc24_15.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc24: init type = call constants.%Int(%.loc24_15.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc24_15.2: type = value_of_initializer %int.make_type_signed.loc24 [template = constants.%i32]
// CHECK:STDOUT:   %.loc24_15.3: type = converted %int.make_type_signed.loc24, %.loc24_15.2 [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RightShift(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.right_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a.param_patt: %i32) -> %i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %RightShift.ref.loc8: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %.loc8_30.1: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_33.1: Core.IntLiteral = int_value 31 [template = constants.%.3]
// CHECK:STDOUT:   %.loc8_30.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc8_30.3: <bound method> = bound_method %.loc8_30.1, %.loc8_30.2 [template = constants.%.28]
// CHECK:STDOUT:   %.loc8_30.4: <specific function> = specific_function %.loc8_30.3, @Convert.2(constants.%.1) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc8_30: init %i32 = call %.loc8_30.4(%.loc8_30.1) [template = constants.%.30]
// CHECK:STDOUT:   %.loc8_30.5: %i32 = value_of_initializer %int.convert_checked.loc8_30 [template = constants.%.30]
// CHECK:STDOUT:   %.loc8_30.6: %i32 = converted %.loc8_30.1, %.loc8_30.5 [template = constants.%.30]
// CHECK:STDOUT:   %.loc8_33.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc8_33.3: <bound method> = bound_method %.loc8_33.1, %.loc8_33.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc8_33.4: <specific function> = specific_function %.loc8_33.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc8_33: init %i32 = call %.loc8_33.4(%.loc8_33.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc8_33.5: %i32 = value_of_initializer %int.convert_checked.loc8_33 [template = constants.%.33]
// CHECK:STDOUT:   %.loc8_33.6: %i32 = converted %.loc8_33.1, %.loc8_33.5 [template = constants.%.33]
// CHECK:STDOUT:   %int.right_shift.loc8: init %i32 = call %RightShift.ref.loc8(%.loc8_30.6, %.loc8_33.6) [template = constants.%.34]
// CHECK:STDOUT:   %.loc8_36.1: %i32 = value_of_initializer %int.right_shift.loc8 [template = constants.%.34]
// CHECK:STDOUT:   %.loc8_36.2: %i32 = converted %int.right_shift.loc8, %.loc8_36.1 [template = constants.%.34]
// CHECK:STDOUT:   %size_1: %i32 = bind_name size_1, %.loc8_36.2
// CHECK:STDOUT:   %RightShift.ref.loc13: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %.loc13_30.1: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc13_33.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %.loc13_30.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc13_30.3: <bound method> = bound_method %.loc13_30.1, %.loc13_30.2 [template = constants.%.28]
// CHECK:STDOUT:   %.loc13_30.4: <specific function> = specific_function %.loc13_30.3, @Convert.2(constants.%.1) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc13_30: init %i32 = call %.loc13_30.4(%.loc13_30.1) [template = constants.%.30]
// CHECK:STDOUT:   %.loc13_30.5: %i32 = value_of_initializer %int.convert_checked.loc13_30 [template = constants.%.30]
// CHECK:STDOUT:   %.loc13_30.6: %i32 = converted %.loc13_30.1, %.loc13_30.5 [template = constants.%.30]
// CHECK:STDOUT:   %.loc13_33.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc13_33.3: <bound method> = bound_method %.loc13_33.1, %.loc13_33.2 [template = constants.%.35]
// CHECK:STDOUT:   %.loc13_33.4: <specific function> = specific_function %.loc13_33.3, @Convert.2(constants.%.1) [template = constants.%.36]
// CHECK:STDOUT:   %int.convert_checked.loc13_33: init %i32 = call %.loc13_33.4(%.loc13_33.1) [template = constants.%.37]
// CHECK:STDOUT:   %.loc13_33.5: %i32 = value_of_initializer %int.convert_checked.loc13_33 [template = constants.%.37]
// CHECK:STDOUT:   %.loc13_33.6: %i32 = converted %.loc13_33.1, %.loc13_33.5 [template = constants.%.37]
// CHECK:STDOUT:   %int.right_shift.loc13: init %i32 = call %RightShift.ref.loc13(%.loc13_30.6, %.loc13_33.6) [template = <error>]
// CHECK:STDOUT:   %.loc13_36.1: %i32 = value_of_initializer %int.right_shift.loc13 [template = <error>]
// CHECK:STDOUT:   %.loc13_36.2: %i32 = converted %int.right_shift.loc13, %.loc13_36.1 [template = <error>]
// CHECK:STDOUT:   %size_2: %i32 = bind_name size_2, %.loc13_36.2
// CHECK:STDOUT:   %RightShift.ref.loc18: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %.loc18_30.1: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc18_33.1: Core.IntLiteral = int_value 33 [template = constants.%.38]
// CHECK:STDOUT:   %.loc18_30.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc18_30.3: <bound method> = bound_method %.loc18_30.1, %.loc18_30.2 [template = constants.%.28]
// CHECK:STDOUT:   %.loc18_30.4: <specific function> = specific_function %.loc18_30.3, @Convert.2(constants.%.1) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc18_30: init %i32 = call %.loc18_30.4(%.loc18_30.1) [template = constants.%.30]
// CHECK:STDOUT:   %.loc18_30.5: %i32 = value_of_initializer %int.convert_checked.loc18_30 [template = constants.%.30]
// CHECK:STDOUT:   %.loc18_30.6: %i32 = converted %.loc18_30.1, %.loc18_30.5 [template = constants.%.30]
// CHECK:STDOUT:   %.loc18_33.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc18_33.3: <bound method> = bound_method %.loc18_33.1, %.loc18_33.2 [template = constants.%.39]
// CHECK:STDOUT:   %.loc18_33.4: <specific function> = specific_function %.loc18_33.3, @Convert.2(constants.%.1) [template = constants.%.40]
// CHECK:STDOUT:   %int.convert_checked.loc18_33: init %i32 = call %.loc18_33.4(%.loc18_33.1) [template = constants.%.41]
// CHECK:STDOUT:   %.loc18_33.5: %i32 = value_of_initializer %int.convert_checked.loc18_33 [template = constants.%.41]
// CHECK:STDOUT:   %.loc18_33.6: %i32 = converted %.loc18_33.1, %.loc18_33.5 [template = constants.%.41]
// CHECK:STDOUT:   %int.right_shift.loc18: init %i32 = call %RightShift.ref.loc18(%.loc18_30.6, %.loc18_33.6) [template = <error>]
// CHECK:STDOUT:   %.loc18_36.1: %i32 = value_of_initializer %int.right_shift.loc18 [template = <error>]
// CHECK:STDOUT:   %.loc18_36.2: %i32 = converted %int.right_shift.loc18, %.loc18_36.1 [template = <error>]
// CHECK:STDOUT:   %size_3: %i32 = bind_name size_3, %.loc18_36.2
// CHECK:STDOUT:   %RightShift.ref.loc24: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %.loc24_32.1: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %Negate.ref: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc24_42.1: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc24_42.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc24_42.3: <bound method> = bound_method %.loc24_42.1, %.loc24_42.2 [template = constants.%.28]
// CHECK:STDOUT:   %.loc24_42.4: <specific function> = specific_function %.loc24_42.3, @Convert.2(constants.%.1) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc24_42: init %i32 = call %.loc24_42.4(%.loc24_42.1) [template = constants.%.30]
// CHECK:STDOUT:   %.loc24_42.5: %i32 = value_of_initializer %int.convert_checked.loc24_42 [template = constants.%.30]
// CHECK:STDOUT:   %.loc24_42.6: %i32 = converted %.loc24_42.1, %.loc24_42.5 [template = constants.%.30]
// CHECK:STDOUT:   %int.snegate: init %i32 = call %Negate.ref(%.loc24_42.6) [template = constants.%.42]
// CHECK:STDOUT:   %.loc24_32.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc24_32.3: <bound method> = bound_method %.loc24_32.1, %.loc24_32.2 [template = constants.%.28]
// CHECK:STDOUT:   %.loc24_32.4: <specific function> = specific_function %.loc24_32.3, @Convert.2(constants.%.1) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc24_32: init %i32 = call %.loc24_32.4(%.loc24_32.1) [template = constants.%.30]
// CHECK:STDOUT:   %.loc24_32.5: %i32 = value_of_initializer %int.convert_checked.loc24_32 [template = constants.%.30]
// CHECK:STDOUT:   %.loc24_32.6: %i32 = converted %.loc24_32.1, %.loc24_32.5 [template = constants.%.30]
// CHECK:STDOUT:   %.loc24_41.1: %i32 = value_of_initializer %int.snegate [template = constants.%.42]
// CHECK:STDOUT:   %.loc24_41.2: %i32 = converted %int.snegate, %.loc24_41.1 [template = constants.%.42]
// CHECK:STDOUT:   %int.right_shift.loc24: init %i32 = call %RightShift.ref.loc24(%.loc24_32.6, %.loc24_41.2) [template = <error>]
// CHECK:STDOUT:   %.loc24_45.1: %i32 = value_of_initializer %int.right_shift.loc24 [template = <error>]
// CHECK:STDOUT:   %.loc24_45.2: %i32 = converted %int.right_shift.loc24, %.loc24_45.1 [template = <error>]
// CHECK:STDOUT:   %negative: %i32 = bind_name negative, %.loc24_45.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
