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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %RightShift.type: type = fn_type @RightShift [template]
// CHECK:STDOUT:   %RightShift: %RightShift.type = struct_value () [template]
// CHECK:STDOUT:   %.1: i32 = int_value 22 [template]
// CHECK:STDOUT:   %.2: i32 = int_value 2 [template]
// CHECK:STDOUT:   %.3: i32 = int_value 5 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.28: <bound method> = bound_method %.3, %Convert.15 [template]
// CHECK:STDOUT:   %.29: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %.30: type = array_type %.29, i32 [template]
// CHECK:STDOUT:   %.31: type = ptr_type %.30 [template]
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
// CHECK:STDOUT:     .RightShift = %RightShift.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .arr_p = @__global_init.%arr_p
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %RightShift.decl: %RightShift.type = fn_decl @RightShift [template = constants.%RightShift] {
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc2_18: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc2_18.1: type = value_of_initializer %int.make_type_32.loc2_18 [template = i32]
// CHECK:STDOUT:     %.loc2_18.2: type = converted %int.make_type_32.loc2_18, %.loc2_18.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc2_26: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc2_26.1: type = value_of_initializer %int.make_type_32.loc2_26 [template = i32]
// CHECK:STDOUT:     %.loc2_26.2: type = converted %int.make_type_32.loc2_26, %.loc2_26.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc2_34: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc2_34.1: type = value_of_initializer %int.make_type_32.loc2_34 [template = i32]
// CHECK:STDOUT:     %.loc2_34.2: type = converted %int.make_type_32.loc2_34, %.loc2_34.1 [template = i32]
// CHECK:STDOUT:     %a.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc4: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %RightShift.ref: %RightShift.type = name_ref RightShift, %RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %.loc4_27: i32 = int_value 22 [template = constants.%.1]
// CHECK:STDOUT:   %.loc4_31: i32 = int_value 2 [template = constants.%.2]
// CHECK:STDOUT:   %int.right_shift: init i32 = call %RightShift.ref(%.loc4_27, %.loc4_31) [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_11.1: type = value_of_initializer %int.make_type_32.loc4 [template = i32]
// CHECK:STDOUT:   %.loc4_11.2: type = converted %int.make_type_32.loc4, %.loc4_11.1 [template = i32]
// CHECK:STDOUT:   %.loc4_26.1: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc4_26.2: <bound method> = bound_method %int.right_shift, %.loc4_26.1 [template = constants.%.28]
// CHECK:STDOUT:   %.loc4_26.3: i32 = value_of_initializer %int.right_shift [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_26.4: i32 = converted %int.right_shift, %.loc4_26.3 [template = constants.%.3]
// CHECK:STDOUT:   %int.convert_checked.loc4: init Core.IntLiteral = call %.loc4_26.2(%.loc4_26.4) [template = constants.%.29]
// CHECK:STDOUT:   %.loc4_26.5: Core.IntLiteral = value_of_initializer %int.convert_checked.loc4 [template = constants.%.29]
// CHECK:STDOUT:   %.loc4_26.6: Core.IntLiteral = converted %int.right_shift, %.loc4_26.5 [template = constants.%.29]
// CHECK:STDOUT:   %.loc4_33: type = array_type %.loc4_26.6, i32 [template = constants.%.30]
// CHECK:STDOUT:   %arr.var: ref %.30 = var arr
// CHECK:STDOUT:   %arr: ref %.30 = bind_name arr, %arr.var
// CHECK:STDOUT:   %int.make_type_32.loc5: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_18.1: i32 = int_value 5 [template = constants.%.3]
// CHECK:STDOUT:   %.loc5_13.1: type = value_of_initializer %int.make_type_32.loc5 [template = i32]
// CHECK:STDOUT:   %.loc5_13.2: type = converted %int.make_type_32.loc5, %.loc5_13.1 [template = i32]
// CHECK:STDOUT:   %.loc5_18.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc5_18.3: <bound method> = bound_method %.loc5_18.1, %.loc5_18.2 [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc5: init Core.IntLiteral = call %.loc5_18.3(%.loc5_18.1) [template = constants.%.29]
// CHECK:STDOUT:   %.loc5_18.4: Core.IntLiteral = value_of_initializer %int.convert_checked.loc5 [template = constants.%.29]
// CHECK:STDOUT:   %.loc5_18.5: Core.IntLiteral = converted %.loc5_18.1, %.loc5_18.4 [template = constants.%.29]
// CHECK:STDOUT:   %.loc5_19: type = array_type %.loc5_18.5, i32 [template = constants.%.30]
// CHECK:STDOUT:   %.loc5_20: type = ptr_type %.30 [template = constants.%.31]
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
// CHECK:STDOUT: fn @RightShift(%a.param_patt: i32, %b.param_patt: i32) -> i32 = "int.right_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: i32, %b.param_patt: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %RightShift.ref: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %int.right_shift: init i32 = call %RightShift.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc8_26.1: i32 = value_of_initializer %int.right_shift
// CHECK:STDOUT:   %.loc8_26.2: i32 = converted %int.right_shift, %.loc8_26.1
// CHECK:STDOUT:   return %.loc8_26.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr.ref: ref %.30 = name_ref arr, file.%arr
// CHECK:STDOUT:   %.loc5: %.31 = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: %.31 = bind_name arr_p, %.loc5
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- arith_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %RightShift.type: type = fn_type @RightShift [template]
// CHECK:STDOUT:   %RightShift: %RightShift.type = struct_value () [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %.1: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.2: i32 = int_value -1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.28: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.29: type = array_type %.28, i32 [template]
// CHECK:STDOUT:   %.30: type = ptr_type %.29 [template]
// CHECK:STDOUT:   %.31: i32 = int_value 10 [template]
// CHECK:STDOUT:   %.32: i32 = int_value -10 [template]
// CHECK:STDOUT:   %.33: i32 = int_value 2 [template]
// CHECK:STDOUT:   %.34: i32 = int_value -3 [template]
// CHECK:STDOUT:   %.35: i32 = int_value 3 [template]
// CHECK:STDOUT:   %.36: <bound method> = bound_method %.35, %Convert.15 [template]
// CHECK:STDOUT:   %.37: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.38: type = array_type %.37, i32 [template]
// CHECK:STDOUT:   %.39: type = ptr_type %.38 [template]
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
// CHECK:STDOUT:     .RightShift = %RightShift.decl
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .arr1 = %arr1
// CHECK:STDOUT:     .arr1_p = @__global_init.%arr1_p
// CHECK:STDOUT:     .arr2 = %arr2
// CHECK:STDOUT:     .arr2_p = @__global_init.%arr2_p
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %RightShift.decl: %RightShift.type = fn_decl @RightShift [template = constants.%RightShift] {
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc6_18: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_18.1: type = value_of_initializer %int.make_type_32.loc6_18 [template = i32]
// CHECK:STDOUT:     %.loc6_18.2: type = converted %int.make_type_32.loc6_18, %.loc6_18.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc6_26: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_26.1: type = value_of_initializer %int.make_type_32.loc6_26 [template = i32]
// CHECK:STDOUT:     %.loc6_26.2: type = converted %int.make_type_32.loc6_26, %.loc6_26.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc6_34: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_34.1: type = value_of_initializer %int.make_type_32.loc6_34 [template = i32]
// CHECK:STDOUT:     %.loc6_34.2: type = converted %int.make_type_32.loc6_34, %.loc6_34.1 [template = i32]
// CHECK:STDOUT:     %a.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate.decl: %Negate.type = fn_decl @Negate [template = constants.%Negate] {
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc7_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc7_14.1: type = value_of_initializer %int.make_type_32.loc7_14 [template = i32]
// CHECK:STDOUT:     %.loc7_14.2: type = converted %int.make_type_32.loc7_14, %.loc7_14.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc7_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc7_22.1: type = value_of_initializer %int.make_type_32.loc7_22 [template = i32]
// CHECK:STDOUT:     %.loc7_22.2: type = converted %int.make_type_32.loc7_22, %.loc7_22.1 [template = i32]
// CHECK:STDOUT:     %a.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc10: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %Negate.ref.loc10_17: %Negate.type = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %RightShift.ref.loc10: %RightShift.type = name_ref RightShift, %RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %Negate.ref.loc10_35: %Negate.type = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc10_42: i32 = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %int.snegate.loc10_41: init i32 = call %Negate.ref.loc10_35(%.loc10_42) [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_46: i32 = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_41.1: i32 = value_of_initializer %int.snegate.loc10_41 [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_41.2: i32 = converted %int.snegate.loc10_41, %.loc10_41.1 [template = constants.%.2]
// CHECK:STDOUT:   %int.right_shift.loc10: init i32 = call %RightShift.ref.loc10(%.loc10_41.2, %.loc10_46) [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_34.1: i32 = value_of_initializer %int.right_shift.loc10 [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_34.2: i32 = converted %int.right_shift.loc10, %.loc10_34.1 [template = constants.%.2]
// CHECK:STDOUT:   %int.snegate.loc10_23: init i32 = call %Negate.ref.loc10_17(%.loc10_34.2) [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_12.1: type = value_of_initializer %int.make_type_32.loc10 [template = i32]
// CHECK:STDOUT:   %.loc10_12.2: type = converted %int.make_type_32.loc10, %.loc10_12.1 [template = i32]
// CHECK:STDOUT:   %.loc10_23.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc10_23.2: <bound method> = bound_method %int.snegate.loc10_23, %.loc10_23.1 [template = constants.%.27]
// CHECK:STDOUT:   %.loc10_23.3: i32 = value_of_initializer %int.snegate.loc10_23 [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_23.4: i32 = converted %int.snegate.loc10_23, %.loc10_23.3 [template = constants.%.1]
// CHECK:STDOUT:   %int.convert_checked.loc10: init Core.IntLiteral = call %.loc10_23.2(%.loc10_23.4) [template = constants.%.28]
// CHECK:STDOUT:   %.loc10_23.5: Core.IntLiteral = value_of_initializer %int.convert_checked.loc10 [template = constants.%.28]
// CHECK:STDOUT:   %.loc10_23.6: Core.IntLiteral = converted %int.snegate.loc10_23, %.loc10_23.5 [template = constants.%.28]
// CHECK:STDOUT:   %.loc10_49: type = array_type %.loc10_23.6, i32 [template = constants.%.29]
// CHECK:STDOUT:   %arr1.var: ref %.29 = var arr1
// CHECK:STDOUT:   %arr1: ref %.29 = bind_name arr1, %arr1.var
// CHECK:STDOUT:   %int.make_type_32.loc11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_19.1: i32 = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc11_14.1: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:   %.loc11_14.2: type = converted %int.make_type_32.loc11, %.loc11_14.1 [template = i32]
// CHECK:STDOUT:   %.loc11_19.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_19.3: <bound method> = bound_method %.loc11_19.1, %.loc11_19.2 [template = constants.%.27]
// CHECK:STDOUT:   %int.convert_checked.loc11: init Core.IntLiteral = call %.loc11_19.3(%.loc11_19.1) [template = constants.%.28]
// CHECK:STDOUT:   %.loc11_19.4: Core.IntLiteral = value_of_initializer %int.convert_checked.loc11 [template = constants.%.28]
// CHECK:STDOUT:   %.loc11_19.5: Core.IntLiteral = converted %.loc11_19.1, %.loc11_19.4 [template = constants.%.28]
// CHECK:STDOUT:   %.loc11_20: type = array_type %.loc11_19.5, i32 [template = constants.%.29]
// CHECK:STDOUT:   %.loc11_21: type = ptr_type %.29 [template = constants.%.30]
// CHECK:STDOUT:   %int.make_type_32.loc14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %Negate.ref.loc14_17: %Negate.type = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %RightShift.ref.loc14: %RightShift.type = name_ref RightShift, %RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %Negate.ref.loc14_35: %Negate.type = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc14_42: i32 = int_value 10 [template = constants.%.31]
// CHECK:STDOUT:   %int.snegate.loc14_41: init i32 = call %Negate.ref.loc14_35(%.loc14_42) [template = constants.%.32]
// CHECK:STDOUT:   %.loc14_47: i32 = int_value 2 [template = constants.%.33]
// CHECK:STDOUT:   %.loc14_41.1: i32 = value_of_initializer %int.snegate.loc14_41 [template = constants.%.32]
// CHECK:STDOUT:   %.loc14_41.2: i32 = converted %int.snegate.loc14_41, %.loc14_41.1 [template = constants.%.32]
// CHECK:STDOUT:   %int.right_shift.loc14: init i32 = call %RightShift.ref.loc14(%.loc14_41.2, %.loc14_47) [template = constants.%.34]
// CHECK:STDOUT:   %.loc14_34.1: i32 = value_of_initializer %int.right_shift.loc14 [template = constants.%.34]
// CHECK:STDOUT:   %.loc14_34.2: i32 = converted %int.right_shift.loc14, %.loc14_34.1 [template = constants.%.34]
// CHECK:STDOUT:   %int.snegate.loc14_23: init i32 = call %Negate.ref.loc14_17(%.loc14_34.2) [template = constants.%.35]
// CHECK:STDOUT:   %.loc14_12.1: type = value_of_initializer %int.make_type_32.loc14 [template = i32]
// CHECK:STDOUT:   %.loc14_12.2: type = converted %int.make_type_32.loc14, %.loc14_12.1 [template = i32]
// CHECK:STDOUT:   %.loc14_23.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc14_23.2: <bound method> = bound_method %int.snegate.loc14_23, %.loc14_23.1 [template = constants.%.36]
// CHECK:STDOUT:   %.loc14_23.3: i32 = value_of_initializer %int.snegate.loc14_23 [template = constants.%.35]
// CHECK:STDOUT:   %.loc14_23.4: i32 = converted %int.snegate.loc14_23, %.loc14_23.3 [template = constants.%.35]
// CHECK:STDOUT:   %int.convert_checked.loc14: init Core.IntLiteral = call %.loc14_23.2(%.loc14_23.4) [template = constants.%.37]
// CHECK:STDOUT:   %.loc14_23.5: Core.IntLiteral = value_of_initializer %int.convert_checked.loc14 [template = constants.%.37]
// CHECK:STDOUT:   %.loc14_23.6: Core.IntLiteral = converted %int.snegate.loc14_23, %.loc14_23.5 [template = constants.%.37]
// CHECK:STDOUT:   %.loc14_50: type = array_type %.loc14_23.6, i32 [template = constants.%.38]
// CHECK:STDOUT:   %arr2.var: ref %.38 = var arr2
// CHECK:STDOUT:   %arr2: ref %.38 = bind_name arr2, %arr2.var
// CHECK:STDOUT:   %int.make_type_32.loc15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc15_19.1: i32 = int_value 3 [template = constants.%.35]
// CHECK:STDOUT:   %.loc15_14.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:   %.loc15_14.2: type = converted %int.make_type_32.loc15, %.loc15_14.1 [template = i32]
// CHECK:STDOUT:   %.loc15_19.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc15_19.3: <bound method> = bound_method %.loc15_19.1, %.loc15_19.2 [template = constants.%.36]
// CHECK:STDOUT:   %int.convert_checked.loc15: init Core.IntLiteral = call %.loc15_19.3(%.loc15_19.1) [template = constants.%.37]
// CHECK:STDOUT:   %.loc15_19.4: Core.IntLiteral = value_of_initializer %int.convert_checked.loc15 [template = constants.%.37]
// CHECK:STDOUT:   %.loc15_19.5: Core.IntLiteral = converted %.loc15_19.1, %.loc15_19.4 [template = constants.%.37]
// CHECK:STDOUT:   %.loc15_20: type = array_type %.loc15_19.5, i32 [template = constants.%.38]
// CHECK:STDOUT:   %.loc15_21: type = ptr_type %.38 [template = constants.%.39]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RightShift(%a.param_patt: i32, %b.param_patt: i32) -> i32 = "int.right_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a.param_patt: i32) -> i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr1.ref: ref %.29 = name_ref arr1, file.%arr1
// CHECK:STDOUT:   %.loc11: %.30 = addr_of %arr1.ref
// CHECK:STDOUT:   %arr1_p: %.30 = bind_name arr1_p, %.loc11
// CHECK:STDOUT:   %arr2.ref: ref %.38 = name_ref arr2, file.%arr2
// CHECK:STDOUT:   %.loc15: %.39 = addr_of %arr2.ref
// CHECK:STDOUT:   %arr2_p: %.39 = bind_name arr2_p, %.loc15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %RightShift.type: type = fn_type @RightShift [template]
// CHECK:STDOUT:   %RightShift: %RightShift.type = struct_value () [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %.1: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.2: i32 = int_value 31 [template]
// CHECK:STDOUT:   %.3: i32 = int_value 0 [template]
// CHECK:STDOUT:   %.4: i32 = int_value 32 [template]
// CHECK:STDOUT:   %.5: i32 = int_value 33 [template]
// CHECK:STDOUT:   %.6: i32 = int_value -1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
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
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc4_18: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_18.1: type = value_of_initializer %int.make_type_32.loc4_18 [template = i32]
// CHECK:STDOUT:     %.loc4_18.2: type = converted %int.make_type_32.loc4_18, %.loc4_18.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc4_26: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_26.1: type = value_of_initializer %int.make_type_32.loc4_26 [template = i32]
// CHECK:STDOUT:     %.loc4_26.2: type = converted %int.make_type_32.loc4_26, %.loc4_26.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc4_34: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_34.1: type = value_of_initializer %int.make_type_32.loc4_34 [template = i32]
// CHECK:STDOUT:     %.loc4_34.2: type = converted %int.make_type_32.loc4_34, %.loc4_34.1 [template = i32]
// CHECK:STDOUT:     %a.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate.decl: %Negate.type = fn_decl @Negate [template = constants.%Negate] {
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc5_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_14.1: type = value_of_initializer %int.make_type_32.loc5_14 [template = i32]
// CHECK:STDOUT:     %.loc5_14.2: type = converted %int.make_type_32.loc5_14, %.loc5_14.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc5_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_22.1: type = value_of_initializer %int.make_type_32.loc5_22 [template = i32]
// CHECK:STDOUT:     %.loc5_22.2: type = converted %int.make_type_32.loc5_22, %.loc5_22.1 [template = i32]
// CHECK:STDOUT:     %a.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc8: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc8_13.1: type = value_of_initializer %int.make_type_32.loc8 [template = i32]
// CHECK:STDOUT:   %.loc8_13.2: type = converted %int.make_type_32.loc8, %.loc8_13.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_13.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:   %.loc13_13.2: type = converted %int.make_type_32.loc13, %.loc13_13.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc18: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc18_13.1: type = value_of_initializer %int.make_type_32.loc18 [template = i32]
// CHECK:STDOUT:   %.loc18_13.2: type = converted %int.make_type_32.loc18, %.loc18_13.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc24: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc24_15.1: type = value_of_initializer %int.make_type_32.loc24 [template = i32]
// CHECK:STDOUT:   %.loc24_15.2: type = converted %int.make_type_32.loc24, %.loc24_15.1 [template = i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RightShift(%a.param_patt: i32, %b.param_patt: i32) -> i32 = "int.right_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a.param_patt: i32) -> i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %RightShift.ref.loc8: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %.loc8_30: i32 = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc8_33: i32 = int_value 31 [template = constants.%.2]
// CHECK:STDOUT:   %int.right_shift.loc8: init i32 = call %RightShift.ref.loc8(%.loc8_30, %.loc8_33) [template = constants.%.3]
// CHECK:STDOUT:   %.loc8_36.1: i32 = value_of_initializer %int.right_shift.loc8 [template = constants.%.3]
// CHECK:STDOUT:   %.loc8_36.2: i32 = converted %int.right_shift.loc8, %.loc8_36.1 [template = constants.%.3]
// CHECK:STDOUT:   %size_1: i32 = bind_name size_1, %.loc8_36.2
// CHECK:STDOUT:   %RightShift.ref.loc13: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %.loc13_30: i32 = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc13_33: i32 = int_value 32 [template = constants.%.4]
// CHECK:STDOUT:   %int.right_shift.loc13: init i32 = call %RightShift.ref.loc13(%.loc13_30, %.loc13_33) [template = <error>]
// CHECK:STDOUT:   %.loc13_36.1: i32 = value_of_initializer %int.right_shift.loc13 [template = <error>]
// CHECK:STDOUT:   %.loc13_36.2: i32 = converted %int.right_shift.loc13, %.loc13_36.1 [template = <error>]
// CHECK:STDOUT:   %size_2: i32 = bind_name size_2, %.loc13_36.2
// CHECK:STDOUT:   %RightShift.ref.loc18: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %.loc18_30: i32 = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc18_33: i32 = int_value 33 [template = constants.%.5]
// CHECK:STDOUT:   %int.right_shift.loc18: init i32 = call %RightShift.ref.loc18(%.loc18_30, %.loc18_33) [template = <error>]
// CHECK:STDOUT:   %.loc18_36.1: i32 = value_of_initializer %int.right_shift.loc18 [template = <error>]
// CHECK:STDOUT:   %.loc18_36.2: i32 = converted %int.right_shift.loc18, %.loc18_36.1 [template = <error>]
// CHECK:STDOUT:   %size_3: i32 = bind_name size_3, %.loc18_36.2
// CHECK:STDOUT:   %RightShift.ref.loc24: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %.loc24_32: i32 = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %Negate.ref: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc24_42: i32 = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %int.snegate: init i32 = call %Negate.ref(%.loc24_42) [template = constants.%.6]
// CHECK:STDOUT:   %.loc24_41.1: i32 = value_of_initializer %int.snegate [template = constants.%.6]
// CHECK:STDOUT:   %.loc24_41.2: i32 = converted %int.snegate, %.loc24_41.1 [template = constants.%.6]
// CHECK:STDOUT:   %int.right_shift.loc24: init i32 = call %RightShift.ref.loc24(%.loc24_32, %.loc24_41.2) [template = <error>]
// CHECK:STDOUT:   %.loc24_45.1: i32 = value_of_initializer %int.right_shift.loc24 [template = <error>]
// CHECK:STDOUT:   %.loc24_45.2: i32 = converted %int.right_shift.loc24, %.loc24_45.1 [template = <error>]
// CHECK:STDOUT:   %negative: i32 = bind_name negative, %.loc24_45.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
