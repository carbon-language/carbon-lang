// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

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

package ArithShift api;

fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
fn Negate(a: i32) -> i32 = "int.negate";

// -1 >> 1 is -1.
var arr1: [i32; Negate(RightShift(Negate(1), 1))];
let arr1_p: [i32; 1]* = &arr1;

// -10 >> 2 is -3.
var arr2: [i32; Negate(RightShift(Negate(10), 2))];
let arr2_p: [i32; 3]* = &arr2;

// --- fail_bad_shift.carbon

package BadShift api;

fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
fn Negate(a: i32) -> i32 = "int.negate";

// Shift greater than size is disallowed.
let size_1: i32 = RightShift(1, 31);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:19: ERROR: Shift distance not in range [0, 32) in 1 >> 32.
// CHECK:STDERR: let size_2: i32 = RightShift(1, 32);
// CHECK:STDERR:                   ^~~~~~~~~~~
// CHECK:STDERR:
let size_2: i32 = RightShift(1, 32);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:19: ERROR: Shift distance not in range [0, 32) in 1 >> 33.
// CHECK:STDERR: let size_3: i32 = RightShift(1, 33);
// CHECK:STDERR:                   ^~~~~~~~~~~
// CHECK:STDERR:
let size_3: i32 = RightShift(1, 33);

// Negative shifts aren't allowed either.
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+3]]:21: ERROR: Shift distance not in range [0, 32) in 1 >> -1.
// CHECK:STDERR: let negative: i32 = RightShift(1, Negate(1));
// CHECK:STDERR:                     ^~~~~~~~~~~
let negative: i32 = RightShift(1, Negate(1));

// CHECK:STDOUT: --- int_right_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 22 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.4: type = array_type %.3, i32 [template]
// CHECK:STDOUT:   %.5: type = ptr_type [i32; 5] [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .RightShift = %RightShift
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %RightShift: <function> = fn_decl @RightShift [template] {
// CHECK:STDOUT:     %a.loc2_15.1: i32 = param a
// CHECK:STDOUT:     @RightShift.%a: i32 = bind_name a, %a.loc2_15.1
// CHECK:STDOUT:     %b.loc2_23.1: i32 = param b
// CHECK:STDOUT:     @RightShift.%b: i32 = bind_name b, %b.loc2_23.1
// CHECK:STDOUT:     @RightShift.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RightShift.ref: <function> = name_ref RightShift, %RightShift [template = %RightShift]
// CHECK:STDOUT:   %.loc4_27: i32 = int_literal 22 [template = constants.%.1]
// CHECK:STDOUT:   %.loc4_31: i32 = int_literal 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_26: init i32 = call %RightShift.ref(%.loc4_27, %.loc4_31) [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_33: type = array_type %.loc4_26, i32 [template = constants.%.4]
// CHECK:STDOUT:   %arr.var: ref [i32; 5] = var arr
// CHECK:STDOUT:   %arr: ref [i32; 5] = bind_name arr, %arr.var
// CHECK:STDOUT:   %.loc5_18: i32 = int_literal 5 [template = constants.%.3]
// CHECK:STDOUT:   %.loc5_19: type = array_type %.loc5_18, i32 [template = constants.%.4]
// CHECK:STDOUT:   %.loc5_20: type = ptr_type [i32; 5] [template = constants.%.5]
// CHECK:STDOUT:   %arr.ref: ref [i32; 5] = name_ref arr, %arr
// CHECK:STDOUT:   %.loc5_24: [i32; 5]* = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: [i32; 5]* = bind_name arr_p, %.loc5_24
// CHECK:STDOUT:   %RuntimeCall: <function> = fn_decl @RuntimeCall [template] {
// CHECK:STDOUT:     %a.loc7_16.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCall.%a: i32 = bind_name a, %a.loc7_16.1
// CHECK:STDOUT:     %b.loc7_24.1: i32 = param b
// CHECK:STDOUT:     @RuntimeCall.%b: i32 = bind_name b, %b.loc7_24.1
// CHECK:STDOUT:     @RuntimeCall.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RightShift(%a: i32, %b: i32) -> i32 = "int.right_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a: i32, %b: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %RightShift.ref: <function> = name_ref RightShift, file.%RightShift [template = file.%RightShift]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %.loc8_20.1: init i32 = call %RightShift.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc8_26: i32 = value_of_initializer %.loc8_20.1
// CHECK:STDOUT:   %.loc8_20.2: i32 = converted %.loc8_20.1, %.loc8_26
// CHECK:STDOUT:   return %.loc8_20.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- arith_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal -1 [template]
// CHECK:STDOUT:   %.3: type = array_type %.1, i32 [template]
// CHECK:STDOUT:   %.4: type = ptr_type [i32; 1] [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 10 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal -10 [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.8: i32 = int_literal -3 [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.10: type = array_type %.9, i32 [template]
// CHECK:STDOUT:   %.11: type = ptr_type [i32; 3] [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .RightShift = %RightShift
// CHECK:STDOUT:     .Negate = %Negate
// CHECK:STDOUT:     .arr1 = %arr1
// CHECK:STDOUT:     .arr2 = %arr2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %RightShift: <function> = fn_decl @RightShift [template] {
// CHECK:STDOUT:     %a.loc6_15.1: i32 = param a
// CHECK:STDOUT:     @RightShift.%a: i32 = bind_name a, %a.loc6_15.1
// CHECK:STDOUT:     %b.loc6_23.1: i32 = param b
// CHECK:STDOUT:     @RightShift.%b: i32 = bind_name b, %b.loc6_23.1
// CHECK:STDOUT:     @RightShift.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate: <function> = fn_decl @Negate [template] {
// CHECK:STDOUT:     %a.loc7_11.1: i32 = param a
// CHECK:STDOUT:     @Negate.%a: i32 = bind_name a, %a.loc7_11.1
// CHECK:STDOUT:     @Negate.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate.ref.loc10_17: <function> = name_ref Negate, %Negate [template = %Negate]
// CHECK:STDOUT:   %RightShift.ref.loc10: <function> = name_ref RightShift, %RightShift [template = %RightShift]
// CHECK:STDOUT:   %Negate.ref.loc10_35: <function> = name_ref Negate, %Negate [template = %Negate]
// CHECK:STDOUT:   %.loc10_42: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_41.1: init i32 = call %Negate.ref.loc10_35(%.loc10_42) [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_46: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_34.1: i32 = value_of_initializer %.loc10_41.1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_41.2: i32 = converted %.loc10_41.1, %.loc10_34.1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_34.2: init i32 = call %RightShift.ref.loc10(%.loc10_41.2, %.loc10_46) [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_23.1: i32 = value_of_initializer %.loc10_34.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_34.3: i32 = converted %.loc10_34.2, %.loc10_23.1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_23.2: init i32 = call %Negate.ref.loc10_17(%.loc10_34.3) [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_49: type = array_type %.loc10_23.2, i32 [template = constants.%.3]
// CHECK:STDOUT:   %arr1.var: ref [i32; 1] = var arr1
// CHECK:STDOUT:   %arr1: ref [i32; 1] = bind_name arr1, %arr1.var
// CHECK:STDOUT:   %.loc11_19: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc11_20: type = array_type %.loc11_19, i32 [template = constants.%.3]
// CHECK:STDOUT:   %.loc11_21: type = ptr_type [i32; 1] [template = constants.%.4]
// CHECK:STDOUT:   %arr1.ref: ref [i32; 1] = name_ref arr1, %arr1
// CHECK:STDOUT:   %.loc11_25: [i32; 1]* = addr_of %arr1.ref
// CHECK:STDOUT:   %arr1_p: [i32; 1]* = bind_name arr1_p, %.loc11_25
// CHECK:STDOUT:   %Negate.ref.loc14_17: <function> = name_ref Negate, %Negate [template = %Negate]
// CHECK:STDOUT:   %RightShift.ref.loc14: <function> = name_ref RightShift, %RightShift [template = %RightShift]
// CHECK:STDOUT:   %Negate.ref.loc14_35: <function> = name_ref Negate, %Negate [template = %Negate]
// CHECK:STDOUT:   %.loc14_42: i32 = int_literal 10 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_41.1: init i32 = call %Negate.ref.loc14_35(%.loc14_42) [template = constants.%.6]
// CHECK:STDOUT:   %.loc14_47: i32 = int_literal 2 [template = constants.%.7]
// CHECK:STDOUT:   %.loc14_34.1: i32 = value_of_initializer %.loc14_41.1 [template = constants.%.6]
// CHECK:STDOUT:   %.loc14_41.2: i32 = converted %.loc14_41.1, %.loc14_34.1 [template = constants.%.6]
// CHECK:STDOUT:   %.loc14_34.2: init i32 = call %RightShift.ref.loc14(%.loc14_41.2, %.loc14_47) [template = constants.%.8]
// CHECK:STDOUT:   %.loc14_23.1: i32 = value_of_initializer %.loc14_34.2 [template = constants.%.8]
// CHECK:STDOUT:   %.loc14_34.3: i32 = converted %.loc14_34.2, %.loc14_23.1 [template = constants.%.8]
// CHECK:STDOUT:   %.loc14_23.2: init i32 = call %Negate.ref.loc14_17(%.loc14_34.3) [template = constants.%.9]
// CHECK:STDOUT:   %.loc14_50: type = array_type %.loc14_23.2, i32 [template = constants.%.10]
// CHECK:STDOUT:   %arr2.var: ref [i32; 3] = var arr2
// CHECK:STDOUT:   %arr2: ref [i32; 3] = bind_name arr2, %arr2.var
// CHECK:STDOUT:   %.loc15_19: i32 = int_literal 3 [template = constants.%.9]
// CHECK:STDOUT:   %.loc15_20: type = array_type %.loc15_19, i32 [template = constants.%.10]
// CHECK:STDOUT:   %.loc15_21: type = ptr_type [i32; 3] [template = constants.%.11]
// CHECK:STDOUT:   %arr2.ref: ref [i32; 3] = name_ref arr2, %arr2
// CHECK:STDOUT:   %.loc15_25: [i32; 3]* = addr_of %arr2.ref
// CHECK:STDOUT:   %arr2_p: [i32; 3]* = bind_name arr2_p, %.loc15_25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RightShift(%a: i32, %b: i32) -> i32 = "int.right_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a: i32) -> i32 = "int.negate";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 31 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 32 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 33 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal -1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .RightShift = %RightShift
// CHECK:STDOUT:     .Negate = %Negate
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %RightShift: <function> = fn_decl @RightShift [template] {
// CHECK:STDOUT:     %a.loc4_15.1: i32 = param a
// CHECK:STDOUT:     @RightShift.%a: i32 = bind_name a, %a.loc4_15.1
// CHECK:STDOUT:     %b.loc4_23.1: i32 = param b
// CHECK:STDOUT:     @RightShift.%b: i32 = bind_name b, %b.loc4_23.1
// CHECK:STDOUT:     @RightShift.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate: <function> = fn_decl @Negate [template] {
// CHECK:STDOUT:     %a.loc5_11.1: i32 = param a
// CHECK:STDOUT:     @Negate.%a: i32 = bind_name a, %a.loc5_11.1
// CHECK:STDOUT:     @Negate.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RightShift.ref.loc8: <function> = name_ref RightShift, %RightShift [template = %RightShift]
// CHECK:STDOUT:   %.loc8_30: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc8_33: i32 = int_literal 31 [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_29.1: init i32 = call %RightShift.ref.loc8(%.loc8_30, %.loc8_33) [template = constants.%.3]
// CHECK:STDOUT:   %.loc8_36: i32 = value_of_initializer %.loc8_29.1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc8_29.2: i32 = converted %.loc8_29.1, %.loc8_36 [template = constants.%.3]
// CHECK:STDOUT:   %size_1: i32 = bind_name size_1, %.loc8_29.2
// CHECK:STDOUT:   %RightShift.ref.loc13: <function> = name_ref RightShift, %RightShift [template = %RightShift]
// CHECK:STDOUT:   %.loc13_30: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc13_33: i32 = int_literal 32 [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_29.1: init i32 = call %RightShift.ref.loc13(%.loc13_30, %.loc13_33) [template = <error>]
// CHECK:STDOUT:   %.loc13_36: i32 = value_of_initializer %.loc13_29.1 [template = <error>]
// CHECK:STDOUT:   %.loc13_29.2: i32 = converted %.loc13_29.1, %.loc13_36 [template = <error>]
// CHECK:STDOUT:   %size_2: i32 = bind_name size_2, %.loc13_29.2
// CHECK:STDOUT:   %RightShift.ref.loc18: <function> = name_ref RightShift, %RightShift [template = %RightShift]
// CHECK:STDOUT:   %.loc18_30: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc18_33: i32 = int_literal 33 [template = constants.%.5]
// CHECK:STDOUT:   %.loc18_29.1: init i32 = call %RightShift.ref.loc18(%.loc18_30, %.loc18_33) [template = <error>]
// CHECK:STDOUT:   %.loc18_36: i32 = value_of_initializer %.loc18_29.1 [template = <error>]
// CHECK:STDOUT:   %.loc18_29.2: i32 = converted %.loc18_29.1, %.loc18_36 [template = <error>]
// CHECK:STDOUT:   %size_3: i32 = bind_name size_3, %.loc18_29.2
// CHECK:STDOUT:   %RightShift.ref.loc24: <function> = name_ref RightShift, %RightShift [template = %RightShift]
// CHECK:STDOUT:   %.loc24_32: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %Negate.ref: <function> = name_ref Negate, %Negate [template = %Negate]
// CHECK:STDOUT:   %.loc24_42: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc24_41.1: init i32 = call %Negate.ref(%.loc24_42) [template = constants.%.6]
// CHECK:STDOUT:   %.loc24_31.1: i32 = value_of_initializer %.loc24_41.1 [template = constants.%.6]
// CHECK:STDOUT:   %.loc24_41.2: i32 = converted %.loc24_41.1, %.loc24_31.1 [template = constants.%.6]
// CHECK:STDOUT:   %.loc24_31.2: init i32 = call %RightShift.ref.loc24(%.loc24_32, %.loc24_41.2) [template = <error>]
// CHECK:STDOUT:   %.loc24_45: i32 = value_of_initializer %.loc24_31.2 [template = <error>]
// CHECK:STDOUT:   %.loc24_31.3: i32 = converted %.loc24_31.2, %.loc24_45 [template = <error>]
// CHECK:STDOUT:   %negative: i32 = bind_name negative, %.loc24_31.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RightShift(%a: i32, %b: i32) -> i32 = "int.right_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a: i32) -> i32 = "int.negate";
// CHECK:STDOUT:
