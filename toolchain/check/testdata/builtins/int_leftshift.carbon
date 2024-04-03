// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- int_leftshift.carbon

fn LeftShift(a: i32, b: i32) -> i32 = "int.leftshift";

var arr: [i32; LeftShift(5, 2)];
let arr_p: [i32; 20]* = &arr;

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return LeftShift(a, b);
}

// TODO: Test mixed types for LHS and RHS.

// --- fail_bad_shift.carbon

package BadShift api;

fn LeftShift(a: i32, b: i32) -> i32 = "int.leftshift";
fn Negate(a: i32) -> i32 = "int.negate";

// Shift greater than size is disallowed.
let size_1: i32 = LeftShift(1, 31);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:19: ERROR: Shift distance not in range [0, 32) in 1 << 32.
// CHECK:STDERR: let size_2: i32 = LeftShift(1, 32);
// CHECK:STDERR:                   ^~~~~~~~~~
// CHECK:STDERR:
let size_2: i32 = LeftShift(1, 32);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:19: ERROR: Shift distance not in range [0, 32) in 1 << 33.
// CHECK:STDERR: let size_3: i32 = LeftShift(1, 33);
// CHECK:STDERR:                   ^~~~~~~~~~
// CHECK:STDERR:
let size_3: i32 = LeftShift(1, 33);

// Overflow is allowed if the shift distance is in bounds.
let overflow_1: i32 = LeftShift(1000, 31);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:23: ERROR: Shift distance not in range [0, 32) in 1000 << 32.
// CHECK:STDERR: let overflow_2: i32 = LeftShift(1000, 32);
// CHECK:STDERR:                       ^~~~~~~~~~
// CHECK:STDERR:
let overflow_2: i32 = LeftShift(1000, 32);

// Oversize shifts aren't allowed even if there's no overflow.
let no_overflow_1: i32 = LeftShift(0, 31);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:26: ERROR: Shift distance not in range [0, 32) in 0 << 32.
// CHECK:STDERR: let no_overflow_2: i32 = LeftShift(0, 32);
// CHECK:STDERR:                          ^~~~~~~~~~
// CHECK:STDERR:
let no_overflow_2: i32 = LeftShift(0, 32);

// Negative shifts aren't allowed either.
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+3]]:21: ERROR: Shift distance not in range [0, 32) in 1 << -1.
// CHECK:STDERR: let negative: i32 = LeftShift(1, Negate(1));
// CHECK:STDERR:                     ^~~~~~~~~~
let negative: i32 = LeftShift(1, Negate(1));

// CHECK:STDOUT: --- int_leftshift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 20 [template]
// CHECK:STDOUT:   %.4: type = array_type %.3, i32 [template]
// CHECK:STDOUT:   %.5: type = ptr_type [i32; 20] [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .LeftShift = %LeftShift
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %LeftShift: <function> = fn_decl @LeftShift [template] {
// CHECK:STDOUT:     %a.loc2_14.1: i32 = param a
// CHECK:STDOUT:     @LeftShift.%a: i32 = bind_name a, %a.loc2_14.1
// CHECK:STDOUT:     %b.loc2_22.1: i32 = param b
// CHECK:STDOUT:     @LeftShift.%b: i32 = bind_name b, %b.loc2_22.1
// CHECK:STDOUT:     %return.var.loc2: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %LeftShift.ref: <function> = name_ref LeftShift, %LeftShift [template = %LeftShift]
// CHECK:STDOUT:   %.loc4_26: i32 = int_literal 5 [template = constants.%.1]
// CHECK:STDOUT:   %.loc4_29: i32 = int_literal 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_25: init i32 = call %LeftShift.ref(%.loc4_26, %.loc4_29) [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_31: type = array_type %.loc4_25, i32 [template = constants.%.4]
// CHECK:STDOUT:   %arr.var: ref [i32; 20] = var arr
// CHECK:STDOUT:   %arr: ref [i32; 20] = bind_name arr, %arr.var
// CHECK:STDOUT:   %.loc5_18: i32 = int_literal 20 [template = constants.%.3]
// CHECK:STDOUT:   %.loc5_20: type = array_type %.loc5_18, i32 [template = constants.%.4]
// CHECK:STDOUT:   %.loc5_21: type = ptr_type [i32; 20] [template = constants.%.5]
// CHECK:STDOUT:   %arr.ref: ref [i32; 20] = name_ref arr, %arr
// CHECK:STDOUT:   %.loc5_25: [i32; 20]* = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: [i32; 20]* = bind_name arr_p, %.loc5_25
// CHECK:STDOUT:   %RuntimeCall: <function> = fn_decl @RuntimeCall [template] {
// CHECK:STDOUT:     %a.loc7_16.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCall.%a: i32 = bind_name a, %a.loc7_16.1
// CHECK:STDOUT:     %b.loc7_24.1: i32 = param b
// CHECK:STDOUT:     @RuntimeCall.%b: i32 = bind_name b, %b.loc7_24.1
// CHECK:STDOUT:     %return.var.loc7: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LeftShift(%a: i32, %b: i32) -> i32 = "int.leftshift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a: i32, %b: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %LeftShift.ref: <function> = name_ref LeftShift, file.%LeftShift [template = file.%LeftShift]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %.loc8_19.1: init i32 = call %LeftShift.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc8_25: i32 = value_of_initializer %.loc8_19.1
// CHECK:STDOUT:   %.loc8_19.2: i32 = converted %.loc8_19.1, %.loc8_25
// CHECK:STDOUT:   return %.loc8_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 31 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal -2147483648 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 32 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 33 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 1000 [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.8: i32 = int_literal -1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .LeftShift = %LeftShift
// CHECK:STDOUT:     .Negate = %Negate
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %LeftShift: <function> = fn_decl @LeftShift [template] {
// CHECK:STDOUT:     %a.loc4_14.1: i32 = param a
// CHECK:STDOUT:     @LeftShift.%a: i32 = bind_name a, %a.loc4_14.1
// CHECK:STDOUT:     %b.loc4_22.1: i32 = param b
// CHECK:STDOUT:     @LeftShift.%b: i32 = bind_name b, %b.loc4_22.1
// CHECK:STDOUT:     %return.var.loc4: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate: <function> = fn_decl @Negate [template] {
// CHECK:STDOUT:     %a.loc5_11.1: i32 = param a
// CHECK:STDOUT:     @Negate.%a: i32 = bind_name a, %a.loc5_11.1
// CHECK:STDOUT:     %return.var.loc5: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %LeftShift.ref.loc8: <function> = name_ref LeftShift, %LeftShift [template = %LeftShift]
// CHECK:STDOUT:   %.loc8_29: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc8_32: i32 = int_literal 31 [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_28.1: init i32 = call %LeftShift.ref.loc8(%.loc8_29, %.loc8_32) [template = constants.%.3]
// CHECK:STDOUT:   %.loc8_35: i32 = value_of_initializer %.loc8_28.1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc8_28.2: i32 = converted %.loc8_28.1, %.loc8_35 [template = constants.%.3]
// CHECK:STDOUT:   %size_1: i32 = bind_name size_1, %.loc8_28.2
// CHECK:STDOUT:   %LeftShift.ref.loc13: <function> = name_ref LeftShift, %LeftShift [template = %LeftShift]
// CHECK:STDOUT:   %.loc13_29: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc13_32: i32 = int_literal 32 [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_28.1: init i32 = call %LeftShift.ref.loc13(%.loc13_29, %.loc13_32) [template = <error>]
// CHECK:STDOUT:   %.loc13_35: i32 = value_of_initializer %.loc13_28.1 [template = <error>]
// CHECK:STDOUT:   %.loc13_28.2: i32 = converted %.loc13_28.1, %.loc13_35 [template = <error>]
// CHECK:STDOUT:   %size_2: i32 = bind_name size_2, %.loc13_28.2
// CHECK:STDOUT:   %LeftShift.ref.loc18: <function> = name_ref LeftShift, %LeftShift [template = %LeftShift]
// CHECK:STDOUT:   %.loc18_29: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc18_32: i32 = int_literal 33 [template = constants.%.5]
// CHECK:STDOUT:   %.loc18_28.1: init i32 = call %LeftShift.ref.loc18(%.loc18_29, %.loc18_32) [template = <error>]
// CHECK:STDOUT:   %.loc18_35: i32 = value_of_initializer %.loc18_28.1 [template = <error>]
// CHECK:STDOUT:   %.loc18_28.2: i32 = converted %.loc18_28.1, %.loc18_35 [template = <error>]
// CHECK:STDOUT:   %size_3: i32 = bind_name size_3, %.loc18_28.2
// CHECK:STDOUT:   %LeftShift.ref.loc21: <function> = name_ref LeftShift, %LeftShift [template = %LeftShift]
// CHECK:STDOUT:   %.loc21_33: i32 = int_literal 1000 [template = constants.%.6]
// CHECK:STDOUT:   %.loc21_39: i32 = int_literal 31 [template = constants.%.2]
// CHECK:STDOUT:   %.loc21_32.1: init i32 = call %LeftShift.ref.loc21(%.loc21_33, %.loc21_39) [template = constants.%.7]
// CHECK:STDOUT:   %.loc21_42: i32 = value_of_initializer %.loc21_32.1 [template = constants.%.7]
// CHECK:STDOUT:   %.loc21_32.2: i32 = converted %.loc21_32.1, %.loc21_42 [template = constants.%.7]
// CHECK:STDOUT:   %overflow_1: i32 = bind_name overflow_1, %.loc21_32.2
// CHECK:STDOUT:   %LeftShift.ref.loc26: <function> = name_ref LeftShift, %LeftShift [template = %LeftShift]
// CHECK:STDOUT:   %.loc26_33: i32 = int_literal 1000 [template = constants.%.6]
// CHECK:STDOUT:   %.loc26_39: i32 = int_literal 32 [template = constants.%.4]
// CHECK:STDOUT:   %.loc26_32.1: init i32 = call %LeftShift.ref.loc26(%.loc26_33, %.loc26_39) [template = <error>]
// CHECK:STDOUT:   %.loc26_42: i32 = value_of_initializer %.loc26_32.1 [template = <error>]
// CHECK:STDOUT:   %.loc26_32.2: i32 = converted %.loc26_32.1, %.loc26_42 [template = <error>]
// CHECK:STDOUT:   %overflow_2: i32 = bind_name overflow_2, %.loc26_32.2
// CHECK:STDOUT:   %LeftShift.ref.loc29: <function> = name_ref LeftShift, %LeftShift [template = %LeftShift]
// CHECK:STDOUT:   %.loc29_36: i32 = int_literal 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc29_39: i32 = int_literal 31 [template = constants.%.2]
// CHECK:STDOUT:   %.loc29_35.1: init i32 = call %LeftShift.ref.loc29(%.loc29_36, %.loc29_39) [template = constants.%.7]
// CHECK:STDOUT:   %.loc29_42: i32 = value_of_initializer %.loc29_35.1 [template = constants.%.7]
// CHECK:STDOUT:   %.loc29_35.2: i32 = converted %.loc29_35.1, %.loc29_42 [template = constants.%.7]
// CHECK:STDOUT:   %no_overflow_1: i32 = bind_name no_overflow_1, %.loc29_35.2
// CHECK:STDOUT:   %LeftShift.ref.loc34: <function> = name_ref LeftShift, %LeftShift [template = %LeftShift]
// CHECK:STDOUT:   %.loc34_36: i32 = int_literal 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc34_39: i32 = int_literal 32 [template = constants.%.4]
// CHECK:STDOUT:   %.loc34_35.1: init i32 = call %LeftShift.ref.loc34(%.loc34_36, %.loc34_39) [template = <error>]
// CHECK:STDOUT:   %.loc34_42: i32 = value_of_initializer %.loc34_35.1 [template = <error>]
// CHECK:STDOUT:   %.loc34_35.2: i32 = converted %.loc34_35.1, %.loc34_42 [template = <error>]
// CHECK:STDOUT:   %no_overflow_2: i32 = bind_name no_overflow_2, %.loc34_35.2
// CHECK:STDOUT:   %LeftShift.ref.loc40: <function> = name_ref LeftShift, %LeftShift [template = %LeftShift]
// CHECK:STDOUT:   %.loc40_31: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %Negate.ref: <function> = name_ref Negate, %Negate [template = %Negate]
// CHECK:STDOUT:   %.loc40_41: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc40_40.1: init i32 = call %Negate.ref(%.loc40_41) [template = constants.%.8]
// CHECK:STDOUT:   %.loc40_30.1: i32 = value_of_initializer %.loc40_40.1 [template = constants.%.8]
// CHECK:STDOUT:   %.loc40_40.2: i32 = converted %.loc40_40.1, %.loc40_30.1 [template = constants.%.8]
// CHECK:STDOUT:   %.loc40_30.2: init i32 = call %LeftShift.ref.loc40(%.loc40_31, %.loc40_40.2) [template = <error>]
// CHECK:STDOUT:   %.loc40_44: i32 = value_of_initializer %.loc40_30.2 [template = <error>]
// CHECK:STDOUT:   %.loc40_30.3: i32 = converted %.loc40_30.2, %.loc40_44 [template = <error>]
// CHECK:STDOUT:   %negative: i32 = bind_name negative, %.loc40_30.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LeftShift(%a: i32, %b: i32) -> i32 = "int.leftshift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a: i32) -> i32 = "int.negate";
// CHECK:STDOUT:
