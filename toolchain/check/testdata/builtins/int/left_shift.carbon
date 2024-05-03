// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- int_left_shift.carbon

fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";

var arr: [i32; LeftShift(5, 2)];
let arr_p: [i32; 20]* = &arr;

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return LeftShift(a, b);
}

// TODO: Test mixed types for LHS and RHS.

// --- fail_bad_shift.carbon

package BadShift api;

fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
fn Negate(a: i32) -> i32 = "int.snegate";

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

// CHECK:STDOUT: --- int_left_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %LeftShift: type = fn_type @LeftShift [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: LeftShift = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 20 [template]
// CHECK:STDOUT:   %.5: type = array_type %.4, i32 [template]
// CHECK:STDOUT:   %.6: type = ptr_type [i32; 20] [template]
// CHECK:STDOUT:   %RuntimeCall: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %struct.2: RuntimeCall = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .LeftShift = %LeftShift.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %LeftShift.decl: LeftShift = fn_decl @LeftShift [template = constants.%struct.1] {
// CHECK:STDOUT:     %a.loc2_14.1: i32 = param a
// CHECK:STDOUT:     @LeftShift.%a: i32 = bind_name a, %a.loc2_14.1
// CHECK:STDOUT:     %b.loc2_22.1: i32 = param b
// CHECK:STDOUT:     @LeftShift.%b: i32 = bind_name b, %b.loc2_22.1
// CHECK:STDOUT:     @LeftShift.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %LeftShift.ref: LeftShift = name_ref LeftShift, %LeftShift.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc4_26: i32 = int_literal 5 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_29: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %int.left_shift: init i32 = call %LeftShift.ref(%.loc4_26, %.loc4_29) [template = constants.%.4]
// CHECK:STDOUT:   %.loc4_31: type = array_type %int.left_shift, i32 [template = constants.%.5]
// CHECK:STDOUT:   %arr.var: ref [i32; 20] = var arr
// CHECK:STDOUT:   %arr: ref [i32; 20] = bind_name arr, %arr.var
// CHECK:STDOUT:   %.loc5_18: i32 = int_literal 20 [template = constants.%.4]
// CHECK:STDOUT:   %.loc5_20: type = array_type %.loc5_18, i32 [template = constants.%.5]
// CHECK:STDOUT:   %.loc5_21: type = ptr_type [i32; 20] [template = constants.%.6]
// CHECK:STDOUT:   %arr.ref: ref [i32; 20] = name_ref arr, %arr
// CHECK:STDOUT:   %.loc5_25: [i32; 20]* = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: [i32; 20]* = bind_name arr_p, %.loc5_25
// CHECK:STDOUT:   %RuntimeCall.decl: RuntimeCall = fn_decl @RuntimeCall [template = constants.%struct.2] {
// CHECK:STDOUT:     %a.loc7_16.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCall.%a: i32 = bind_name a, %a.loc7_16.1
// CHECK:STDOUT:     %b.loc7_24.1: i32 = param b
// CHECK:STDOUT:     @RuntimeCall.%b: i32 = bind_name b, %b.loc7_24.1
// CHECK:STDOUT:     @RuntimeCall.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LeftShift(%a: i32, %b: i32) -> i32 = "int.left_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a: i32, %b: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %LeftShift.ref: LeftShift = name_ref LeftShift, file.%LeftShift.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %int.left_shift: init i32 = call %LeftShift.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc8_25.1: i32 = value_of_initializer %int.left_shift
// CHECK:STDOUT:   %.loc8_25.2: i32 = converted %int.left_shift, %.loc8_25.1
// CHECK:STDOUT:   return %.loc8_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %LeftShift: type = fn_type @LeftShift [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: LeftShift = struct_value () [template]
// CHECK:STDOUT:   %Negate: type = fn_type @Negate [template]
// CHECK:STDOUT:   %struct.2: Negate = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 31 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal -2147483648 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 32 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 33 [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 1000 [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.9: i32 = int_literal -1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .LeftShift = %LeftShift.decl
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %LeftShift.decl: LeftShift = fn_decl @LeftShift [template = constants.%struct.1] {
// CHECK:STDOUT:     %a.loc4_14.1: i32 = param a
// CHECK:STDOUT:     @LeftShift.%a: i32 = bind_name a, %a.loc4_14.1
// CHECK:STDOUT:     %b.loc4_22.1: i32 = param b
// CHECK:STDOUT:     @LeftShift.%b: i32 = bind_name b, %b.loc4_22.1
// CHECK:STDOUT:     @LeftShift.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate.decl: Negate = fn_decl @Negate [template = constants.%struct.2] {
// CHECK:STDOUT:     %a.loc5_11.1: i32 = param a
// CHECK:STDOUT:     @Negate.%a: i32 = bind_name a, %a.loc5_11.1
// CHECK:STDOUT:     @Negate.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %LeftShift.ref.loc8: LeftShift = name_ref LeftShift, %LeftShift.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc8_29: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_32: i32 = int_literal 31 [template = constants.%.3]
// CHECK:STDOUT:   %int.left_shift.loc8: init i32 = call %LeftShift.ref.loc8(%.loc8_29, %.loc8_32) [template = constants.%.4]
// CHECK:STDOUT:   %.loc8_35.1: i32 = value_of_initializer %int.left_shift.loc8 [template = constants.%.4]
// CHECK:STDOUT:   %.loc8_35.2: i32 = converted %int.left_shift.loc8, %.loc8_35.1 [template = constants.%.4]
// CHECK:STDOUT:   %size_1: i32 = bind_name size_1, %.loc8_35.2
// CHECK:STDOUT:   %LeftShift.ref.loc13: LeftShift = name_ref LeftShift, %LeftShift.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc13_29: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc13_32: i32 = int_literal 32 [template = constants.%.5]
// CHECK:STDOUT:   %int.left_shift.loc13: init i32 = call %LeftShift.ref.loc13(%.loc13_29, %.loc13_32) [template = <error>]
// CHECK:STDOUT:   %.loc13_35.1: i32 = value_of_initializer %int.left_shift.loc13 [template = <error>]
// CHECK:STDOUT:   %.loc13_35.2: i32 = converted %int.left_shift.loc13, %.loc13_35.1 [template = <error>]
// CHECK:STDOUT:   %size_2: i32 = bind_name size_2, %.loc13_35.2
// CHECK:STDOUT:   %LeftShift.ref.loc18: LeftShift = name_ref LeftShift, %LeftShift.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc18_29: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc18_32: i32 = int_literal 33 [template = constants.%.6]
// CHECK:STDOUT:   %int.left_shift.loc18: init i32 = call %LeftShift.ref.loc18(%.loc18_29, %.loc18_32) [template = <error>]
// CHECK:STDOUT:   %.loc18_35.1: i32 = value_of_initializer %int.left_shift.loc18 [template = <error>]
// CHECK:STDOUT:   %.loc18_35.2: i32 = converted %int.left_shift.loc18, %.loc18_35.1 [template = <error>]
// CHECK:STDOUT:   %size_3: i32 = bind_name size_3, %.loc18_35.2
// CHECK:STDOUT:   %LeftShift.ref.loc21: LeftShift = name_ref LeftShift, %LeftShift.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc21_33: i32 = int_literal 1000 [template = constants.%.7]
// CHECK:STDOUT:   %.loc21_39: i32 = int_literal 31 [template = constants.%.3]
// CHECK:STDOUT:   %int.left_shift.loc21: init i32 = call %LeftShift.ref.loc21(%.loc21_33, %.loc21_39) [template = constants.%.8]
// CHECK:STDOUT:   %.loc21_42.1: i32 = value_of_initializer %int.left_shift.loc21 [template = constants.%.8]
// CHECK:STDOUT:   %.loc21_42.2: i32 = converted %int.left_shift.loc21, %.loc21_42.1 [template = constants.%.8]
// CHECK:STDOUT:   %overflow_1: i32 = bind_name overflow_1, %.loc21_42.2
// CHECK:STDOUT:   %LeftShift.ref.loc26: LeftShift = name_ref LeftShift, %LeftShift.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc26_33: i32 = int_literal 1000 [template = constants.%.7]
// CHECK:STDOUT:   %.loc26_39: i32 = int_literal 32 [template = constants.%.5]
// CHECK:STDOUT:   %int.left_shift.loc26: init i32 = call %LeftShift.ref.loc26(%.loc26_33, %.loc26_39) [template = <error>]
// CHECK:STDOUT:   %.loc26_42.1: i32 = value_of_initializer %int.left_shift.loc26 [template = <error>]
// CHECK:STDOUT:   %.loc26_42.2: i32 = converted %int.left_shift.loc26, %.loc26_42.1 [template = <error>]
// CHECK:STDOUT:   %overflow_2: i32 = bind_name overflow_2, %.loc26_42.2
// CHECK:STDOUT:   %LeftShift.ref.loc29: LeftShift = name_ref LeftShift, %LeftShift.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc29_36: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc29_39: i32 = int_literal 31 [template = constants.%.3]
// CHECK:STDOUT:   %int.left_shift.loc29: init i32 = call %LeftShift.ref.loc29(%.loc29_36, %.loc29_39) [template = constants.%.8]
// CHECK:STDOUT:   %.loc29_42.1: i32 = value_of_initializer %int.left_shift.loc29 [template = constants.%.8]
// CHECK:STDOUT:   %.loc29_42.2: i32 = converted %int.left_shift.loc29, %.loc29_42.1 [template = constants.%.8]
// CHECK:STDOUT:   %no_overflow_1: i32 = bind_name no_overflow_1, %.loc29_42.2
// CHECK:STDOUT:   %LeftShift.ref.loc34: LeftShift = name_ref LeftShift, %LeftShift.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc34_36: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc34_39: i32 = int_literal 32 [template = constants.%.5]
// CHECK:STDOUT:   %int.left_shift.loc34: init i32 = call %LeftShift.ref.loc34(%.loc34_36, %.loc34_39) [template = <error>]
// CHECK:STDOUT:   %.loc34_42.1: i32 = value_of_initializer %int.left_shift.loc34 [template = <error>]
// CHECK:STDOUT:   %.loc34_42.2: i32 = converted %int.left_shift.loc34, %.loc34_42.1 [template = <error>]
// CHECK:STDOUT:   %no_overflow_2: i32 = bind_name no_overflow_2, %.loc34_42.2
// CHECK:STDOUT:   %LeftShift.ref.loc40: LeftShift = name_ref LeftShift, %LeftShift.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc40_31: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %Negate.ref: Negate = name_ref Negate, %Negate.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc40_41: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %int.snegate: init i32 = call %Negate.ref(%.loc40_41) [template = constants.%.9]
// CHECK:STDOUT:   %.loc40_30.1: i32 = value_of_initializer %int.snegate [template = constants.%.9]
// CHECK:STDOUT:   %.loc40_30.2: i32 = converted %int.snegate, %.loc40_30.1 [template = constants.%.9]
// CHECK:STDOUT:   %int.left_shift.loc40: init i32 = call %LeftShift.ref.loc40(%.loc40_31, %.loc40_30.2) [template = <error>]
// CHECK:STDOUT:   %.loc40_44.1: i32 = value_of_initializer %int.left_shift.loc40 [template = <error>]
// CHECK:STDOUT:   %.loc40_44.2: i32 = converted %int.left_shift.loc40, %.loc40_44.1 [template = <error>]
// CHECK:STDOUT:   %negative: i32 = bind_name negative, %.loc40_44.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LeftShift(%a: i32, %b: i32) -> i32 = "int.left_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a: i32) -> i32 = "int.snegate";
// CHECK:STDOUT:
