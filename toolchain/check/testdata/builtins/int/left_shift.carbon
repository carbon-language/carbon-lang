// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/left_shift.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/left_shift.carbon

// --- int_left_shift.carbon

fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";

var arr: [i32; LeftShift(5, 2)];
let arr_p: [i32; 20]* = &arr;

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return LeftShift(a, b);
}

// TODO: Test mixed types for LHS and RHS.

// --- fail_bad_shift.carbon

package BadShift;

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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %LeftShift.type: type = fn_type @LeftShift [template]
// CHECK:STDOUT:   %LeftShift: %LeftShift.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 20 [template]
// CHECK:STDOUT:   %.5: type = array_type %.4, i32 [template]
// CHECK:STDOUT:   %.6: type = ptr_type %.5 [template]
// CHECK:STDOUT:   %RuntimeCall.type: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %RuntimeCall: %RuntimeCall.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .LeftShift = %LeftShift.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .arr_p = @__global_init.%arr_p
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %LeftShift.decl: %LeftShift.type = fn_decl @LeftShift [template = constants.%LeftShift] {
// CHECK:STDOUT:     %int.make_type_32.loc2_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc2_17.1: type = value_of_initializer %int.make_type_32.loc2_17 [template = i32]
// CHECK:STDOUT:     %.loc2_17.2: type = converted %int.make_type_32.loc2_17, %.loc2_17.1 [template = i32]
// CHECK:STDOUT:     %a.loc2_14.1: i32 = param a
// CHECK:STDOUT:     @LeftShift.%a: i32 = bind_name a, %a.loc2_14.1
// CHECK:STDOUT:     %int.make_type_32.loc2_25: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc2_25.1: type = value_of_initializer %int.make_type_32.loc2_25 [template = i32]
// CHECK:STDOUT:     %.loc2_25.2: type = converted %int.make_type_32.loc2_25, %.loc2_25.1 [template = i32]
// CHECK:STDOUT:     %b.loc2_22.1: i32 = param b
// CHECK:STDOUT:     @LeftShift.%b: i32 = bind_name b, %b.loc2_22.1
// CHECK:STDOUT:     %int.make_type_32.loc2_33: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc2_33.1: type = value_of_initializer %int.make_type_32.loc2_33 [template = i32]
// CHECK:STDOUT:     %.loc2_33.2: type = converted %int.make_type_32.loc2_33, %.loc2_33.1 [template = i32]
// CHECK:STDOUT:     @LeftShift.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc4: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %LeftShift.ref: %LeftShift.type = name_ref LeftShift, %LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %.loc4_26: i32 = int_literal 5 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_29: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %int.left_shift: init i32 = call %LeftShift.ref(%.loc4_26, %.loc4_29) [template = constants.%.4]
// CHECK:STDOUT:   %.loc4_11.1: type = value_of_initializer %int.make_type_32.loc4 [template = i32]
// CHECK:STDOUT:   %.loc4_11.2: type = converted %int.make_type_32.loc4, %.loc4_11.1 [template = i32]
// CHECK:STDOUT:   %.loc4_31: type = array_type %int.left_shift, i32 [template = constants.%.5]
// CHECK:STDOUT:   %arr.var: ref %.5 = var arr
// CHECK:STDOUT:   %arr: ref %.5 = bind_name arr, %arr.var
// CHECK:STDOUT:   %int.make_type_32.loc5: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_18: i32 = int_literal 20 [template = constants.%.4]
// CHECK:STDOUT:   %.loc5_13.1: type = value_of_initializer %int.make_type_32.loc5 [template = i32]
// CHECK:STDOUT:   %.loc5_13.2: type = converted %int.make_type_32.loc5, %.loc5_13.1 [template = i32]
// CHECK:STDOUT:   %.loc5_20: type = array_type %.loc5_18, i32 [template = constants.%.5]
// CHECK:STDOUT:   %.loc5_21: type = ptr_type %.5 [template = constants.%.6]
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %int.make_type_32.loc7_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc7_19.1: type = value_of_initializer %int.make_type_32.loc7_19 [template = i32]
// CHECK:STDOUT:     %.loc7_19.2: type = converted %int.make_type_32.loc7_19, %.loc7_19.1 [template = i32]
// CHECK:STDOUT:     %a.loc7_16.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCall.%a: i32 = bind_name a, %a.loc7_16.1
// CHECK:STDOUT:     %int.make_type_32.loc7_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc7_27.1: type = value_of_initializer %int.make_type_32.loc7_27 [template = i32]
// CHECK:STDOUT:     %.loc7_27.2: type = converted %int.make_type_32.loc7_27, %.loc7_27.1 [template = i32]
// CHECK:STDOUT:     %b.loc7_24.1: i32 = param b
// CHECK:STDOUT:     @RuntimeCall.%b: i32 = bind_name b, %b.loc7_24.1
// CHECK:STDOUT:     %int.make_type_32.loc7_35: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc7_35.1: type = value_of_initializer %int.make_type_32.loc7_35 [template = i32]
// CHECK:STDOUT:     %.loc7_35.2: type = converted %int.make_type_32.loc7_35, %.loc7_35.1 [template = i32]
// CHECK:STDOUT:     @RuntimeCall.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LeftShift(%a: i32, %b: i32) -> i32 = "int.left_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a: i32, %b: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %LeftShift.ref: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %int.left_shift: init i32 = call %LeftShift.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc8_25.1: i32 = value_of_initializer %int.left_shift
// CHECK:STDOUT:   %.loc8_25.2: i32 = converted %int.left_shift, %.loc8_25.1
// CHECK:STDOUT:   return %.loc8_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr.ref: ref %.5 = name_ref arr, file.%arr
// CHECK:STDOUT:   %.loc5: %.6 = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: %.6 = bind_name arr_p, %.loc5
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %LeftShift.type: type = fn_type @LeftShift [template]
// CHECK:STDOUT:   %LeftShift: %LeftShift.type = struct_value () [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
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
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .LeftShift = %LeftShift.decl
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .size_1 = @__global_init.%size_1
// CHECK:STDOUT:     .size_2 = @__global_init.%size_2
// CHECK:STDOUT:     .size_3 = @__global_init.%size_3
// CHECK:STDOUT:     .overflow_1 = @__global_init.%overflow_1
// CHECK:STDOUT:     .overflow_2 = @__global_init.%overflow_2
// CHECK:STDOUT:     .no_overflow_1 = @__global_init.%no_overflow_1
// CHECK:STDOUT:     .no_overflow_2 = @__global_init.%no_overflow_2
// CHECK:STDOUT:     .negative = @__global_init.%negative
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %LeftShift.decl: %LeftShift.type = fn_decl @LeftShift [template = constants.%LeftShift] {
// CHECK:STDOUT:     %int.make_type_32.loc4_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_17.1: type = value_of_initializer %int.make_type_32.loc4_17 [template = i32]
// CHECK:STDOUT:     %.loc4_17.2: type = converted %int.make_type_32.loc4_17, %.loc4_17.1 [template = i32]
// CHECK:STDOUT:     %a.loc4_14.1: i32 = param a
// CHECK:STDOUT:     @LeftShift.%a: i32 = bind_name a, %a.loc4_14.1
// CHECK:STDOUT:     %int.make_type_32.loc4_25: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_25.1: type = value_of_initializer %int.make_type_32.loc4_25 [template = i32]
// CHECK:STDOUT:     %.loc4_25.2: type = converted %int.make_type_32.loc4_25, %.loc4_25.1 [template = i32]
// CHECK:STDOUT:     %b.loc4_22.1: i32 = param b
// CHECK:STDOUT:     @LeftShift.%b: i32 = bind_name b, %b.loc4_22.1
// CHECK:STDOUT:     %int.make_type_32.loc4_33: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_33.1: type = value_of_initializer %int.make_type_32.loc4_33 [template = i32]
// CHECK:STDOUT:     %.loc4_33.2: type = converted %int.make_type_32.loc4_33, %.loc4_33.1 [template = i32]
// CHECK:STDOUT:     @LeftShift.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate.decl: %Negate.type = fn_decl @Negate [template = constants.%Negate] {
// CHECK:STDOUT:     %int.make_type_32.loc5_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_14.1: type = value_of_initializer %int.make_type_32.loc5_14 [template = i32]
// CHECK:STDOUT:     %.loc5_14.2: type = converted %int.make_type_32.loc5_14, %.loc5_14.1 [template = i32]
// CHECK:STDOUT:     %a.loc5_11.1: i32 = param a
// CHECK:STDOUT:     @Negate.%a: i32 = bind_name a, %a.loc5_11.1
// CHECK:STDOUT:     %int.make_type_32.loc5_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_22.1: type = value_of_initializer %int.make_type_32.loc5_22 [template = i32]
// CHECK:STDOUT:     %.loc5_22.2: type = converted %int.make_type_32.loc5_22, %.loc5_22.1 [template = i32]
// CHECK:STDOUT:     @Negate.%return: ref i32 = var <return slot>
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
// CHECK:STDOUT:   %int.make_type_32.loc21: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc21_17.1: type = value_of_initializer %int.make_type_32.loc21 [template = i32]
// CHECK:STDOUT:   %.loc21_17.2: type = converted %int.make_type_32.loc21, %.loc21_17.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc26: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc26_17.1: type = value_of_initializer %int.make_type_32.loc26 [template = i32]
// CHECK:STDOUT:   %.loc26_17.2: type = converted %int.make_type_32.loc26, %.loc26_17.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc29: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc29_20.1: type = value_of_initializer %int.make_type_32.loc29 [template = i32]
// CHECK:STDOUT:   %.loc29_20.2: type = converted %int.make_type_32.loc29, %.loc29_20.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc34: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc34_20.1: type = value_of_initializer %int.make_type_32.loc34 [template = i32]
// CHECK:STDOUT:   %.loc34_20.2: type = converted %int.make_type_32.loc34, %.loc34_20.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc40: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc40_15.1: type = value_of_initializer %int.make_type_32.loc40 [template = i32]
// CHECK:STDOUT:   %.loc40_15.2: type = converted %int.make_type_32.loc40, %.loc40_15.1 [template = i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LeftShift(%a: i32, %b: i32) -> i32 = "int.left_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a: i32) -> i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %LeftShift.ref.loc8: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %.loc8_29: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_32: i32 = int_literal 31 [template = constants.%.3]
// CHECK:STDOUT:   %int.left_shift.loc8: init i32 = call %LeftShift.ref.loc8(%.loc8_29, %.loc8_32) [template = constants.%.4]
// CHECK:STDOUT:   %.loc8_35.1: i32 = value_of_initializer %int.left_shift.loc8 [template = constants.%.4]
// CHECK:STDOUT:   %.loc8_35.2: i32 = converted %int.left_shift.loc8, %.loc8_35.1 [template = constants.%.4]
// CHECK:STDOUT:   %size_1: i32 = bind_name size_1, %.loc8_35.2
// CHECK:STDOUT:   %LeftShift.ref.loc13: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %.loc13_29: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc13_32: i32 = int_literal 32 [template = constants.%.5]
// CHECK:STDOUT:   %int.left_shift.loc13: init i32 = call %LeftShift.ref.loc13(%.loc13_29, %.loc13_32) [template = <error>]
// CHECK:STDOUT:   %.loc13_35.1: i32 = value_of_initializer %int.left_shift.loc13 [template = <error>]
// CHECK:STDOUT:   %.loc13_35.2: i32 = converted %int.left_shift.loc13, %.loc13_35.1 [template = <error>]
// CHECK:STDOUT:   %size_2: i32 = bind_name size_2, %.loc13_35.2
// CHECK:STDOUT:   %LeftShift.ref.loc18: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %.loc18_29: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc18_32: i32 = int_literal 33 [template = constants.%.6]
// CHECK:STDOUT:   %int.left_shift.loc18: init i32 = call %LeftShift.ref.loc18(%.loc18_29, %.loc18_32) [template = <error>]
// CHECK:STDOUT:   %.loc18_35.1: i32 = value_of_initializer %int.left_shift.loc18 [template = <error>]
// CHECK:STDOUT:   %.loc18_35.2: i32 = converted %int.left_shift.loc18, %.loc18_35.1 [template = <error>]
// CHECK:STDOUT:   %size_3: i32 = bind_name size_3, %.loc18_35.2
// CHECK:STDOUT:   %LeftShift.ref.loc21: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %.loc21_33: i32 = int_literal 1000 [template = constants.%.7]
// CHECK:STDOUT:   %.loc21_39: i32 = int_literal 31 [template = constants.%.3]
// CHECK:STDOUT:   %int.left_shift.loc21: init i32 = call %LeftShift.ref.loc21(%.loc21_33, %.loc21_39) [template = constants.%.8]
// CHECK:STDOUT:   %.loc21_42.1: i32 = value_of_initializer %int.left_shift.loc21 [template = constants.%.8]
// CHECK:STDOUT:   %.loc21_42.2: i32 = converted %int.left_shift.loc21, %.loc21_42.1 [template = constants.%.8]
// CHECK:STDOUT:   %overflow_1: i32 = bind_name overflow_1, %.loc21_42.2
// CHECK:STDOUT:   %LeftShift.ref.loc26: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %.loc26_33: i32 = int_literal 1000 [template = constants.%.7]
// CHECK:STDOUT:   %.loc26_39: i32 = int_literal 32 [template = constants.%.5]
// CHECK:STDOUT:   %int.left_shift.loc26: init i32 = call %LeftShift.ref.loc26(%.loc26_33, %.loc26_39) [template = <error>]
// CHECK:STDOUT:   %.loc26_42.1: i32 = value_of_initializer %int.left_shift.loc26 [template = <error>]
// CHECK:STDOUT:   %.loc26_42.2: i32 = converted %int.left_shift.loc26, %.loc26_42.1 [template = <error>]
// CHECK:STDOUT:   %overflow_2: i32 = bind_name overflow_2, %.loc26_42.2
// CHECK:STDOUT:   %LeftShift.ref.loc29: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %.loc29_36: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc29_39: i32 = int_literal 31 [template = constants.%.3]
// CHECK:STDOUT:   %int.left_shift.loc29: init i32 = call %LeftShift.ref.loc29(%.loc29_36, %.loc29_39) [template = constants.%.8]
// CHECK:STDOUT:   %.loc29_42.1: i32 = value_of_initializer %int.left_shift.loc29 [template = constants.%.8]
// CHECK:STDOUT:   %.loc29_42.2: i32 = converted %int.left_shift.loc29, %.loc29_42.1 [template = constants.%.8]
// CHECK:STDOUT:   %no_overflow_1: i32 = bind_name no_overflow_1, %.loc29_42.2
// CHECK:STDOUT:   %LeftShift.ref.loc34: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %.loc34_36: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc34_39: i32 = int_literal 32 [template = constants.%.5]
// CHECK:STDOUT:   %int.left_shift.loc34: init i32 = call %LeftShift.ref.loc34(%.loc34_36, %.loc34_39) [template = <error>]
// CHECK:STDOUT:   %.loc34_42.1: i32 = value_of_initializer %int.left_shift.loc34 [template = <error>]
// CHECK:STDOUT:   %.loc34_42.2: i32 = converted %int.left_shift.loc34, %.loc34_42.1 [template = <error>]
// CHECK:STDOUT:   %no_overflow_2: i32 = bind_name no_overflow_2, %.loc34_42.2
// CHECK:STDOUT:   %LeftShift.ref.loc40: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %.loc40_31: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %Negate.ref: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc40_41: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %int.snegate: init i32 = call %Negate.ref(%.loc40_41) [template = constants.%.9]
// CHECK:STDOUT:   %.loc40_30.1: i32 = value_of_initializer %int.snegate [template = constants.%.9]
// CHECK:STDOUT:   %.loc40_30.2: i32 = converted %int.snegate, %.loc40_30.1 [template = constants.%.9]
// CHECK:STDOUT:   %int.left_shift.loc40: init i32 = call %LeftShift.ref.loc40(%.loc40_31, %.loc40_30.2) [template = <error>]
// CHECK:STDOUT:   %.loc40_44.1: i32 = value_of_initializer %int.left_shift.loc40 [template = <error>]
// CHECK:STDOUT:   %.loc40_44.2: i32 = converted %int.left_shift.loc40, %.loc40_44.1 [template = <error>]
// CHECK:STDOUT:   %negative: i32 = bind_name negative, %.loc40_44.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
