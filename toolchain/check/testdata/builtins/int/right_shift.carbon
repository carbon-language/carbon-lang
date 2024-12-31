// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
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

fn RuntimeCallIsValid(a: i32, b: i32) -> i32 {
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
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let size_2: i32 = RightShift(1, 32);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:19: error: shift distance not in range [0, 32) in 1 >> 33 [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let size_3: i32 = RightShift(1, 33);
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let size_3: i32 = RightShift(1, 33);

// Negative shifts aren't allowed either.
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+3]]:21: error: shift distance not in range [0, 32) in 1 >> -1 [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let negative: i32 = RightShift(1, Negate(1));
// CHECK:STDERR:                     ^~~~~~~~~~~~~~~~~~~~~~~~
let negative: i32 = RightShift(1, Negate(1));
