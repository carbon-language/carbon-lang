// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
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

fn RuntimeCallIsValid(a: i32, b: i32) -> i32 {
  return LeftShift(a, b);
}

// TODO: Test mixed types for LHS and RHS.

// --- fail_bad_shift.carbon

package BadShift;

fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
fn Negate(a: i32) -> i32 = "int.snegate";

// Shift greater than size is disallowed.
let size_1: i32 = LeftShift(1, 31);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:19: error: shift distance not in range [0, 32) in 1 << 32 [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let size_2: i32 = LeftShift(1, 32);
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
let size_2: i32 = LeftShift(1, 32);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:19: error: shift distance not in range [0, 32) in 1 << 33 [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let size_3: i32 = LeftShift(1, 33);
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
let size_3: i32 = LeftShift(1, 33);

// Overflow is allowed if the shift distance is in bounds.
let overflow_1: i32 = LeftShift(1000, 31);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:23: error: shift distance not in range [0, 32) in 1000 << 32 [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let overflow_2: i32 = LeftShift(1000, 32);
// CHECK:STDERR:                       ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let overflow_2: i32 = LeftShift(1000, 32);

// Oversize shifts aren't allowed even if there's no overflow.
let no_overflow_1: i32 = LeftShift(0, 31);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:26: error: shift distance not in range [0, 32) in 0 << 32 [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let no_overflow_2: i32 = LeftShift(0, 32);
// CHECK:STDERR:                          ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
let no_overflow_2: i32 = LeftShift(0, 32);

// Negative shifts aren't allowed either.
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+3]]:21: error: shift distance not in range [0, 32) in 1 << -1 [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let negative: i32 = LeftShift(1, Negate(1));
// CHECK:STDERR:                     ^~~~~~~~~~~~~~~~~~~~~~~
let negative: i32 = LeftShift(1, Negate(1));
