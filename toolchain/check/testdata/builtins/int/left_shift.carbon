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

// --- i32.carbon

library "[[@TEST_NAME]]";

class Expect(N:! i32) {}
fn Test(N:! i32) -> Expect(N) { return {}; }

fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";

fn F() {
  Test(LeftShift(0, 0)) as Expect(0);
  Test(LeftShift(0, 1)) as Expect(0);
  Test(LeftShift(0, 30)) as Expect(0);
  Test(LeftShift(1, 30)) as Expect(0x4000_0000);
  Test(LeftShift(5, 2)) as Expect(20);
  Test(LeftShift(-1, 1)) as Expect(-2);
  Test(LeftShift(-2, 1)) as Expect(-4);
  Test(LeftShift(-3, 1)) as Expect(-6);
}

fn RuntimeCallIsValid(a: i32, b: i32) -> i32 {
  return LeftShift(a, b);
}

// --- u32.carbon

library "[[@TEST_NAME]]";

class Expect(N:! u32) {}
fn Test(N:! u32) -> Expect(N) { return {}; }

fn LeftShift(a: u32, b: i32) -> u32 = "int.left_shift";

fn F() {
  Test(LeftShift(0, 0)) as Expect(0);
  Test(LeftShift(0, 1)) as Expect(0);
  Test(LeftShift(0, 30)) as Expect(0);
  Test(LeftShift(1, 30)) as Expect(0x4000_0000);
  Test(LeftShift(5, 2)) as Expect(20);
  Test(LeftShift(0xFFFF_FFFF, 1)) as Expect(0xFFFF_FFFE);
  Test(LeftShift(0xFFFF_FFFE, 1)) as Expect(0xFFFF_FFFC);
  Test(LeftShift(0xABCD_EF01, 8)) as Expect(0xCDEF_0100);
}

fn RuntimeCall(a: u32, b: i32) -> u32 {
  return LeftShift(a, b);
}

// --- literal.carbon

library "[[@TEST_NAME]]";

fn LeftShift(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.left_shift";

class Expect(N:! Core.IntLiteral()) {}
fn Test(N:! Core.IntLiteral()) -> Expect(N) { return {}; }

fn F() {
  // Zero can be shifted by any amount.
  Test(LeftShift(0, 0)) as Expect(0);
  Test(LeftShift(0, 0)) as Expect(0);
  Test(LeftShift(0, 1)) as Expect(0);
  Test(LeftShift(0, 30)) as Expect(0);
  Test(LeftShift(0, 1_000_000_000)) as Expect(0);

  // Positive numbers can be shifted.
  Test(LeftShift(1, 0)) as Expect(1);
  Test(LeftShift(1, 1)) as Expect(2);
  Test(LeftShift(2, 1)) as Expect(4);
  Test(LeftShift(1, 2)) as Expect(4);
  Test(LeftShift(3, 2)) as Expect(12);
  Test(LeftShift(1, 30)) as Expect(0x4000_0000);
  Test(LeftShift(5, 2)) as Expect(20);

  // Negative numbers can be shifted too.
  Test(LeftShift(-1, 0)) as Expect(-1);
  Test(LeftShift(-1, 1)) as Expect(-2);
  Test(LeftShift(-2, 1)) as Expect(-4);
  Test(LeftShift(-3, 1)) as Expect(-6);

  // Large numbers can be shifted losslessly.
  Test(LeftShift(0xFFFF_FFFF, 1)) as Expect(0x1_FFFF_FFFE);
  Test(LeftShift(0xFFFF_FFFE, 1)) as Expect(0x1_FFFF_FFFC);
  Test(LeftShift(0xABCD_EF01, 8)) as Expect(0xAB_CDEF_0100);
  Test(LeftShift(0x7FFF_FFFF_FFFF_FFFF, 1)) as Expect(0xFFFF_FFFF_FFFF_FFFE);
  Test(LeftShift(0xFFFF_FFFF_FFFF_FFFF, 1)) as Expect(0x1_FFFF_FFFF_FFFF_FFFE);
}

// --- fail_bad_shift.carbon

library "[[@TEST_NAME]]";

fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
fn LeftShiftLit(a: Core.IntLiteral(), b: i32) -> Core.IntLiteral() = "int.left_shift";

// Shift greater than size is disallowed for sized types.
let size_1: i32 = LeftShift(1, 31);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:19: error: shift distance >= type width of 32 in `1 << 32` [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let size_2: i32 = LeftShift(1, 32);
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
let size_2: i32 = LeftShift(1, 32);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:19: error: shift distance >= type width of 32 in `1 << 33` [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let size_3: i32 = LeftShift(1, 33);
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
let size_3: i32 = LeftShift(1, 33);

// Overflow is allowed if the shift distance is in bounds.
let overflow_1: i32 = LeftShift(1000, 31);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:23: error: shift distance >= type width of 32 in `1000 << 32` [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let overflow_2: i32 = LeftShift(1000, 32);
// CHECK:STDERR:                       ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let overflow_2: i32 = LeftShift(1000, 32);

// Oversize shifts aren't allowed even if there's no overflow.
let no_overflow_1: i32 = LeftShift(0, 31);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:26: error: shift distance >= type width of 32 in `0 << 32` [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let no_overflow_2: i32 = LeftShift(0, 32);
// CHECK:STDERR:                          ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
let no_overflow_2: i32 = LeftShift(0, 32);

// Negative shifts aren't allowed either, even for literals, even if the lhs is zero.
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:21: error: shift distance >= type width of 32 in `1 << -1` [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let negative: i32 = LeftShift(1, -1);
// CHECK:STDERR:                     ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
let negative: i32 = LeftShift(1, -1);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:26: error: shift distance >= type width of 32 in `0 << -1` [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let negative_zero: i32 = LeftShift(0, -1);
// CHECK:STDERR:                          ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
let negative_zero: i32 = LeftShift(0, -1);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:39: error: shift distance negative in `1 << -1` [CompileTimeShiftNegative]
// CHECK:STDERR: let negative_lit: Core.IntLiteral() = LeftShiftLit(1, -1);
// CHECK:STDERR:                                       ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let negative_lit: Core.IntLiteral() = LeftShiftLit(1, -1);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:44: error: shift distance negative in `0 << -1` [CompileTimeShiftNegative]
// CHECK:STDERR: let negative_lit_zero: Core.IntLiteral() = LeftShiftLit(0, -1);
// CHECK:STDERR:                                            ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let negative_lit_zero: Core.IntLiteral() = LeftShiftLit(0, -1);

// --- fail_literal_overflow.carbon

library "[[@TEST_NAME]]";

fn LeftShift(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.left_shift";

// CHECK:STDERR: fail_literal_overflow.carbon:[[@LINE+4]]:16: error: shift distance of 1000000000 would result in an integer whose width is greater than the maximum supported width of 8388608 [CompileTimeUnsizedShiftOutOfRange]
// CHECK:STDERR: let bad: i32 = LeftShift(1, 1_000_000_000);
// CHECK:STDERR:                ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let bad: i32 = LeftShift(1, 1_000_000_000);

// CHECK:STDERR: fail_literal_overflow.carbon:[[@LINE+4]]:25: error: shift distance of 1000000000 would result in an integer whose width is greater than the maximum supported width of 8388608 [CompileTimeUnsizedShiftOutOfRange]
// CHECK:STDERR: let bad_negative: i32 = LeftShift(-1, 1_000_000_000);
// CHECK:STDERR:                         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let bad_negative: i32 = LeftShift(-1, 1_000_000_000);

// --- fail_comp_time_only_shift.carbon

library "[[@TEST_NAME]]";

fn LeftShiftByLit(a: i32, b: Core.IntLiteral()) -> i32 = "int.left_shift";
fn LeftShiftOfLit(a: Core.IntLiteral(), b: i32) -> Core.IntLiteral() = "int.left_shift";

var a_lit: Core.IntLiteral() = 12;
var an_i32: i32 = 34;

// This can't be valid: we don't have a compile-time or runtime integer value for `a_lit`.
// CHECK:STDERR: fail_comp_time_only_shift.carbon:[[@LINE+7]]:17: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let bad1: i32 = LeftShiftByLit(an_i32, a_lit);
// CHECK:STDERR:                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_comp_time_only_shift.carbon:[[@LINE-10]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn LeftShiftByLit(a: i32, b: Core.IntLiteral()) -> i32 = "int.left_shift";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let bad1: i32 = LeftShiftByLit(an_i32, a_lit);

// TODO: This could be valid because we don't actually need the return value at runtime.
// CHECK:STDERR: fail_comp_time_only_shift.carbon:[[@LINE+7]]:31: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let bad2: Core.IntLiteral() = LeftShiftOfLit(a_lit, an_i32);
// CHECK:STDERR:                               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_comp_time_only_shift.carbon:[[@LINE-19]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn LeftShiftOfLit(a: Core.IntLiteral(), b: i32) -> Core.IntLiteral() = "int.left_shift";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let bad2: Core.IntLiteral() = LeftShiftOfLit(a_lit, an_i32);

// TODO: This could be valid because the literal argument has a constant value.
// CHECK:STDERR: fail_comp_time_only_shift.carbon:[[@LINE+7]]:17: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let bad3: i32 = LeftShiftByLit(an_i32, 12);
// CHECK:STDERR:                 ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_comp_time_only_shift.carbon:[[@LINE-30]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn LeftShiftByLit(a: i32, b: Core.IntLiteral()) -> i32 = "int.left_shift";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let bad3: i32 = LeftShiftByLit(an_i32, 12);

// TODO: This could be valid because we don't actually need the return value at runtime.
// CHECK:STDERR: fail_comp_time_only_shift.carbon:[[@LINE+7]]:31: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let bad4: Core.IntLiteral() = LeftShiftOfLit(12, an_i32);
// CHECK:STDERR:                               ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_comp_time_only_shift.carbon:[[@LINE-39]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn LeftShiftOfLit(a: Core.IntLiteral(), b: i32) -> Core.IntLiteral() = "int.left_shift";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let bad4: Core.IntLiteral() = LeftShiftOfLit(12, an_i32);
