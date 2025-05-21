// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
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
  //@dump-sem-ir-begin
  return LeftShift(a, b);
  //@dump-sem-ir-end
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
  //@dump-sem-ir-begin
  return LeftShift(a, b);
  //@dump-sem-ir-end
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

// CHECK:STDOUT: --- i32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %N.51e: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %Expect.185: type = class_type @Expect, @Expect(%N.51e) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.a0f: type = pattern_type %Expect.185 [symbolic]
// CHECK:STDOUT:   %LeftShift.type.ed3: type = fn_type @LeftShift.1 [concrete]
// CHECK:STDOUT:   %LeftShift: %LeftShift.type.ed3 = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Expect.1b5: type = class_type @Expect, @Expect(%int_0.6a9) [concrete]
// CHECK:STDOUT:   %pattern_type.362: type = pattern_type %Expect.1b5 [concrete]
// CHECK:STDOUT:   %int_1073741824.175: %i32 = int_value 1073741824 [concrete]
// CHECK:STDOUT:   %Expect.c60: type = class_type @Expect, @Expect(%int_1073741824.175) [concrete]
// CHECK:STDOUT:   %pattern_type.102: type = pattern_type %Expect.c60 [concrete]
// CHECK:STDOUT:   %int_20.37c: %i32 = int_value 20 [concrete]
// CHECK:STDOUT:   %Expect.240: type = class_type @Expect, @Expect(%int_20.37c) [concrete]
// CHECK:STDOUT:   %pattern_type.136: type = pattern_type %Expect.240 [concrete]
// CHECK:STDOUT:   %int_-2.321: %i32 = int_value -2 [concrete]
// CHECK:STDOUT:   %Expect.675: type = class_type @Expect, @Expect(%int_-2.321) [concrete]
// CHECK:STDOUT:   %pattern_type.c8e: type = pattern_type %Expect.675 [concrete]
// CHECK:STDOUT:   %int_-4.e56: %i32 = int_value -4 [concrete]
// CHECK:STDOUT:   %Expect.e6d: type = class_type @Expect, @Expect(%int_-4.e56) [concrete]
// CHECK:STDOUT:   %pattern_type.f4d: type = pattern_type %Expect.e6d [concrete]
// CHECK:STDOUT:   %int_-6.70c: %i32 = int_value -6 [concrete]
// CHECK:STDOUT:   %Expect.e9d: type = class_type @Expect, @Expect(%int_-6.70c) [concrete]
// CHECK:STDOUT:   %pattern_type.e8a: type = pattern_type %Expect.e9d [concrete]
// CHECK:STDOUT:   %Expect.val.239: %Expect.1b5 = struct_value () [concrete]
// CHECK:STDOUT:   %Expect.val.ae6: %Expect.c60 = struct_value () [concrete]
// CHECK:STDOUT:   %Expect.val.fa0: %Expect.240 = struct_value () [concrete]
// CHECK:STDOUT:   %Expect.val.a9e: %Expect.675 = struct_value () [concrete]
// CHECK:STDOUT:   %Expect.val.ebf: %Expect.e6d = struct_value () [concrete]
// CHECK:STDOUT:   %Expect.val.0b5: %Expect.e9d = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValid(%a.param: %i32, %b.param: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %LeftShift.ref: %LeftShift.type.ed3 = name_ref LeftShift, file.%LeftShift.decl [concrete = constants.%LeftShift]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.left_shift: init %i32 = call %LeftShift.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc22_25.1: %i32 = value_of_initializer %int.left_shift
// CHECK:STDOUT:   %.loc22_25.2: %i32 = converted %int.left_shift, %.loc22_25.1
// CHECK:STDOUT:   return %.loc22_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%N.51e) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%N.51e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%N.51e) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%N.51e
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.185
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a0f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_0.6a9) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_0.6a9
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.1b5
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.362
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.239
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_0.6a9) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_0.6a9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_1073741824.175) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_1073741824.175
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.c60
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.102
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.ae6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_1073741824.175) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_1073741824.175
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_20.37c) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_20.37c
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.240
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.136
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.fa0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_20.37c) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_20.37c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_-2.321) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_-2.321
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.675
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c8e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.a9e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-2.321) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_-2.321
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_-4.e56) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_-4.e56
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.e6d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f4d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.ebf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-4.e56) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_-4.e56
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_-6.70c) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_-6.70c
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.e9d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e8a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.0b5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-6.70c) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_-6.70c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- u32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %N.1ff: %u32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %Expect.c91: type = class_type @Expect, @Expect(%N.1ff) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.f4c: type = pattern_type %Expect.c91 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %LeftShift.type.ed3: type = fn_type @LeftShift.1 [concrete]
// CHECK:STDOUT:   %LeftShift: %LeftShift.type.ed3 = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.7d7: %u32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Expect.eda: type = class_type @Expect, @Expect(%int_0.7d7) [concrete]
// CHECK:STDOUT:   %pattern_type.000: type = pattern_type %Expect.eda [concrete]
// CHECK:STDOUT:   %int_1073741824.d0a: %u32 = int_value 1073741824 [concrete]
// CHECK:STDOUT:   %Expect.13f: type = class_type @Expect, @Expect(%int_1073741824.d0a) [concrete]
// CHECK:STDOUT:   %pattern_type.e46: type = pattern_type %Expect.13f [concrete]
// CHECK:STDOUT:   %int_20.50c: %u32 = int_value 20 [concrete]
// CHECK:STDOUT:   %Expect.3c2: type = class_type @Expect, @Expect(%int_20.50c) [concrete]
// CHECK:STDOUT:   %pattern_type.2eb: type = pattern_type %Expect.3c2 [concrete]
// CHECK:STDOUT:   %int_4294967294.161: %u32 = int_value 4294967294 [concrete]
// CHECK:STDOUT:   %Expect.c4f: type = class_type @Expect, @Expect(%int_4294967294.161) [concrete]
// CHECK:STDOUT:   %pattern_type.837: type = pattern_type %Expect.c4f [concrete]
// CHECK:STDOUT:   %int_4294967292.f53: %u32 = int_value 4294967292 [concrete]
// CHECK:STDOUT:   %Expect.b08: type = class_type @Expect, @Expect(%int_4294967292.f53) [concrete]
// CHECK:STDOUT:   %pattern_type.80b: type = pattern_type %Expect.b08 [concrete]
// CHECK:STDOUT:   %int_3454992640.e56: %u32 = int_value 3454992640 [concrete]
// CHECK:STDOUT:   %Expect.155: type = class_type @Expect, @Expect(%int_3454992640.e56) [concrete]
// CHECK:STDOUT:   %pattern_type.440: type = pattern_type %Expect.155 [concrete]
// CHECK:STDOUT:   %Expect.val.3c2: %Expect.eda = struct_value () [concrete]
// CHECK:STDOUT:   %Expect.val.dc3: %Expect.13f = struct_value () [concrete]
// CHECK:STDOUT:   %Expect.val.f04: %Expect.3c2 = struct_value () [concrete]
// CHECK:STDOUT:   %Expect.val.de9: %Expect.c4f = struct_value () [concrete]
// CHECK:STDOUT:   %Expect.val.8f8: %Expect.b08 = struct_value () [concrete]
// CHECK:STDOUT:   %Expect.val.db6: %Expect.155 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param: %u32, %b.param: %i32) -> %u32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %LeftShift.ref: %LeftShift.type.ed3 = name_ref LeftShift, file.%LeftShift.decl [concrete = constants.%LeftShift]
// CHECK:STDOUT:   %a.ref: %u32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.left_shift: init %u32 = call %LeftShift.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc22_25.1: %u32 = value_of_initializer %int.left_shift
// CHECK:STDOUT:   %.loc22_25.2: %u32 = converted %int.left_shift, %.loc22_25.1
// CHECK:STDOUT:   return %.loc22_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%N.1ff) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%N.1ff
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%N.1ff) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%N.1ff
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.c91
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f4c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_0.7d7) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_0.7d7
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.eda
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.000
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.3c2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_0.7d7) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_0.7d7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_1073741824.d0a) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_1073741824.d0a
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.13f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e46
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.dc3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_1073741824.d0a) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_1073741824.d0a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_20.50c) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_20.50c
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.3c2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2eb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.f04
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_20.50c) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_20.50c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_4294967294.161) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_4294967294.161
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.c4f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.837
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.de9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_4294967294.161) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_4294967294.161
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_4294967292.f53) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_4294967292.f53
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.b08
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.80b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.8f8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_4294967292.f53) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_4294967292.f53
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_3454992640.e56) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_3454992640.e56
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.155
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.440
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.db6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_3454992640.e56) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_3454992640.e56
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
