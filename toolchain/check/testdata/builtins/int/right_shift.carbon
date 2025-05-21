// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/right_shift.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/right_shift.carbon

// --- i32.carbon

library "[[@TEST_NAME]]";

class Expect(N:! i32) {}
fn Test(N:! i32) -> Expect(N) { return {}; }

fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";

fn F() {
  Test(RightShift(0, 31)) as Expect(0);
  Test(RightShift(1, 31)) as Expect(0);
  Test(RightShift(1, 0)) as Expect(1);
  Test(RightShift(1, 2)) as Expect(0);
  Test(RightShift(22, 2)) as Expect(5);
  Test(RightShift(-1, 1)) as Expect(-1);
  Test(RightShift(-2, 1)) as Expect(-1);
  Test(RightShift(-10, 2)) as Expect(-3);
}

fn RuntimeCallIsValid(a: i32, b: i32) -> i32 {
  //@dump-sem-ir-begin
  return RightShift(a, b);
  //@dump-sem-ir-end
}

// --- u32.carbon

library "[[@TEST_NAME]]";

class Expect(N:! u32) {}
fn Test(N:! u32) -> Expect(N) { return {}; }

fn RightShift(a: u32, b: i32) -> u32 = "int.right_shift";

fn F() {
  Test(RightShift(0, 31)) as Expect(0);
  Test(RightShift(1, 31)) as Expect(0);
  Test(RightShift(1, 0)) as Expect(1);
  Test(RightShift(1, 2)) as Expect(0);
  Test(RightShift(22, 2)) as Expect(5);
  Test(RightShift(0xFFFF_FFFF, 1)) as Expect(0x7FFF_FFFF);
  Test(RightShift(0xABCD_EF01, 8)) as Expect(0xAB_CDEF);
}

fn RuntimeCall(a: u32, b: i32) -> u32 {
  //@dump-sem-ir-begin
  return RightShift(a, b);
  //@dump-sem-ir-end
}

// --- literal.carbon

library "[[@TEST_NAME]]";

fn RightShift(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.right_shift";

class Expect(N:! Core.IntLiteral()) {}
fn Test(N:! Core.IntLiteral()) -> Expect(N) { return {}; }

fn F() {
  Test(RightShift(0, 31)) as Expect(0);
  Test(RightShift(1, 31)) as Expect(0);
  Test(RightShift(1, 0)) as Expect(1);
  Test(RightShift(1, 2)) as Expect(0);
  Test(RightShift(22, 2)) as Expect(5);
  Test(RightShift(-1, 1)) as Expect(-1);
  Test(RightShift(-2, 1)) as Expect(-1);
  Test(RightShift(-10, 2)) as Expect(-3);
  Test(RightShift(0xFFFF_FFFF, 1)) as Expect(0x7FFF_FFFF);
  Test(RightShift(0xABCD_EF01, 8)) as Expect(0xAB_CDEF);

  Test(RightShift(0x1234_5678, 1_000_000_000)) as Expect(0);
  Test(RightShift(-0x1234_5678, 1_000_000_000)) as Expect(-1);
  Test(RightShift(0xFFFF_FFFF_FFFF_FFFF, 1_000_000_000)) as Expect(0);
  Test(RightShift(0x7FFF_FFFF_FFFF_FFFF, 1_000_000_000)) as Expect(0);
  Test(RightShift(-0x7FFF_FFFF_FFFF_FFFF, 1_000_000_000)) as Expect(-1);
  Test(RightShift(-0x8000_0000_0000_0000, 1_000_000_000)) as Expect(-1);
}

// --- fail_bad_shift.carbon

library "[[@TEST_NAME]]";

fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
fn RightShiftLit(a: Core.IntLiteral(), b: i32) -> Core.IntLiteral() = "int.right_shift";

// Shift greater than size is disallowed for sized types.
let size_1: i32 = RightShift(1, 31);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:19: error: shift distance >= type width of 32 in `1 >> 32` [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let size_2: i32 = RightShift(1, 32);
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let size_2: i32 = RightShift(1, 32);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:19: error: shift distance >= type width of 32 in `1 >> 33` [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let size_3: i32 = RightShift(1, 33);
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let size_3: i32 = RightShift(1, 33);

// Negative shifts aren't allowed either, even for literals, even if the lhs is zero.
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:21: error: shift distance >= type width of 32 in `1 >> -1` [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let negative: i32 = RightShift(1, -1);
// CHECK:STDERR:                     ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let negative: i32 = RightShift(1, -1);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:26: error: shift distance >= type width of 32 in `0 >> -1` [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let negative_zero: i32 = RightShift(0, -1);
// CHECK:STDERR:                          ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let negative_zero: i32 = RightShift(0, -1);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:39: error: shift distance negative in `1 >> -1` [CompileTimeShiftNegative]
// CHECK:STDERR: let negative_lit: Core.IntLiteral() = RightShiftLit(1, -1);
// CHECK:STDERR:                                       ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let negative_lit: Core.IntLiteral() = RightShiftLit(1, -1);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:44: error: shift distance negative in `0 >> -1` [CompileTimeShiftNegative]
// CHECK:STDERR: let negative_lit_zero: Core.IntLiteral() = RightShiftLit(0, -1);
// CHECK:STDERR:                                            ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let negative_lit_zero: Core.IntLiteral() = RightShiftLit(0, -1);

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
// CHECK:STDOUT:   %RightShift.type.050: type = fn_type @RightShift.1 [concrete]
// CHECK:STDOUT:   %RightShift: %RightShift.type.050 = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Expect.1b5: type = class_type @Expect, @Expect(%int_0.6a9) [concrete]
// CHECK:STDOUT:   %pattern_type.362: type = pattern_type %Expect.1b5 [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Expect.ed1: type = class_type @Expect, @Expect(%int_1.5d2) [concrete]
// CHECK:STDOUT:   %pattern_type.f0b: type = pattern_type %Expect.ed1 [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %Expect.2d6: type = class_type @Expect, @Expect(%int_5.0f6) [concrete]
// CHECK:STDOUT:   %pattern_type.14d: type = pattern_type %Expect.2d6 [concrete]
// CHECK:STDOUT:   %int_-1.251: %i32 = int_value -1 [concrete]
// CHECK:STDOUT:   %Expect.217: type = class_type @Expect, @Expect(%int_-1.251) [concrete]
// CHECK:STDOUT:   %pattern_type.17d: type = pattern_type %Expect.217 [concrete]
// CHECK:STDOUT:   %int_-3.466: %i32 = int_value -3 [concrete]
// CHECK:STDOUT:   %Expect.7da: type = class_type @Expect, @Expect(%int_-3.466) [concrete]
// CHECK:STDOUT:   %pattern_type.01a: type = pattern_type %Expect.7da [concrete]
// CHECK:STDOUT:   %Expect.val.239: %Expect.1b5 = struct_value () [concrete]
// CHECK:STDOUT:   %Expect.val.937: %Expect.ed1 = struct_value () [concrete]
// CHECK:STDOUT:   %Expect.val.4a2: %Expect.2d6 = struct_value () [concrete]
// CHECK:STDOUT:   %Expect.val.301: %Expect.217 = struct_value () [concrete]
// CHECK:STDOUT:   %Expect.val.456: %Expect.7da = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValid(%a.param: %i32, %b.param: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %RightShift.ref: %RightShift.type.050 = name_ref RightShift, file.%RightShift.decl [concrete = constants.%RightShift]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.right_shift: init %i32 = call %RightShift.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc22_26.1: %i32 = value_of_initializer %int.right_shift
// CHECK:STDOUT:   %.loc22_26.2: %i32 = converted %int.right_shift, %.loc22_26.1
// CHECK:STDOUT:   return %.loc22_26.2
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
// CHECK:STDOUT: specific @Test(constants.%int_1.5d2) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_1.5d2
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.ed1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f0b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.937
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_1.5d2) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_1.5d2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_5.0f6) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_5.0f6
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.2d6
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.14d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.4a2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_5.0f6) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_5.0f6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_-1.251) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_-1.251
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.217
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.17d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.301
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-1.251) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_-1.251
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_-3.466) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_-3.466
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.7da
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.01a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.456
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-3.466) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_-3.466
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
// CHECK:STDOUT:   %RightShift.type.050: type = fn_type @RightShift.1 [concrete]
// CHECK:STDOUT:   %RightShift: %RightShift.type.050 = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.7d7: %u32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Expect.eda: type = class_type @Expect, @Expect(%int_0.7d7) [concrete]
// CHECK:STDOUT:   %pattern_type.000: type = pattern_type %Expect.eda [concrete]
// CHECK:STDOUT:   %int_1.c1d: %u32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Expect.637: type = class_type @Expect, @Expect(%int_1.c1d) [concrete]
// CHECK:STDOUT:   %pattern_type.931: type = pattern_type %Expect.637 [concrete]
// CHECK:STDOUT:   %int_5.c46: %u32 = int_value 5 [concrete]
// CHECK:STDOUT:   %Expect.f4a: type = class_type @Expect, @Expect(%int_5.c46) [concrete]
// CHECK:STDOUT:   %pattern_type.ae7: type = pattern_type %Expect.f4a [concrete]
// CHECK:STDOUT:   %int_2147483647.1a3: %u32 = int_value 2147483647 [concrete]
// CHECK:STDOUT:   %Expect.f49: type = class_type @Expect, @Expect(%int_2147483647.1a3) [concrete]
// CHECK:STDOUT:   %pattern_type.611: type = pattern_type %Expect.f49 [concrete]
// CHECK:STDOUT:   %int_11259375.8a6: %u32 = int_value 11259375 [concrete]
// CHECK:STDOUT:   %Expect.a2a: type = class_type @Expect, @Expect(%int_11259375.8a6) [concrete]
// CHECK:STDOUT:   %pattern_type.435: type = pattern_type %Expect.a2a [concrete]
// CHECK:STDOUT:   %Expect.val.3c2: %Expect.eda = struct_value () [concrete]
// CHECK:STDOUT:   %Expect.val.897: %Expect.637 = struct_value () [concrete]
// CHECK:STDOUT:   %Expect.val.d69: %Expect.f4a = struct_value () [concrete]
// CHECK:STDOUT:   %Expect.val.646: %Expect.f49 = struct_value () [concrete]
// CHECK:STDOUT:   %Expect.val.2e0: %Expect.a2a = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param: %u32, %b.param: %i32) -> %u32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %RightShift.ref: %RightShift.type.050 = name_ref RightShift, file.%RightShift.decl [concrete = constants.%RightShift]
// CHECK:STDOUT:   %a.ref: %u32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.right_shift: init %u32 = call %RightShift.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc21_26.1: %u32 = value_of_initializer %int.right_shift
// CHECK:STDOUT:   %.loc21_26.2: %u32 = converted %int.right_shift, %.loc21_26.1
// CHECK:STDOUT:   return %.loc21_26.2
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
// CHECK:STDOUT: specific @Test(constants.%int_1.c1d) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_1.c1d
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.637
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.931
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.897
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_1.c1d) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_1.c1d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_5.c46) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_5.c46
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.f4a
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ae7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.d69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_5.c46) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_5.c46
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_2147483647.1a3) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_2147483647.1a3
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.f49
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.611
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.646
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_2147483647.1a3) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_2147483647.1a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_11259375.8a6) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_11259375.8a6
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.a2a
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.435
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.2e0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_11259375.8a6) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_11259375.8a6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
