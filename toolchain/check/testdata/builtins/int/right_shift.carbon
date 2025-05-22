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
// CHECK:STDOUT:   %RightShift.type.050: type = fn_type @RightShift.1 [concrete]
// CHECK:STDOUT:   %RightShift: %RightShift.type.050 = struct_value () [concrete]
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
// CHECK:STDOUT: --- u32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %RightShift.type.050: type = fn_type @RightShift.1 [concrete]
// CHECK:STDOUT:   %RightShift: %RightShift.type.050 = struct_value () [concrete]
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
