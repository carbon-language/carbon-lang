// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/convert_checked.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/convert_checked.carbon

// --- int_ops.carbon

library "[[@TEST_NAME]]";

fn NegateI32(a: i32) -> i32 = "int.snegate";
fn SubI32(a: i32, b: i32) -> i32 = "int.ssub";
fn AddU32(a: u32, b: u32) -> u32 = "int.uadd";
fn IntLiteral() -> type = "int_literal.make_type";

// Size preserving
fn Int32ToInt32(a: i32) -> i32 = "int.convert_checked";
fn Int32ToUint32(a: i32) -> u32 = "int.convert_checked";
fn Uint32ToInt32(a: u32) -> i32 = "int.convert_checked";
fn Uint32ToUint32(a: u32) -> u32 = "int.convert_checked";
fn IntLiteralToIntLiteral(a: IntLiteral()) -> IntLiteral() =
    "int.convert_checked";

// Narrowing
fn Int32ToInt16(a: i32) -> i16 = "int.convert_checked";
fn Int32ToUint16(a: i32) -> u16 = "int.convert_checked";
fn Uint32ToInt16(a: u32) -> i16 = "int.convert_checked";
fn Uint32ToUint16(a: u32) -> u16 = "int.convert_checked";
fn IntLiteralToInt16(a: IntLiteral()) -> i16 = "int.convert_checked";
fn IntLiteralToUint16(a: IntLiteral()) -> u16 = "int.convert_checked";

// Widening
fn Int32ToInt64(a: i32) -> i64 = "int.convert_checked";
fn Int32ToUint64(a: i32) -> u64 = "int.convert_checked";
fn Uint32ToInt64(a: u32) -> i64 = "int.convert_checked";
fn Uint32ToUint64(a: u32) -> u64 = "int.convert_checked";
fn Int32ToIntLiteral(a: i32) -> IntLiteral() = "int.convert_checked";
fn Uint32ToUintLiteral(a: u32) -> IntLiteral() = "int.convert_checked";

// --- identity.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

let a: i32 = Int32ToInt32(0);
let b: i32 = Int32ToInt32(0x7FFF_FFFF);
let c: i32 = Int32ToInt32(SubI32(NegateI32(0x7FFF_FFFF), 1));
let d: IntLiteral() = IntLiteralToIntLiteral(Int32ToIntLiteral(NegateI32(1)));

// --- same_size.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

let max: u32 = Int32ToUint32(0x7FFF_FFFF);
let max_roundtrip: i32 = Uint32ToInt32(Int32ToUint32(0x7FFF_FFFF));

// --- truncate.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

let a: u16 = Int32ToUint16(0);
let b: u16 = Int32ToUint16(0xFFFF);

let c: i16 = Int32ToInt16(0x7FFF);
let d: i16 = Int32ToInt16(NegateI32(0x8000));

let e: u16 = Uint32ToUint16(Int32ToUint32(0));
let f: u16 = Uint32ToUint16(Int32ToUint32(0xFFFF));

let g: i16 = Uint32ToInt16(Int32ToUint32(0));
let h: i16 = Uint32ToInt16(Int32ToUint32(0x7FFF));

let lit_i16_min: i16 = IntLiteralToInt16(Int32ToIntLiteral(NegateI32(0x8000)));
let lit_i16_max: i16 = IntLiteralToInt16(Int32ToIntLiteral(0x7FFF));

let lit_u16_min: u16 = IntLiteralToUint16(Int32ToIntLiteral(0));
let lit_u16_max: u16 = IntLiteralToUint16(Int32ToIntLiteral(0xFFFF));

// --- zero_extend.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

let a: u64 = Uint32ToUint64(Int32ToUint32(0));
let b: u64 = Uint32ToUint64(
  AddU32(
    AddU32(Int32ToUint32(0x7FFF_FFFF), Int32ToUint32(0x7FFF_FFFF)),
    Int32ToUint32(1)));

let c: i64 = Uint32ToInt64(Int32ToUint32(0));
let d: i64 = Uint32ToInt64(
  AddU32(
    AddU32(Int32ToUint32(0x7FFF_FFFF), Int32ToUint32(0x7FFF_FFFF)),
    Int32ToUint32(1)));

// --- sign_extend.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

let a: u64 = Int32ToUint64(0);
let b: u64 = Int32ToUint64(0x7FFF_FFFF);

let c: i64 = Int32ToInt64(SubI32(NegateI32(0x7FFF_FFFF), 1));
let d: i64 = Int32ToInt64(0x7FFF_FFFF);

// --- fail_too_large_u32_for_i32.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

let max_plus_one: i32 =
  // CHECK:STDERR: fail_too_large_u32_for_i32.carbon:[[@LINE+4]]:3: error: integer value 2147483648 too large for type `i32` [IntTooLargeForType]
  // CHECK:STDERR:   Uint32ToInt32(
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  Uint32ToInt32(
    AddU32(Int32ToUint32(0x7FFF_FFFF),
           Int32ToUint32(1)));

// --- fail_too_large_i32_for_i16.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: fail_too_large_i32_for_i16.carbon:[[@LINE+4]]:25: error: integer value 32768 too large for type `i16` [IntTooLargeForType]
// CHECK:STDERR: let max_plus_one: i16 = Int32ToInt16(0x8000);
// CHECK:STDERR:                         ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let max_plus_one: i16 = Int32ToInt16(0x8000);

// --- fail_too_large_i32_for_u16.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: fail_too_large_i32_for_u16.carbon:[[@LINE+4]]:25: error: integer value 65536 too large for type `u16` [IntTooLargeForType]
// CHECK:STDERR: let max_plus_one: u16 = Int32ToUint16(0x1_0000);
// CHECK:STDERR:                         ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let max_plus_one: u16 = Int32ToUint16(0x1_0000);

// --- fail_too_large_u32_for_i16.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: fail_too_large_u32_for_i16.carbon:[[@LINE+4]]:25: error: integer value 32768 too large for type `i16` [IntTooLargeForType]
// CHECK:STDERR: let max_plus_one: i16 = Uint32ToInt16(Int32ToUint32(0x8000));
// CHECK:STDERR:                         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let max_plus_one: i16 = Uint32ToInt16(Int32ToUint32(0x8000));

// --- fail_too_large_u32_for_u16.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: fail_too_large_u32_for_u16.carbon:[[@LINE+4]]:25: error: integer value 65536 too large for type `u16` [IntTooLargeForType]
// CHECK:STDERR: let max_plus_one: u16 = Uint32ToUint16(Int32ToUint32(0x1_0000));
// CHECK:STDERR:                         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let max_plus_one: u16 = Uint32ToUint16(Int32ToUint32(0x1_0000));

// --- fail_negative_i32_to_u16.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: fail_negative_i32_to_u16.carbon:[[@LINE+4]]:29: error: negative integer value -1 converted to unsigned type `u16` [NegativeIntInUnsignedType]
// CHECK:STDERR: let minus_one_to_u16: u16 = Int32ToUint16(SubI32(0, 1));
// CHECK:STDERR:                             ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let minus_one_to_u16: u16 = Int32ToUint16(SubI32(0, 1));

// --- fail_negative_i32_to_u32.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: fail_negative_i32_to_u32.carbon:[[@LINE+4]]:29: error: negative integer value -1 converted to unsigned type `u32` [NegativeIntInUnsignedType]
// CHECK:STDERR: let minus_one_to_u32: u32 = Int32ToUint32(SubI32(0, 1));
// CHECK:STDERR:                             ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let minus_one_to_u32: u32 = Int32ToUint32(SubI32(0, 1));

// --- fail_negative_i32_to_u64.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: fail_negative_i32_to_u64.carbon:[[@LINE+4]]:29: error: negative integer value -1 converted to unsigned type `u64` [NegativeIntInUnsignedType]
// CHECK:STDERR: let minus_one_to_u64: u64 = Int32ToUint64(SubI32(0, 1));
// CHECK:STDERR:                             ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let minus_one_to_u64: u64 = Int32ToUint64(SubI32(0, 1));

// --- fail_too_small_i32_for_i16.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: fail_too_small_i32_for_i16.carbon:[[@LINE+4]]:26: error: integer value -32769 too large for type `i16` [IntTooLargeForType]
// CHECK:STDERR: let min_minus_one: i16 = Int32ToInt16(NegateI32(0x8001));
// CHECK:STDERR:                          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let min_minus_one: i16 = Int32ToInt16(NegateI32(0x8001));

// --- fail_not_constant.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

let not_constant: i32 = 0;

// CHECK:STDERR: fail_not_constant.carbon:[[@LINE+8]]:40: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let convert_not_constant_narrow: i16 = Int32ToInt16(not_constant);
// CHECK:STDERR:                                        ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_not_constant.carbon:[[@LINE-7]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:18:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn Int32ToInt16(a: i32) -> i16 = "int.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let convert_not_constant_narrow: i16 = Int32ToInt16(not_constant);

// CHECK:STDERR: fail_not_constant.carbon:[[@LINE+8]]:38: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let convert_not_constant_same: i32 = Int32ToInt32(not_constant);
// CHECK:STDERR:                                      ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_not_constant.carbon:[[@LINE-17]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:10:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn Int32ToInt32(a: i32) -> i32 = "int.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let convert_not_constant_same: i32 = Int32ToInt32(not_constant);

// CHECK:STDERR: fail_not_constant.carbon:[[@LINE+8]]:39: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let convert_not_constant_widen: i64 = Int32ToInt64(not_constant);
// CHECK:STDERR:                                       ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_not_constant.carbon:[[@LINE-27]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:26:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn Int32ToInt64(a: i32) -> i64 = "int.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let convert_not_constant_widen: i64 = Int32ToInt64(not_constant);
