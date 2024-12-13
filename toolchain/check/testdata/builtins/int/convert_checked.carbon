// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDERR: identity.carbon:[[@LINE+8]]:14: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let c: i32 = Int32ToInt32(SubI32(NegateI32(0x7FFF_FFFF), 1));
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: identity.carbon:[[@LINE-7]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:10:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn Int32ToInt32(a: i32) -> i32 = "int.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let c: i32 = Int32ToInt32(SubI32(NegateI32(0x7FFF_FFFF), 1));
// CHECK:STDERR: identity.carbon:[[@LINE+16]]:23: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let d: IntLiteral() = IntLiteralToIntLiteral(Int32ToIntLiteral(NegateI32(1)));
// CHECK:STDERR:                       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: identity.carbon:[[@LINE-16]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:14:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn IntLiteralToIntLiteral(a: IntLiteral()) -> IntLiteral() =
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: identity.carbon:[[@LINE+8]]:46: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let d: IntLiteral() = IntLiteralToIntLiteral(Int32ToIntLiteral(NegateI32(1)));
// CHECK:STDERR:                                              ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: identity.carbon:[[@LINE-24]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:30:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn Int32ToIntLiteral(a: i32) -> IntLiteral() = "int.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
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
// CHECK:STDERR: truncate.carbon:[[@LINE+8]]:14: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let d: i16 = Int32ToInt16(NegateI32(0x8000));
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: truncate.carbon:[[@LINE-9]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:18:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn Int32ToInt16(a: i32) -> i16 = "int.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let d: i16 = Int32ToInt16(NegateI32(0x8000));

let e: u16 = Uint32ToUint16(Int32ToUint32(0));
let f: u16 = Uint32ToUint16(Int32ToUint32(0xFFFF));

let g: i16 = Uint32ToInt16(Int32ToUint32(0));
let h: i16 = Uint32ToInt16(Int32ToUint32(0x7FFF));

// CHECK:STDERR: truncate.carbon:[[@LINE+16]]:24: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let lit_i16_min: i16 = IntLiteralToInt16(Int32ToIntLiteral(NegateI32(0x8000)));
// CHECK:STDERR:                        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: truncate.carbon:[[@LINE-25]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:22:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn IntLiteralToInt16(a: IntLiteral()) -> i16 = "int.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: truncate.carbon:[[@LINE+8]]:42: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let lit_i16_min: i16 = IntLiteralToInt16(Int32ToIntLiteral(NegateI32(0x8000)));
// CHECK:STDERR:                                          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: truncate.carbon:[[@LINE-33]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:30:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn Int32ToIntLiteral(a: i32) -> IntLiteral() = "int.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let lit_i16_min: i16 = IntLiteralToInt16(Int32ToIntLiteral(NegateI32(0x8000)));
let lit_i16_max: i16 = IntLiteralToInt16(Int32ToIntLiteral(0x7FFF));

let lit_u16_min: u16 = IntLiteralToUint16(Int32ToIntLiteral(0));
let lit_u16_max: u16 = IntLiteralToUint16(Int32ToIntLiteral(0xFFFF));

// --- zero_extend.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

let a: u64 = Uint32ToUint64(Int32ToUint32(0));
// CHECK:STDERR: zero_extend.carbon:[[@LINE+8]]:14: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let b: u64 = Uint32ToUint64(
// CHECK:STDERR:              ^~~~~~~~~~~~~~~
// CHECK:STDERR: zero_extend.carbon:[[@LINE-6]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:29:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn Uint32ToUint64(a: u32) -> u64 = "int.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let b: u64 = Uint32ToUint64(
  AddU32(
    AddU32(Int32ToUint32(0x7FFF_FFFF), Int32ToUint32(0x7FFF_FFFF)),
    Int32ToUint32(1)));

let c: i64 = Uint32ToInt64(Int32ToUint32(0));
// CHECK:STDERR: zero_extend.carbon:[[@LINE+8]]:14: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let d: i64 = Uint32ToInt64(
// CHECK:STDERR:              ^~~~~~~~~~~~~~
// CHECK:STDERR: zero_extend.carbon:[[@LINE-20]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:28:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn Uint32ToInt64(a: u32) -> i64 = "int.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let d: i64 = Uint32ToInt64(
  AddU32(
    AddU32(Int32ToUint32(0x7FFF_FFFF), Int32ToUint32(0x7FFF_FFFF)),
    Int32ToUint32(1)));

// --- sign_extend.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

let a: u64 = Int32ToUint64(0);
let b: u64 = Int32ToUint64(0x7FFF_FFFF);

// CHECK:STDERR: sign_extend.carbon:[[@LINE+8]]:14: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let c: i64 = Int32ToInt64(SubI32(NegateI32(0x7FFF_FFFF), 1));
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: sign_extend.carbon:[[@LINE-8]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:26:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn Int32ToInt64(a: i32) -> i64 = "int.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let c: i64 = Int32ToInt64(SubI32(NegateI32(0x7FFF_FFFF), 1));
let d: i64 = Int32ToInt64(0x7FFF_FFFF);

// --- fail_too_large_u32_for_i32.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

let max_plus_one: i32 =
  // CHECK:STDERR: fail_too_large_u32_for_i32.carbon:[[@LINE+8]]:3: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   Uint32ToInt32(
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_too_large_u32_for_i32.carbon:[[@LINE-6]]:1: in import [InImport]
  // CHECK:STDERR: int_ops.carbon:12:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR: fn Uint32ToInt32(a: u32) -> i32 = "int.convert_checked";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Uint32ToInt32(
    AddU32(Int32ToUint32(0x7FFF_FFFF),
           Int32ToUint32(1)));

// --- fail_too_large_i32_for_i16.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

let max_plus_one: i16 = Int32ToInt16(0x8000);

// --- fail_too_large_i32_for_u16.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

let max_plus_one: u16 = Int32ToUint16(0x1_0000);

// --- fail_too_large_u32_for_i16.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

let max_plus_one: i16 = Uint32ToInt16(Int32ToUint32(0x8000));

// --- fail_too_large_u32_for_u16.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

let max_plus_one: u16 = Uint32ToUint16(Int32ToUint32(0x1_0000));

// --- fail_negative_i32_to_u16.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: fail_negative_i32_to_u16.carbon:[[@LINE+8]]:29: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let minus_one_to_u16: u16 = Int32ToUint16(SubI32(0, 1));
// CHECK:STDERR:                             ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_negative_i32_to_u16.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:19:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn Int32ToUint16(a: i32) -> u16 = "int.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let minus_one_to_u16: u16 = Int32ToUint16(SubI32(0, 1));

// --- fail_negative_i32_to_u32.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: fail_negative_i32_to_u32.carbon:[[@LINE+8]]:29: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let minus_one_to_u32: u32 = Int32ToUint32(SubI32(0, 1));
// CHECK:STDERR:                             ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_negative_i32_to_u32.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:11:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn Int32ToUint32(a: i32) -> u32 = "int.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let minus_one_to_u32: u32 = Int32ToUint32(SubI32(0, 1));

// --- fail_negative_i32_to_u64.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: fail_negative_i32_to_u64.carbon:[[@LINE+8]]:29: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let minus_one_to_u64: u64 = Int32ToUint64(SubI32(0, 1));
// CHECK:STDERR:                             ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_negative_i32_to_u64.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:27:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn Int32ToUint64(a: i32) -> u64 = "int.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let minus_one_to_u64: u64 = Int32ToUint64(SubI32(0, 1));

// --- fail_too_small_i32_for_i16.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: fail_too_small_i32_for_i16.carbon:[[@LINE+8]]:26: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let min_minus_one: i16 = Int32ToInt16(NegateI32(0x8001));
// CHECK:STDERR:                          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_too_small_i32_for_i16.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:18:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn Int32ToInt16(a: i32) -> i16 = "int.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

// CHECK:STDERR: fail_not_constant.carbon:[[@LINE+7]]:39: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let convert_not_constant_widen: i64 = Int32ToInt64(not_constant);
// CHECK:STDERR:                                       ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_not_constant.carbon:[[@LINE-27]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:26:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn Int32ToInt64(a: i32) -> i64 = "int.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
let convert_not_constant_widen: i64 = Int32ToInt64(not_constant);

// CHECK:STDOUT: --- int_ops.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %NegateI32.type: type = fn_type @NegateI32 [template]
// CHECK:STDOUT:   %NegateI32: %NegateI32.type = struct_value () [template]
// CHECK:STDOUT:   %SubI32.type: type = fn_type @SubI32 [template]
// CHECK:STDOUT:   %SubI32: %SubI32.type = struct_value () [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %u32: type = int_type unsigned, %int_32 [template]
// CHECK:STDOUT:   %AddU32.type: type = fn_type @AddU32 [template]
// CHECK:STDOUT:   %AddU32: %AddU32.type = struct_value () [template]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %Int32ToInt32.type: type = fn_type @Int32ToInt32 [template]
// CHECK:STDOUT:   %Int32ToInt32: %Int32ToInt32.type = struct_value () [template]
// CHECK:STDOUT:   %Int32ToUint32.type: type = fn_type @Int32ToUint32 [template]
// CHECK:STDOUT:   %Int32ToUint32: %Int32ToUint32.type = struct_value () [template]
// CHECK:STDOUT:   %Uint32ToInt32.type: type = fn_type @Uint32ToInt32 [template]
// CHECK:STDOUT:   %Uint32ToInt32: %Uint32ToInt32.type = struct_value () [template]
// CHECK:STDOUT:   %Uint32ToUint32.type: type = fn_type @Uint32ToUint32 [template]
// CHECK:STDOUT:   %Uint32ToUint32: %Uint32ToUint32.type = struct_value () [template]
// CHECK:STDOUT:   %IntLiteralToIntLiteral.type: type = fn_type @IntLiteralToIntLiteral [template]
// CHECK:STDOUT:   %IntLiteralToIntLiteral: %IntLiteralToIntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [template]
// CHECK:STDOUT:   %i16: type = int_type signed, %int_16 [template]
// CHECK:STDOUT:   %Int32ToInt16.type: type = fn_type @Int32ToInt16 [template]
// CHECK:STDOUT:   %Int32ToInt16: %Int32ToInt16.type = struct_value () [template]
// CHECK:STDOUT:   %u16: type = int_type unsigned, %int_16 [template]
// CHECK:STDOUT:   %Int32ToUint16.type: type = fn_type @Int32ToUint16 [template]
// CHECK:STDOUT:   %Int32ToUint16: %Int32ToUint16.type = struct_value () [template]
// CHECK:STDOUT:   %Uint32ToInt16.type: type = fn_type @Uint32ToInt16 [template]
// CHECK:STDOUT:   %Uint32ToInt16: %Uint32ToInt16.type = struct_value () [template]
// CHECK:STDOUT:   %Uint32ToUint16.type: type = fn_type @Uint32ToUint16 [template]
// CHECK:STDOUT:   %Uint32ToUint16: %Uint32ToUint16.type = struct_value () [template]
// CHECK:STDOUT:   %IntLiteralToInt16.type: type = fn_type @IntLiteralToInt16 [template]
// CHECK:STDOUT:   %IntLiteralToInt16: %IntLiteralToInt16.type = struct_value () [template]
// CHECK:STDOUT:   %IntLiteralToUint16.type: type = fn_type @IntLiteralToUint16 [template]
// CHECK:STDOUT:   %IntLiteralToUint16: %IntLiteralToUint16.type = struct_value () [template]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %i64: type = int_type signed, %int_64 [template]
// CHECK:STDOUT:   %Int32ToInt64.type: type = fn_type @Int32ToInt64 [template]
// CHECK:STDOUT:   %Int32ToInt64: %Int32ToInt64.type = struct_value () [template]
// CHECK:STDOUT:   %u64: type = int_type unsigned, %int_64 [template]
// CHECK:STDOUT:   %Int32ToUint64.type: type = fn_type @Int32ToUint64 [template]
// CHECK:STDOUT:   %Int32ToUint64: %Int32ToUint64.type = struct_value () [template]
// CHECK:STDOUT:   %Uint32ToInt64.type: type = fn_type @Uint32ToInt64 [template]
// CHECK:STDOUT:   %Uint32ToInt64: %Uint32ToInt64.type = struct_value () [template]
// CHECK:STDOUT:   %Uint32ToUint64.type: type = fn_type @Uint32ToUint64 [template]
// CHECK:STDOUT:   %Uint32ToUint64: %Uint32ToUint64.type = struct_value () [template]
// CHECK:STDOUT:   %Int32ToIntLiteral.type: type = fn_type @Int32ToIntLiteral [template]
// CHECK:STDOUT:   %Int32ToIntLiteral: %Int32ToIntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %Uint32ToUintLiteral.type: type = fn_type @Uint32ToUintLiteral [template]
// CHECK:STDOUT:   %Uint32ToUintLiteral: %Uint32ToUintLiteral.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .UInt = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .NegateI32 = %NegateI32.decl
// CHECK:STDOUT:     .SubI32 = %SubI32.decl
// CHECK:STDOUT:     .AddU32 = %AddU32.decl
// CHECK:STDOUT:     .IntLiteral = %IntLiteral.decl
// CHECK:STDOUT:     .Int32ToInt32 = %Int32ToInt32.decl
// CHECK:STDOUT:     .Int32ToUint32 = %Int32ToUint32.decl
// CHECK:STDOUT:     .Uint32ToInt32 = %Uint32ToInt32.decl
// CHECK:STDOUT:     .Uint32ToUint32 = %Uint32ToUint32.decl
// CHECK:STDOUT:     .IntLiteralToIntLiteral = %IntLiteralToIntLiteral.decl
// CHECK:STDOUT:     .Int32ToInt16 = %Int32ToInt16.decl
// CHECK:STDOUT:     .Int32ToUint16 = %Int32ToUint16.decl
// CHECK:STDOUT:     .Uint32ToInt16 = %Uint32ToInt16.decl
// CHECK:STDOUT:     .Uint32ToUint16 = %Uint32ToUint16.decl
// CHECK:STDOUT:     .IntLiteralToInt16 = %IntLiteralToInt16.decl
// CHECK:STDOUT:     .IntLiteralToUint16 = %IntLiteralToUint16.decl
// CHECK:STDOUT:     .Int32ToInt64 = %Int32ToInt64.decl
// CHECK:STDOUT:     .Int32ToUint64 = %Int32ToUint64.decl
// CHECK:STDOUT:     .Uint32ToInt64 = %Uint32ToInt64.decl
// CHECK:STDOUT:     .Uint32ToUint64 = %Uint32ToUint64.decl
// CHECK:STDOUT:     .Int32ToIntLiteral = %Int32ToIntLiteral.decl
// CHECK:STDOUT:     .Uint32ToUintLiteral = %Uint32ToUintLiteral.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NegateI32.decl: %NegateI32.type = fn_decl @NegateI32 [template = constants.%NegateI32] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc4_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc4_17: init type = call constants.%Int(%int_32.loc4_17) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_17.1: type = value_of_initializer %int.make_type_signed.loc4_17 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_17.2: type = converted %int.make_type_signed.loc4_17, %.loc4_17.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_25: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc4_25: init type = call constants.%Int(%int_32.loc4_25) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_25.1: type = value_of_initializer %int.make_type_signed.loc4_25 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_25.2: type = converted %int.make_type_signed.loc4_25, %.loc4_25.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SubI32.decl: %SubI32.type = fn_decl @SubI32 [template = constants.%SubI32] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc5_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc5_14: init type = call constants.%Int(%int_32.loc5_14) [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_14.1: type = value_of_initializer %int.make_type_signed.loc5_14 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_14.2: type = converted %int.make_type_signed.loc5_14, %.loc5_14.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc5_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc5_22: init type = call constants.%Int(%int_32.loc5_22) [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_22.1: type = value_of_initializer %int.make_type_signed.loc5_22 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_22.2: type = converted %int.make_type_signed.loc5_22, %.loc5_22.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc5_30: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc5_30: init type = call constants.%Int(%int_32.loc5_30) [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_30.1: type = value_of_initializer %int.make_type_signed.loc5_30 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_30.2: type = converted %int.make_type_signed.loc5_30, %.loc5_30.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddU32.decl: %AddU32.type = fn_decl @AddU32 [template = constants.%AddU32] {
// CHECK:STDOUT:     %a.patt: %u32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %u32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %u32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %u32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %u32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %u32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc6_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_unsigned.loc6_14: init type = call constants.%UInt(%int_32.loc6_14) [template = constants.%u32]
// CHECK:STDOUT:     %.loc6_14.1: type = value_of_initializer %int.make_type_unsigned.loc6_14 [template = constants.%u32]
// CHECK:STDOUT:     %.loc6_14.2: type = converted %int.make_type_unsigned.loc6_14, %.loc6_14.1 [template = constants.%u32]
// CHECK:STDOUT:     %int_32.loc6_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_unsigned.loc6_22: init type = call constants.%UInt(%int_32.loc6_22) [template = constants.%u32]
// CHECK:STDOUT:     %.loc6_22.1: type = value_of_initializer %int.make_type_unsigned.loc6_22 [template = constants.%u32]
// CHECK:STDOUT:     %.loc6_22.2: type = converted %int.make_type_unsigned.loc6_22, %.loc6_22.1 [template = constants.%u32]
// CHECK:STDOUT:     %int_32.loc6_30: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_unsigned.loc6_30: init type = call constants.%UInt(%int_32.loc6_30) [template = constants.%u32]
// CHECK:STDOUT:     %.loc6_30.1: type = value_of_initializer %int.make_type_unsigned.loc6_30 [template = constants.%u32]
// CHECK:STDOUT:     %.loc6_30.2: type = converted %int.make_type_unsigned.loc6_30, %.loc6_30.1 [template = constants.%u32]
// CHECK:STDOUT:     %a.param: %u32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %u32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %u32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %u32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %u32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %u32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntLiteral.decl: %IntLiteral.type = fn_decl @IntLiteral [template = constants.%IntLiteral] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int32ToInt32.decl: %Int32ToInt32.type = fn_decl @Int32ToInt32 [template = constants.%Int32ToInt32] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc10_20: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc10_20: init type = call constants.%Int(%int_32.loc10_20) [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_20.1: type = value_of_initializer %int.make_type_signed.loc10_20 [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_20.2: type = converted %int.make_type_signed.loc10_20, %.loc10_20.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc10_28: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc10_28: init type = call constants.%Int(%int_32.loc10_28) [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_28.1: type = value_of_initializer %int.make_type_signed.loc10_28 [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_28.2: type = converted %int.make_type_signed.loc10_28, %.loc10_28.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int32ToUint32.decl: %Int32ToUint32.type = fn_decl @Int32ToUint32 [template = constants.%Int32ToUint32] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %u32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %u32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc11_21: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32.loc11_21) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_21.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_21.2: type = converted %int.make_type_signed, %.loc11_21.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_29: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_unsigned: init type = call constants.%UInt(%int_32.loc11_29) [template = constants.%u32]
// CHECK:STDOUT:     %.loc11_29.1: type = value_of_initializer %int.make_type_unsigned [template = constants.%u32]
// CHECK:STDOUT:     %.loc11_29.2: type = converted %int.make_type_unsigned, %.loc11_29.1 [template = constants.%u32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %u32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %u32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Uint32ToInt32.decl: %Uint32ToInt32.type = fn_decl @Uint32ToInt32 [template = constants.%Uint32ToInt32] {
// CHECK:STDOUT:     %a.patt: %u32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %u32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc12_21: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_unsigned: init type = call constants.%UInt(%int_32.loc12_21) [template = constants.%u32]
// CHECK:STDOUT:     %.loc12_21.1: type = value_of_initializer %int.make_type_unsigned [template = constants.%u32]
// CHECK:STDOUT:     %.loc12_21.2: type = converted %int.make_type_unsigned, %.loc12_21.1 [template = constants.%u32]
// CHECK:STDOUT:     %int_32.loc12_29: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32.loc12_29) [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_29.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_29.2: type = converted %int.make_type_signed, %.loc12_29.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %u32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %u32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Uint32ToUint32.decl: %Uint32ToUint32.type = fn_decl @Uint32ToUint32 [template = constants.%Uint32ToUint32] {
// CHECK:STDOUT:     %a.patt: %u32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %u32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %u32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %u32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc13_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_unsigned.loc13_22: init type = call constants.%UInt(%int_32.loc13_22) [template = constants.%u32]
// CHECK:STDOUT:     %.loc13_22.1: type = value_of_initializer %int.make_type_unsigned.loc13_22 [template = constants.%u32]
// CHECK:STDOUT:     %.loc13_22.2: type = converted %int.make_type_unsigned.loc13_22, %.loc13_22.1 [template = constants.%u32]
// CHECK:STDOUT:     %int_32.loc13_30: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_unsigned.loc13_30: init type = call constants.%UInt(%int_32.loc13_30) [template = constants.%u32]
// CHECK:STDOUT:     %.loc13_30.1: type = value_of_initializer %int.make_type_unsigned.loc13_30 [template = constants.%u32]
// CHECK:STDOUT:     %.loc13_30.2: type = converted %int.make_type_unsigned.loc13_30, %.loc13_30.1 [template = constants.%u32]
// CHECK:STDOUT:     %a.param: %u32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %u32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %u32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %u32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntLiteralToIntLiteral.decl: %IntLiteralToIntLiteral.type = fn_decl @IntLiteralToIntLiteral [template = constants.%IntLiteralToIntLiteral] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref.loc14_30: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc14_41: init type = call %IntLiteral.ref.loc14_30() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc14_41.1: type = value_of_initializer %int_literal.make_type.loc14_41 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc14_41.2: type = converted %int_literal.make_type.loc14_41, %.loc14_41.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %IntLiteral.ref.loc14_47: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc14_58: init type = call %IntLiteral.ref.loc14_47() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc14_58.1: type = value_of_initializer %int_literal.make_type.loc14_58 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc14_58.2: type = converted %int_literal.make_type.loc14_58, %.loc14_58.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int32ToInt16.decl: %Int32ToInt16.type = fn_decl @Int32ToInt16 [template = constants.%Int32ToInt16] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i16 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i16 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc18_20: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_20.1: type = value_of_initializer %int.make_type_signed.loc18_20 [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_20.2: type = converted %int.make_type_signed.loc18_20, %.loc18_20.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_16: Core.IntLiteral = int_value 16 [template = constants.%int_16]
// CHECK:STDOUT:     %int.make_type_signed.loc18_28: init type = call constants.%Int(%int_16) [template = constants.%i16]
// CHECK:STDOUT:     %.loc18_28.1: type = value_of_initializer %int.make_type_signed.loc18_28 [template = constants.%i16]
// CHECK:STDOUT:     %.loc18_28.2: type = converted %int.make_type_signed.loc18_28, %.loc18_28.1 [template = constants.%i16]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i16 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i16 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int32ToUint16.decl: %Int32ToUint16.type = fn_decl @Int32ToUint16 [template = constants.%Int32ToUint16] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %u16 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %u16 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc19_21.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc19_21.2: type = converted %int.make_type_signed, %.loc19_21.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_16: Core.IntLiteral = int_value 16 [template = constants.%int_16]
// CHECK:STDOUT:     %int.make_type_unsigned: init type = call constants.%UInt(%int_16) [template = constants.%u16]
// CHECK:STDOUT:     %.loc19_29.1: type = value_of_initializer %int.make_type_unsigned [template = constants.%u16]
// CHECK:STDOUT:     %.loc19_29.2: type = converted %int.make_type_unsigned, %.loc19_29.1 [template = constants.%u16]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %u16 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %u16 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Uint32ToInt16.decl: %Uint32ToInt16.type = fn_decl @Uint32ToInt16 [template = constants.%Uint32ToInt16] {
// CHECK:STDOUT:     %a.patt: %u32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %u32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i16 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i16 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_unsigned: init type = call constants.%UInt(%int_32) [template = constants.%u32]
// CHECK:STDOUT:     %.loc20_21.1: type = value_of_initializer %int.make_type_unsigned [template = constants.%u32]
// CHECK:STDOUT:     %.loc20_21.2: type = converted %int.make_type_unsigned, %.loc20_21.1 [template = constants.%u32]
// CHECK:STDOUT:     %int_16: Core.IntLiteral = int_value 16 [template = constants.%int_16]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_16) [template = constants.%i16]
// CHECK:STDOUT:     %.loc20_29.1: type = value_of_initializer %int.make_type_signed [template = constants.%i16]
// CHECK:STDOUT:     %.loc20_29.2: type = converted %int.make_type_signed, %.loc20_29.1 [template = constants.%i16]
// CHECK:STDOUT:     %a.param: %u32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %u32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i16 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i16 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Uint32ToUint16.decl: %Uint32ToUint16.type = fn_decl @Uint32ToUint16 [template = constants.%Uint32ToUint16] {
// CHECK:STDOUT:     %a.patt: %u32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %u32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %u16 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %u16 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_unsigned.loc21_22: init type = call constants.%UInt(%int_32) [template = constants.%u32]
// CHECK:STDOUT:     %.loc21_22.1: type = value_of_initializer %int.make_type_unsigned.loc21_22 [template = constants.%u32]
// CHECK:STDOUT:     %.loc21_22.2: type = converted %int.make_type_unsigned.loc21_22, %.loc21_22.1 [template = constants.%u32]
// CHECK:STDOUT:     %int_16: Core.IntLiteral = int_value 16 [template = constants.%int_16]
// CHECK:STDOUT:     %int.make_type_unsigned.loc21_30: init type = call constants.%UInt(%int_16) [template = constants.%u16]
// CHECK:STDOUT:     %.loc21_30.1: type = value_of_initializer %int.make_type_unsigned.loc21_30 [template = constants.%u16]
// CHECK:STDOUT:     %.loc21_30.2: type = converted %int.make_type_unsigned.loc21_30, %.loc21_30.1 [template = constants.%u16]
// CHECK:STDOUT:     %a.param: %u32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %u32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %u16 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %u16 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntLiteralToInt16.decl: %IntLiteralToInt16.type = fn_decl @IntLiteralToInt16 [template = constants.%IntLiteralToInt16] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i16 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i16 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc22_36.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc22_36.2: type = converted %int_literal.make_type, %.loc22_36.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %int_16: Core.IntLiteral = int_value 16 [template = constants.%int_16]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_16) [template = constants.%i16]
// CHECK:STDOUT:     %.loc22_42.1: type = value_of_initializer %int.make_type_signed [template = constants.%i16]
// CHECK:STDOUT:     %.loc22_42.2: type = converted %int.make_type_signed, %.loc22_42.1 [template = constants.%i16]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i16 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i16 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntLiteralToUint16.decl: %IntLiteralToUint16.type = fn_decl @IntLiteralToUint16 [template = constants.%IntLiteralToUint16] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %u16 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %u16 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc23_37.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc23_37.2: type = converted %int_literal.make_type, %.loc23_37.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %int_16: Core.IntLiteral = int_value 16 [template = constants.%int_16]
// CHECK:STDOUT:     %int.make_type_unsigned: init type = call constants.%UInt(%int_16) [template = constants.%u16]
// CHECK:STDOUT:     %.loc23_43.1: type = value_of_initializer %int.make_type_unsigned [template = constants.%u16]
// CHECK:STDOUT:     %.loc23_43.2: type = converted %int.make_type_unsigned, %.loc23_43.1 [template = constants.%u16]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %u16 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %u16 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int32ToInt64.decl: %Int32ToInt64.type = fn_decl @Int32ToInt64 [template = constants.%Int32ToInt64] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i64 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i64 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc26_20: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc26_20.1: type = value_of_initializer %int.make_type_signed.loc26_20 [template = constants.%i32]
// CHECK:STDOUT:     %.loc26_20.2: type = converted %int.make_type_signed.loc26_20, %.loc26_20.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %int.make_type_signed.loc26_28: init type = call constants.%Int(%int_64) [template = constants.%i64]
// CHECK:STDOUT:     %.loc26_28.1: type = value_of_initializer %int.make_type_signed.loc26_28 [template = constants.%i64]
// CHECK:STDOUT:     %.loc26_28.2: type = converted %int.make_type_signed.loc26_28, %.loc26_28.1 [template = constants.%i64]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i64 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i64 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int32ToUint64.decl: %Int32ToUint64.type = fn_decl @Int32ToUint64 [template = constants.%Int32ToUint64] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %u64 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %u64 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc27_21.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc27_21.2: type = converted %int.make_type_signed, %.loc27_21.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %int.make_type_unsigned: init type = call constants.%UInt(%int_64) [template = constants.%u64]
// CHECK:STDOUT:     %.loc27_29.1: type = value_of_initializer %int.make_type_unsigned [template = constants.%u64]
// CHECK:STDOUT:     %.loc27_29.2: type = converted %int.make_type_unsigned, %.loc27_29.1 [template = constants.%u64]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %u64 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %u64 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Uint32ToInt64.decl: %Uint32ToInt64.type = fn_decl @Uint32ToInt64 [template = constants.%Uint32ToInt64] {
// CHECK:STDOUT:     %a.patt: %u32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %u32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i64 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i64 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_unsigned: init type = call constants.%UInt(%int_32) [template = constants.%u32]
// CHECK:STDOUT:     %.loc28_21.1: type = value_of_initializer %int.make_type_unsigned [template = constants.%u32]
// CHECK:STDOUT:     %.loc28_21.2: type = converted %int.make_type_unsigned, %.loc28_21.1 [template = constants.%u32]
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_64) [template = constants.%i64]
// CHECK:STDOUT:     %.loc28_29.1: type = value_of_initializer %int.make_type_signed [template = constants.%i64]
// CHECK:STDOUT:     %.loc28_29.2: type = converted %int.make_type_signed, %.loc28_29.1 [template = constants.%i64]
// CHECK:STDOUT:     %a.param: %u32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %u32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i64 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i64 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Uint32ToUint64.decl: %Uint32ToUint64.type = fn_decl @Uint32ToUint64 [template = constants.%Uint32ToUint64] {
// CHECK:STDOUT:     %a.patt: %u32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %u32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %u64 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %u64 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_unsigned.loc29_22: init type = call constants.%UInt(%int_32) [template = constants.%u32]
// CHECK:STDOUT:     %.loc29_22.1: type = value_of_initializer %int.make_type_unsigned.loc29_22 [template = constants.%u32]
// CHECK:STDOUT:     %.loc29_22.2: type = converted %int.make_type_unsigned.loc29_22, %.loc29_22.1 [template = constants.%u32]
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %int.make_type_unsigned.loc29_30: init type = call constants.%UInt(%int_64) [template = constants.%u64]
// CHECK:STDOUT:     %.loc29_30.1: type = value_of_initializer %int.make_type_unsigned.loc29_30 [template = constants.%u64]
// CHECK:STDOUT:     %.loc29_30.2: type = converted %int.make_type_unsigned.loc29_30, %.loc29_30.1 [template = constants.%u64]
// CHECK:STDOUT:     %a.param: %u32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %u32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %u64 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %u64 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int32ToIntLiteral.decl: %Int32ToIntLiteral.type = fn_decl @Int32ToIntLiteral [template = constants.%Int32ToIntLiteral] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc30_25.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc30_25.2: type = converted %int.make_type_signed, %.loc30_25.1 [template = constants.%i32]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc30_44.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc30_44.2: type = converted %int_literal.make_type, %.loc30_44.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Uint32ToUintLiteral.decl: %Uint32ToUintLiteral.type = fn_decl @Uint32ToUintLiteral [template = constants.%Uint32ToUintLiteral] {
// CHECK:STDOUT:     %a.patt: %u32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %u32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_unsigned: init type = call constants.%UInt(%int_32) [template = constants.%u32]
// CHECK:STDOUT:     %.loc31_27.1: type = value_of_initializer %int.make_type_unsigned [template = constants.%u32]
// CHECK:STDOUT:     %.loc31_27.2: type = converted %int.make_type_unsigned, %.loc31_27.1 [template = constants.%u32]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc31_46.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc31_46.2: type = converted %int_literal.make_type, %.loc31_46.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: %u32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %u32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NegateI32(%a.param_patt: %i32) -> %i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubI32(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.ssub";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddU32(%a.param_patt: %u32, %b.param_patt: %u32) -> %u32 = "int.uadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToInt32(%a.param_patt: %i32) -> %i32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint32(%a.param_patt: %i32) -> %u32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToInt32(%a.param_patt: %u32) -> %i32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToUint32(%a.param_patt: %u32) -> %u32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteralToIntLiteral(%a.param_patt: Core.IntLiteral) -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToInt16(%a.param_patt: %i32) -> %i16 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint16(%a.param_patt: %i32) -> %u16 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToInt16(%a.param_patt: %u32) -> %i16 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToUint16(%a.param_patt: %u32) -> %u16 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteralToInt16(%a.param_patt: Core.IntLiteral) -> %i16 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteralToUint16(%a.param_patt: Core.IntLiteral) -> %u16 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToInt64(%a.param_patt: %i32) -> %i64 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint64(%a.param_patt: %i32) -> %u64 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToInt64(%a.param_patt: %u32) -> %i64 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToUint64(%a.param_patt: %u32) -> %u64 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToIntLiteral(%a.param_patt: %i32) -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToUintLiteral(%a.param_patt: %u32) -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: --- identity.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Int32ToInt32.type: type = fn_type @Int32ToInt32 [template]
// CHECK:STDOUT:   %Int32ToInt32: %Int32ToInt32.type = struct_value () [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Convert.3) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_0, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %iN = call %Convert.specific_fn.1(%int_0) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %i32 = call %Int32ToInt32(%int.convert_checked.1) [symbolic]
// CHECK:STDOUT:   %int_2147483647: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2147483647, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.3: init %iN = call %Convert.specific_fn.2(%int_2147483647) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.4: init %i32 = call %Int32ToInt32(%int.convert_checked.3) [symbolic]
// CHECK:STDOUT:   %SubI32.type: type = fn_type @SubI32 [template]
// CHECK:STDOUT:   %SubI32: %SubI32.type = struct_value () [template]
// CHECK:STDOUT:   %NegateI32.type: type = fn_type @NegateI32 [template]
// CHECK:STDOUT:   %NegateI32: %NegateI32.type = struct_value () [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_1, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.5: init %iN = call %Convert.specific_fn.3(%int_1) [symbolic]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %IntLiteralToIntLiteral.type: type = fn_type @IntLiteralToIntLiteral [template]
// CHECK:STDOUT:   %IntLiteralToIntLiteral: %IntLiteralToIntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %Int32ToIntLiteral.type: type = fn_type @Int32ToIntLiteral [template]
// CHECK:STDOUT:   %Int32ToIntLiteral: %Int32ToIntLiteral.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %NegateI32.type = import_ref Main//int_ops, NegateI32, loaded [template = constants.%NegateI32]
// CHECK:STDOUT:   %import_ref.2: %SubI32.type = import_ref Main//int_ops, SubI32, loaded [template = constants.%SubI32]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//int_ops, AddU32, unloaded
// CHECK:STDOUT:   %import_ref.4: %IntLiteral.type = import_ref Main//int_ops, IntLiteral, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT:   %import_ref.5: %Int32ToInt32.type = import_ref Main//int_ops, Int32ToInt32, loaded [template = constants.%Int32ToInt32]
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//int_ops, Int32ToUint32, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//int_ops, Uint32ToInt32, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, Uint32ToUint32, unloaded
// CHECK:STDOUT:   %import_ref.9: %IntLiteralToIntLiteral.type = import_ref Main//int_ops, IntLiteralToIntLiteral, loaded [template = constants.%IntLiteralToIntLiteral]
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//int_ops, Int32ToInt16, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//int_ops, Int32ToUint16, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//int_ops, Uint32ToInt16, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//int_ops, Uint32ToUint16, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//int_ops, IntLiteralToInt16, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//int_ops, IntLiteralToUint16, unloaded
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//int_ops, Int32ToInt64, unloaded
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//int_ops, Int32ToUint64, unloaded
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//int_ops, Uint32ToInt64, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//int_ops, Uint32ToUint64, unloaded
// CHECK:STDOUT:   %import_ref.20: %Int32ToIntLiteral.type = import_ref Main//int_ops, Int32ToIntLiteral, loaded [template = constants.%Int32ToIntLiteral]
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, Uint32ToUintLiteral, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.22
// CHECK:STDOUT:     .ImplicitAs = %import_ref.23
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NegateI32 = imports.%import_ref.1
// CHECK:STDOUT:     .SubI32 = imports.%import_ref.2
// CHECK:STDOUT:     .AddU32 = imports.%import_ref.3
// CHECK:STDOUT:     .IntLiteral = imports.%import_ref.4
// CHECK:STDOUT:     .Int32ToInt32 = imports.%import_ref.5
// CHECK:STDOUT:     .Int32ToUint32 = imports.%import_ref.6
// CHECK:STDOUT:     .Uint32ToInt32 = imports.%import_ref.7
// CHECK:STDOUT:     .Uint32ToUint32 = imports.%import_ref.8
// CHECK:STDOUT:     .IntLiteralToIntLiteral = imports.%import_ref.9
// CHECK:STDOUT:     .Int32ToInt16 = imports.%import_ref.10
// CHECK:STDOUT:     .Int32ToUint16 = imports.%import_ref.11
// CHECK:STDOUT:     .Uint32ToInt16 = imports.%import_ref.12
// CHECK:STDOUT:     .Uint32ToUint16 = imports.%import_ref.13
// CHECK:STDOUT:     .IntLiteralToInt16 = imports.%import_ref.14
// CHECK:STDOUT:     .IntLiteralToUint16 = imports.%import_ref.15
// CHECK:STDOUT:     .Int32ToInt64 = imports.%import_ref.16
// CHECK:STDOUT:     .Int32ToUint64 = imports.%import_ref.17
// CHECK:STDOUT:     .Uint32ToInt64 = imports.%import_ref.18
// CHECK:STDOUT:     .Uint32ToUint64 = imports.%import_ref.19
// CHECK:STDOUT:     .Int32ToIntLiteral = imports.%import_ref.20
// CHECK:STDOUT:     .Uint32ToUintLiteral = imports.%import_ref.21
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:     .c = @__global_init.%c
// CHECK:STDOUT:     .d = @__global_init.%d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %int_32.loc5: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc5: init type = call constants.%Int(%int_32.loc5) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_8.1: type = value_of_initializer %int.make_type_signed.loc5 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_8.2: type = converted %int.make_type_signed.loc5, %.loc5_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc6: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc6: init type = call constants.%Int(%int_32.loc6) [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_8.1: type = value_of_initializer %int.make_type_signed.loc6 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_8.2: type = converted %int.make_type_signed.loc6, %.loc6_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc7: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc7: init type = call constants.%Int(%int_32.loc7) [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_8.1: type = value_of_initializer %int.make_type_signed.loc7 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_8.2: type = converted %int.make_type_signed.loc7, %.loc7_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.4 [template = constants.%IntLiteral]
// CHECK:STDOUT:   %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:   %.loc8_19.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:   %.loc8_19.2: type = converted %int_literal.make_type, %.loc8_19.1 [template = Core.IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToInt32(%a.param_patt: %i32) -> %i32 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubI32(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.ssub" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NegateI32(%a.param_patt: %i32) -> %i32 = "int.snegate" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteralToIntLiteral(%a.param_patt: Core.IntLiteral) -> Core.IntLiteral = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToIntLiteral(%a.param_patt: %i32) -> Core.IntLiteral = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Int32ToInt32.ref.loc5: %Int32ToInt32.type = name_ref Int32ToInt32, imports.%import_ref.5 [template = constants.%Int32ToInt32]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %impl.elem0.loc5: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc5: <bound method> = bound_method %int_0, %impl.elem0.loc5 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc5: <specific function> = specific_function %Convert.bound.loc5, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc5_27: init %iN = call %Convert.specific_fn.loc5(%int_0) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc5_27.1: %i32 = value_of_initializer %int.convert_checked.loc5_27 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc5_27.2: %i32 = converted %int_0, %.loc5_27.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %int.convert_checked.loc5_28: init %i32 = call %Int32ToInt32.ref.loc5(%.loc5_27.2) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc5_29.1: %i32 = value_of_initializer %int.convert_checked.loc5_28 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc5_29.2: %i32 = converted %int.convert_checked.loc5_28, %.loc5_29.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %a: %i32 = bind_name a, %.loc5_29.2
// CHECK:STDOUT:   %Int32ToInt32.ref.loc6: %Int32ToInt32.type = name_ref Int32ToInt32, imports.%import_ref.5 [template = constants.%Int32ToInt32]
// CHECK:STDOUT:   %int_2147483647.loc6: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %impl.elem0.loc6: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc6: <bound method> = bound_method %int_2147483647.loc6, %impl.elem0.loc6 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc6: <specific function> = specific_function %Convert.bound.loc6, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc6_27: init %iN = call %Convert.specific_fn.loc6(%int_2147483647.loc6) [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc6_27.1: %i32 = value_of_initializer %int.convert_checked.loc6_27 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc6_27.2: %i32 = converted %int_2147483647.loc6, %.loc6_27.1 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %int.convert_checked.loc6_38: init %i32 = call %Int32ToInt32.ref.loc6(%.loc6_27.2) [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %.loc6_39.1: %i32 = value_of_initializer %int.convert_checked.loc6_38 [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %.loc6_39.2: %i32 = converted %int.convert_checked.loc6_38, %.loc6_39.1 [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %b: %i32 = bind_name b, %.loc6_39.2
// CHECK:STDOUT:   %Int32ToInt32.ref.loc7: %Int32ToInt32.type = name_ref Int32ToInt32, imports.%import_ref.5 [template = constants.%Int32ToInt32]
// CHECK:STDOUT:   %SubI32.ref: %SubI32.type = name_ref SubI32, imports.%import_ref.2 [template = constants.%SubI32]
// CHECK:STDOUT:   %NegateI32.ref.loc7: %NegateI32.type = name_ref NegateI32, imports.%import_ref.1 [template = constants.%NegateI32]
// CHECK:STDOUT:   %int_2147483647.loc7: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %impl.elem0.loc7_44: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc7_44: <bound method> = bound_method %int_2147483647.loc7, %impl.elem0.loc7_44 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc7_44: <specific function> = specific_function %Convert.bound.loc7_44, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc7_44: init %iN = call %Convert.specific_fn.loc7_44(%int_2147483647.loc7) [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc7_44.1: %i32 = value_of_initializer %int.convert_checked.loc7_44 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc7_44.2: %i32 = converted %int_2147483647.loc7, %.loc7_44.1 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %int.snegate.loc7: init %i32 = call %NegateI32.ref.loc7(%.loc7_44.2)
// CHECK:STDOUT:   %int_1.loc7: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc7_55.1: %i32 = value_of_initializer %int.snegate.loc7
// CHECK:STDOUT:   %.loc7_55.2: %i32 = converted %int.snegate.loc7, %.loc7_55.1
// CHECK:STDOUT:   %impl.elem0.loc7_58: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc7_58: <bound method> = bound_method %int_1.loc7, %impl.elem0.loc7_58 [symbolic = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc7_58: <specific function> = specific_function %Convert.bound.loc7_58, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc7_58: init %iN = call %Convert.specific_fn.loc7_58(%int_1.loc7) [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %.loc7_58.1: %i32 = value_of_initializer %int.convert_checked.loc7_58 [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %.loc7_58.2: %i32 = converted %int_1.loc7, %.loc7_58.1 [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %int.ssub: init %i32 = call %SubI32.ref(%.loc7_55.2, %.loc7_58.2)
// CHECK:STDOUT:   %.loc7_59.1: %i32 = value_of_initializer %int.ssub
// CHECK:STDOUT:   %.loc7_59.2: %i32 = converted %int.ssub, %.loc7_59.1
// CHECK:STDOUT:   %int.convert_checked.loc7_60: init %i32 = call %Int32ToInt32.ref.loc7(%.loc7_59.2)
// CHECK:STDOUT:   %.loc7_61.1: %i32 = value_of_initializer %int.convert_checked.loc7_60
// CHECK:STDOUT:   %.loc7_61.2: %i32 = converted %int.convert_checked.loc7_60, %.loc7_61.1
// CHECK:STDOUT:   %c: %i32 = bind_name c, %.loc7_61.2
// CHECK:STDOUT:   %IntLiteralToIntLiteral.ref: %IntLiteralToIntLiteral.type = name_ref IntLiteralToIntLiteral, imports.%import_ref.9 [template = constants.%IntLiteralToIntLiteral]
// CHECK:STDOUT:   %Int32ToIntLiteral.ref: %Int32ToIntLiteral.type = name_ref Int32ToIntLiteral, imports.%import_ref.20 [template = constants.%Int32ToIntLiteral]
// CHECK:STDOUT:   %NegateI32.ref.loc8: %NegateI32.type = name_ref NegateI32, imports.%import_ref.1 [template = constants.%NegateI32]
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc8: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [symbolic = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc8: <specific function> = specific_function %Convert.bound.loc8, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc8_74: init %iN = call %Convert.specific_fn.loc8(%int_1.loc8) [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %.loc8_74.1: %i32 = value_of_initializer %int.convert_checked.loc8_74 [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %.loc8_74.2: %i32 = converted %int_1.loc8, %.loc8_74.1 [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %int.snegate.loc8: init %i32 = call %NegateI32.ref.loc8(%.loc8_74.2)
// CHECK:STDOUT:   %.loc8_75.1: %i32 = value_of_initializer %int.snegate.loc8
// CHECK:STDOUT:   %.loc8_75.2: %i32 = converted %int.snegate.loc8, %.loc8_75.1
// CHECK:STDOUT:   %int.convert_checked.loc8_76: init Core.IntLiteral = call %Int32ToIntLiteral.ref(%.loc8_75.2)
// CHECK:STDOUT:   %.loc8_76.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc8_76
// CHECK:STDOUT:   %.loc8_76.2: Core.IntLiteral = converted %int.convert_checked.loc8_76, %.loc8_76.1
// CHECK:STDOUT:   %int.convert_checked.loc8_77: init Core.IntLiteral = call %IntLiteralToIntLiteral.ref(%.loc8_76.2)
// CHECK:STDOUT:   %.loc8_78.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc8_77
// CHECK:STDOUT:   %.loc8_78.2: Core.IntLiteral = converted %int.convert_checked.loc8_77, %.loc8_78.1
// CHECK:STDOUT:   %d: Core.IntLiteral = bind_name d, %.loc8_78.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- same_size.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %u32: type = int_type unsigned, %int_32 [template]
// CHECK:STDOUT:   %Int32ToUint32.type: type = fn_type @Int32ToUint32 [template]
// CHECK:STDOUT:   %Int32ToUint32: %Int32ToUint32.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %int_2147483647: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Convert.3) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_2147483647, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %iN = call %Convert.specific_fn(%int_2147483647) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %u32 = call %Int32ToUint32(%int.convert_checked.1) [symbolic]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %Uint32ToInt32.type: type = fn_type @Uint32ToInt32 [template]
// CHECK:STDOUT:   %Uint32ToInt32: %Uint32ToInt32.type = struct_value () [template]
// CHECK:STDOUT:   %int.convert_checked.3: init %i32 = call %Uint32ToInt32(%int.convert_checked.2) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//int_ops, NegateI32, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//int_ops, SubI32, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//int_ops, AddU32, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//int_ops, IntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//int_ops, Int32ToInt32, unloaded
// CHECK:STDOUT:   %import_ref.6: %Int32ToUint32.type = import_ref Main//int_ops, Int32ToUint32, loaded [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %import_ref.7: %Uint32ToInt32.type = import_ref Main//int_ops, Uint32ToInt32, loaded [template = constants.%Uint32ToInt32]
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, Uint32ToUint32, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//int_ops, IntLiteralToIntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//int_ops, Int32ToInt16, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//int_ops, Int32ToUint16, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//int_ops, Uint32ToInt16, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//int_ops, Uint32ToUint16, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//int_ops, IntLiteralToInt16, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//int_ops, IntLiteralToUint16, unloaded
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//int_ops, Int32ToInt64, unloaded
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//int_ops, Int32ToUint64, unloaded
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//int_ops, Uint32ToInt64, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//int_ops, Uint32ToUint64, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Main//int_ops, Int32ToIntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, Uint32ToUintLiteral, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .UInt = %import_ref.22
// CHECK:STDOUT:     .ImplicitAs = %import_ref.23
// CHECK:STDOUT:     .Int = %import_ref.59
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NegateI32 = imports.%import_ref.1
// CHECK:STDOUT:     .SubI32 = imports.%import_ref.2
// CHECK:STDOUT:     .AddU32 = imports.%import_ref.3
// CHECK:STDOUT:     .IntLiteral = imports.%import_ref.4
// CHECK:STDOUT:     .Int32ToInt32 = imports.%import_ref.5
// CHECK:STDOUT:     .Int32ToUint32 = imports.%import_ref.6
// CHECK:STDOUT:     .Uint32ToInt32 = imports.%import_ref.7
// CHECK:STDOUT:     .Uint32ToUint32 = imports.%import_ref.8
// CHECK:STDOUT:     .IntLiteralToIntLiteral = imports.%import_ref.9
// CHECK:STDOUT:     .Int32ToInt16 = imports.%import_ref.10
// CHECK:STDOUT:     .Int32ToUint16 = imports.%import_ref.11
// CHECK:STDOUT:     .Uint32ToInt16 = imports.%import_ref.12
// CHECK:STDOUT:     .Uint32ToUint16 = imports.%import_ref.13
// CHECK:STDOUT:     .IntLiteralToInt16 = imports.%import_ref.14
// CHECK:STDOUT:     .IntLiteralToUint16 = imports.%import_ref.15
// CHECK:STDOUT:     .Int32ToInt64 = imports.%import_ref.16
// CHECK:STDOUT:     .Int32ToUint64 = imports.%import_ref.17
// CHECK:STDOUT:     .Uint32ToInt64 = imports.%import_ref.18
// CHECK:STDOUT:     .Uint32ToUint64 = imports.%import_ref.19
// CHECK:STDOUT:     .Int32ToIntLiteral = imports.%import_ref.20
// CHECK:STDOUT:     .Uint32ToUintLiteral = imports.%import_ref.21
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .max = @__global_init.%max
// CHECK:STDOUT:     .max_roundtrip = @__global_init.%max_roundtrip
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %int_32.loc5: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_unsigned: init type = call constants.%UInt(%int_32.loc5) [template = constants.%u32]
// CHECK:STDOUT:   %.loc5_10.1: type = value_of_initializer %int.make_type_unsigned [template = constants.%u32]
// CHECK:STDOUT:   %.loc5_10.2: type = converted %int.make_type_unsigned, %.loc5_10.1 [template = constants.%u32]
// CHECK:STDOUT:   %int_32.loc6: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32.loc6) [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_20.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_20.2: type = converted %int.make_type_signed, %.loc6_20.1 [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint32(%a.param_patt: %i32) -> %u32 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToInt32(%a.param_patt: %u32) -> %i32 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Int32ToUint32.ref.loc5: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %int_2147483647.loc5: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %impl.elem0.loc5: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc5: <bound method> = bound_method %int_2147483647.loc5, %impl.elem0.loc5 [symbolic = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc5: <specific function> = specific_function %Convert.bound.loc5, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc5_30: init %iN = call %Convert.specific_fn.loc5(%int_2147483647.loc5) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc5_30.1: %i32 = value_of_initializer %int.convert_checked.loc5_30 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc5_30.2: %i32 = converted %int_2147483647.loc5, %.loc5_30.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %int.convert_checked.loc5_41: init %u32 = call %Int32ToUint32.ref.loc5(%.loc5_30.2) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc5_42.1: %u32 = value_of_initializer %int.convert_checked.loc5_41 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc5_42.2: %u32 = converted %int.convert_checked.loc5_41, %.loc5_42.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %max: %u32 = bind_name max, %.loc5_42.2
// CHECK:STDOUT:   %Uint32ToInt32.ref: %Uint32ToInt32.type = name_ref Uint32ToInt32, imports.%import_ref.7 [template = constants.%Uint32ToInt32]
// CHECK:STDOUT:   %Int32ToUint32.ref.loc6: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %int_2147483647.loc6: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %impl.elem0.loc6: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc6: <bound method> = bound_method %int_2147483647.loc6, %impl.elem0.loc6 [symbolic = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc6: <specific function> = specific_function %Convert.bound.loc6, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc6_54: init %iN = call %Convert.specific_fn.loc6(%int_2147483647.loc6) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc6_54.1: %i32 = value_of_initializer %int.convert_checked.loc6_54 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc6_54.2: %i32 = converted %int_2147483647.loc6, %.loc6_54.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %int.convert_checked.loc6_65: init %u32 = call %Int32ToUint32.ref.loc6(%.loc6_54.2) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc6_65.1: %u32 = value_of_initializer %int.convert_checked.loc6_65 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc6_65.2: %u32 = converted %int.convert_checked.loc6_65, %.loc6_65.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.convert_checked.loc6_66: init %i32 = call %Uint32ToInt32.ref(%.loc6_65.2) [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc6_67.1: %i32 = value_of_initializer %int.convert_checked.loc6_66 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc6_67.2: %i32 = converted %int.convert_checked.loc6_66, %.loc6_67.1 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %max_roundtrip: %i32 = bind_name max_roundtrip, %.loc6_67.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- truncate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %u16: type = int_type unsigned, %int_16 [template]
// CHECK:STDOUT:   %Int32ToUint16.type: type = fn_type @Int32ToUint16 [template]
// CHECK:STDOUT:   %Int32ToUint16: %Int32ToUint16.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Convert.3) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_0, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %iN = call %Convert.specific_fn.1(%int_0) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %u16 = call %Int32ToUint16(%int.convert_checked.1) [symbolic]
// CHECK:STDOUT:   %int_65535: Core.IntLiteral = int_value 65535 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_65535, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.3: init %iN = call %Convert.specific_fn.2(%int_65535) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.4: init %u16 = call %Int32ToUint16(%int.convert_checked.3) [symbolic]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i16: type = int_type signed, %int_16 [template]
// CHECK:STDOUT:   %Int32ToInt16.type: type = fn_type @Int32ToInt16 [template]
// CHECK:STDOUT:   %Int32ToInt16: %Int32ToInt16.type = struct_value () [template]
// CHECK:STDOUT:   %int_32767: Core.IntLiteral = int_value 32767 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_32767, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.5: init %iN = call %Convert.specific_fn.3(%int_32767) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.6: init %i16 = call %Int32ToInt16(%int.convert_checked.5) [symbolic]
// CHECK:STDOUT:   %NegateI32.type: type = fn_type @NegateI32 [template]
// CHECK:STDOUT:   %NegateI32: %NegateI32.type = struct_value () [template]
// CHECK:STDOUT:   %int_32768: Core.IntLiteral = int_value 32768 [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_32768, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.7: init %iN = call %Convert.specific_fn.4(%int_32768) [symbolic]
// CHECK:STDOUT:   %Uint32ToUint16.type: type = fn_type @Uint32ToUint16 [template]
// CHECK:STDOUT:   %Uint32ToUint16: %Uint32ToUint16.type = struct_value () [template]
// CHECK:STDOUT:   %u32: type = int_type unsigned, %int_32 [template]
// CHECK:STDOUT:   %Int32ToUint32.type: type = fn_type @Int32ToUint32 [template]
// CHECK:STDOUT:   %Int32ToUint32: %Int32ToUint32.type = struct_value () [template]
// CHECK:STDOUT:   %int.convert_checked.8: init %u32 = call %Int32ToUint32(%int.convert_checked.1) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.9: init %u16 = call %Uint32ToUint16(%int.convert_checked.8) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.10: init %u32 = call %Int32ToUint32(%int.convert_checked.3) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.11: init %u16 = call %Uint32ToUint16(%int.convert_checked.10) [symbolic]
// CHECK:STDOUT:   %Uint32ToInt16.type: type = fn_type @Uint32ToInt16 [template]
// CHECK:STDOUT:   %Uint32ToInt16: %Uint32ToInt16.type = struct_value () [template]
// CHECK:STDOUT:   %int.convert_checked.12: init %i16 = call %Uint32ToInt16(%int.convert_checked.8) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.13: init %u32 = call %Int32ToUint32(%int.convert_checked.5) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.14: init %i16 = call %Uint32ToInt16(%int.convert_checked.13) [symbolic]
// CHECK:STDOUT:   %IntLiteralToInt16.type: type = fn_type @IntLiteralToInt16 [template]
// CHECK:STDOUT:   %IntLiteralToInt16: %IntLiteralToInt16.type = struct_value () [template]
// CHECK:STDOUT:   %Int32ToIntLiteral.type: type = fn_type @Int32ToIntLiteral [template]
// CHECK:STDOUT:   %Int32ToIntLiteral: %Int32ToIntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %int.convert_checked.15: init Core.IntLiteral = call %Int32ToIntLiteral(%int.convert_checked.5) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.16: init %i16 = call %IntLiteralToInt16(%int.convert_checked.15) [symbolic]
// CHECK:STDOUT:   %IntLiteralToUint16.type: type = fn_type @IntLiteralToUint16 [template]
// CHECK:STDOUT:   %IntLiteralToUint16: %IntLiteralToUint16.type = struct_value () [template]
// CHECK:STDOUT:   %int.convert_checked.17: init Core.IntLiteral = call %Int32ToIntLiteral(%int.convert_checked.1) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.18: init %u16 = call %IntLiteralToUint16(%int.convert_checked.17) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.19: init Core.IntLiteral = call %Int32ToIntLiteral(%int.convert_checked.3) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.20: init %u16 = call %IntLiteralToUint16(%int.convert_checked.19) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %NegateI32.type = import_ref Main//int_ops, NegateI32, loaded [template = constants.%NegateI32]
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//int_ops, SubI32, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//int_ops, AddU32, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//int_ops, IntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//int_ops, Int32ToInt32, unloaded
// CHECK:STDOUT:   %import_ref.6: %Int32ToUint32.type = import_ref Main//int_ops, Int32ToUint32, loaded [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//int_ops, Uint32ToInt32, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, Uint32ToUint32, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//int_ops, IntLiteralToIntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.10: %Int32ToInt16.type = import_ref Main//int_ops, Int32ToInt16, loaded [template = constants.%Int32ToInt16]
// CHECK:STDOUT:   %import_ref.11: %Int32ToUint16.type = import_ref Main//int_ops, Int32ToUint16, loaded [template = constants.%Int32ToUint16]
// CHECK:STDOUT:   %import_ref.12: %Uint32ToInt16.type = import_ref Main//int_ops, Uint32ToInt16, loaded [template = constants.%Uint32ToInt16]
// CHECK:STDOUT:   %import_ref.13: %Uint32ToUint16.type = import_ref Main//int_ops, Uint32ToUint16, loaded [template = constants.%Uint32ToUint16]
// CHECK:STDOUT:   %import_ref.14: %IntLiteralToInt16.type = import_ref Main//int_ops, IntLiteralToInt16, loaded [template = constants.%IntLiteralToInt16]
// CHECK:STDOUT:   %import_ref.15: %IntLiteralToUint16.type = import_ref Main//int_ops, IntLiteralToUint16, loaded [template = constants.%IntLiteralToUint16]
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//int_ops, Int32ToInt64, unloaded
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//int_ops, Int32ToUint64, unloaded
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//int_ops, Uint32ToInt64, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//int_ops, Uint32ToUint64, unloaded
// CHECK:STDOUT:   %import_ref.20: %Int32ToIntLiteral.type = import_ref Main//int_ops, Int32ToIntLiteral, loaded [template = constants.%Int32ToIntLiteral]
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, Uint32ToUintLiteral, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .UInt = %import_ref.22
// CHECK:STDOUT:     .ImplicitAs = %import_ref.23
// CHECK:STDOUT:     .Int = %import_ref.59
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NegateI32 = imports.%import_ref.1
// CHECK:STDOUT:     .SubI32 = imports.%import_ref.2
// CHECK:STDOUT:     .AddU32 = imports.%import_ref.3
// CHECK:STDOUT:     .IntLiteral = imports.%import_ref.4
// CHECK:STDOUT:     .Int32ToInt32 = imports.%import_ref.5
// CHECK:STDOUT:     .Int32ToUint32 = imports.%import_ref.6
// CHECK:STDOUT:     .Uint32ToInt32 = imports.%import_ref.7
// CHECK:STDOUT:     .Uint32ToUint32 = imports.%import_ref.8
// CHECK:STDOUT:     .IntLiteralToIntLiteral = imports.%import_ref.9
// CHECK:STDOUT:     .Int32ToInt16 = imports.%import_ref.10
// CHECK:STDOUT:     .Int32ToUint16 = imports.%import_ref.11
// CHECK:STDOUT:     .Uint32ToInt16 = imports.%import_ref.12
// CHECK:STDOUT:     .Uint32ToUint16 = imports.%import_ref.13
// CHECK:STDOUT:     .IntLiteralToInt16 = imports.%import_ref.14
// CHECK:STDOUT:     .IntLiteralToUint16 = imports.%import_ref.15
// CHECK:STDOUT:     .Int32ToInt64 = imports.%import_ref.16
// CHECK:STDOUT:     .Int32ToUint64 = imports.%import_ref.17
// CHECK:STDOUT:     .Uint32ToInt64 = imports.%import_ref.18
// CHECK:STDOUT:     .Uint32ToUint64 = imports.%import_ref.19
// CHECK:STDOUT:     .Int32ToIntLiteral = imports.%import_ref.20
// CHECK:STDOUT:     .Uint32ToUintLiteral = imports.%import_ref.21
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:     .c = @__global_init.%c
// CHECK:STDOUT:     .d = @__global_init.%d
// CHECK:STDOUT:     .e = @__global_init.%e
// CHECK:STDOUT:     .f = @__global_init.%f
// CHECK:STDOUT:     .g = @__global_init.%g
// CHECK:STDOUT:     .h = @__global_init.%h
// CHECK:STDOUT:     .lit_i16_min = @__global_init.%lit_i16_min
// CHECK:STDOUT:     .lit_i16_max = @__global_init.%lit_i16_max
// CHECK:STDOUT:     .lit_u16_min = @__global_init.%lit_u16_min
// CHECK:STDOUT:     .lit_u16_max = @__global_init.%lit_u16_max
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %int_16.loc5: Core.IntLiteral = int_value 16 [template = constants.%int_16]
// CHECK:STDOUT:   %int.make_type_unsigned.loc5: init type = call constants.%UInt(%int_16.loc5) [template = constants.%u16]
// CHECK:STDOUT:   %.loc5_8.1: type = value_of_initializer %int.make_type_unsigned.loc5 [template = constants.%u16]
// CHECK:STDOUT:   %.loc5_8.2: type = converted %int.make_type_unsigned.loc5, %.loc5_8.1 [template = constants.%u16]
// CHECK:STDOUT:   %int_16.loc6: Core.IntLiteral = int_value 16 [template = constants.%int_16]
// CHECK:STDOUT:   %int.make_type_unsigned.loc6: init type = call constants.%UInt(%int_16.loc6) [template = constants.%u16]
// CHECK:STDOUT:   %.loc6_8.1: type = value_of_initializer %int.make_type_unsigned.loc6 [template = constants.%u16]
// CHECK:STDOUT:   %.loc6_8.2: type = converted %int.make_type_unsigned.loc6, %.loc6_8.1 [template = constants.%u16]
// CHECK:STDOUT:   %int_16.loc8: Core.IntLiteral = int_value 16 [template = constants.%int_16]
// CHECK:STDOUT:   %int.make_type_signed.loc8: init type = call constants.%Int(%int_16.loc8) [template = constants.%i16]
// CHECK:STDOUT:   %.loc8_8.1: type = value_of_initializer %int.make_type_signed.loc8 [template = constants.%i16]
// CHECK:STDOUT:   %.loc8_8.2: type = converted %int.make_type_signed.loc8, %.loc8_8.1 [template = constants.%i16]
// CHECK:STDOUT:   %int_16.loc9: Core.IntLiteral = int_value 16 [template = constants.%int_16]
// CHECK:STDOUT:   %int.make_type_signed.loc9: init type = call constants.%Int(%int_16.loc9) [template = constants.%i16]
// CHECK:STDOUT:   %.loc9_8.1: type = value_of_initializer %int.make_type_signed.loc9 [template = constants.%i16]
// CHECK:STDOUT:   %.loc9_8.2: type = converted %int.make_type_signed.loc9, %.loc9_8.1 [template = constants.%i16]
// CHECK:STDOUT:   %int_16.loc11: Core.IntLiteral = int_value 16 [template = constants.%int_16]
// CHECK:STDOUT:   %int.make_type_unsigned.loc11: init type = call constants.%UInt(%int_16.loc11) [template = constants.%u16]
// CHECK:STDOUT:   %.loc11_8.1: type = value_of_initializer %int.make_type_unsigned.loc11 [template = constants.%u16]
// CHECK:STDOUT:   %.loc11_8.2: type = converted %int.make_type_unsigned.loc11, %.loc11_8.1 [template = constants.%u16]
// CHECK:STDOUT:   %int_16.loc12: Core.IntLiteral = int_value 16 [template = constants.%int_16]
// CHECK:STDOUT:   %int.make_type_unsigned.loc12: init type = call constants.%UInt(%int_16.loc12) [template = constants.%u16]
// CHECK:STDOUT:   %.loc12_8.1: type = value_of_initializer %int.make_type_unsigned.loc12 [template = constants.%u16]
// CHECK:STDOUT:   %.loc12_8.2: type = converted %int.make_type_unsigned.loc12, %.loc12_8.1 [template = constants.%u16]
// CHECK:STDOUT:   %int_16.loc14: Core.IntLiteral = int_value 16 [template = constants.%int_16]
// CHECK:STDOUT:   %int.make_type_signed.loc14: init type = call constants.%Int(%int_16.loc14) [template = constants.%i16]
// CHECK:STDOUT:   %.loc14_8.1: type = value_of_initializer %int.make_type_signed.loc14 [template = constants.%i16]
// CHECK:STDOUT:   %.loc14_8.2: type = converted %int.make_type_signed.loc14, %.loc14_8.1 [template = constants.%i16]
// CHECK:STDOUT:   %int_16.loc15: Core.IntLiteral = int_value 16 [template = constants.%int_16]
// CHECK:STDOUT:   %int.make_type_signed.loc15: init type = call constants.%Int(%int_16.loc15) [template = constants.%i16]
// CHECK:STDOUT:   %.loc15_8.1: type = value_of_initializer %int.make_type_signed.loc15 [template = constants.%i16]
// CHECK:STDOUT:   %.loc15_8.2: type = converted %int.make_type_signed.loc15, %.loc15_8.1 [template = constants.%i16]
// CHECK:STDOUT:   %int_16.loc17: Core.IntLiteral = int_value 16 [template = constants.%int_16]
// CHECK:STDOUT:   %int.make_type_signed.loc17: init type = call constants.%Int(%int_16.loc17) [template = constants.%i16]
// CHECK:STDOUT:   %.loc17_18.1: type = value_of_initializer %int.make_type_signed.loc17 [template = constants.%i16]
// CHECK:STDOUT:   %.loc17_18.2: type = converted %int.make_type_signed.loc17, %.loc17_18.1 [template = constants.%i16]
// CHECK:STDOUT:   %int_16.loc18: Core.IntLiteral = int_value 16 [template = constants.%int_16]
// CHECK:STDOUT:   %int.make_type_signed.loc18: init type = call constants.%Int(%int_16.loc18) [template = constants.%i16]
// CHECK:STDOUT:   %.loc18_18.1: type = value_of_initializer %int.make_type_signed.loc18 [template = constants.%i16]
// CHECK:STDOUT:   %.loc18_18.2: type = converted %int.make_type_signed.loc18, %.loc18_18.1 [template = constants.%i16]
// CHECK:STDOUT:   %int_16.loc20: Core.IntLiteral = int_value 16 [template = constants.%int_16]
// CHECK:STDOUT:   %int.make_type_unsigned.loc20: init type = call constants.%UInt(%int_16.loc20) [template = constants.%u16]
// CHECK:STDOUT:   %.loc20_18.1: type = value_of_initializer %int.make_type_unsigned.loc20 [template = constants.%u16]
// CHECK:STDOUT:   %.loc20_18.2: type = converted %int.make_type_unsigned.loc20, %.loc20_18.1 [template = constants.%u16]
// CHECK:STDOUT:   %int_16.loc21: Core.IntLiteral = int_value 16 [template = constants.%int_16]
// CHECK:STDOUT:   %int.make_type_unsigned.loc21: init type = call constants.%UInt(%int_16.loc21) [template = constants.%u16]
// CHECK:STDOUT:   %.loc21_18.1: type = value_of_initializer %int.make_type_unsigned.loc21 [template = constants.%u16]
// CHECK:STDOUT:   %.loc21_18.2: type = converted %int.make_type_unsigned.loc21, %.loc21_18.1 [template = constants.%u16]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint16(%a.param_patt: %i32) -> %u16 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToInt16(%a.param_patt: %i32) -> %i16 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NegateI32(%a.param_patt: %i32) -> %i32 = "int.snegate" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToUint16(%a.param_patt: %u32) -> %u16 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint32(%a.param_patt: %i32) -> %u32 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToInt16(%a.param_patt: %u32) -> %i16 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteralToInt16(%a.param_patt: Core.IntLiteral) -> %i16 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToIntLiteral(%a.param_patt: %i32) -> Core.IntLiteral = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteralToUint16(%a.param_patt: Core.IntLiteral) -> %u16 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Int32ToUint16.ref.loc5: %Int32ToUint16.type = name_ref Int32ToUint16, imports.%import_ref.11 [template = constants.%Int32ToUint16]
// CHECK:STDOUT:   %int_0.loc5: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %impl.elem0.loc5: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc5: <bound method> = bound_method %int_0.loc5, %impl.elem0.loc5 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc5: <specific function> = specific_function %Convert.bound.loc5, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc5_28: init %iN = call %Convert.specific_fn.loc5(%int_0.loc5) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc5_28.1: %i32 = value_of_initializer %int.convert_checked.loc5_28 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc5_28.2: %i32 = converted %int_0.loc5, %.loc5_28.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %int.convert_checked.loc5_29: init %u16 = call %Int32ToUint16.ref.loc5(%.loc5_28.2) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc5_30.1: %u16 = value_of_initializer %int.convert_checked.loc5_29 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc5_30.2: %u16 = converted %int.convert_checked.loc5_29, %.loc5_30.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %a: %u16 = bind_name a, %.loc5_30.2
// CHECK:STDOUT:   %Int32ToUint16.ref.loc6: %Int32ToUint16.type = name_ref Int32ToUint16, imports.%import_ref.11 [template = constants.%Int32ToUint16]
// CHECK:STDOUT:   %int_65535.loc6: Core.IntLiteral = int_value 65535 [template = constants.%int_65535]
// CHECK:STDOUT:   %impl.elem0.loc6: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc6: <bound method> = bound_method %int_65535.loc6, %impl.elem0.loc6 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc6: <specific function> = specific_function %Convert.bound.loc6, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc6_28: init %iN = call %Convert.specific_fn.loc6(%int_65535.loc6) [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc6_28.1: %i32 = value_of_initializer %int.convert_checked.loc6_28 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc6_28.2: %i32 = converted %int_65535.loc6, %.loc6_28.1 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %int.convert_checked.loc6_34: init %u16 = call %Int32ToUint16.ref.loc6(%.loc6_28.2) [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %.loc6_35.1: %u16 = value_of_initializer %int.convert_checked.loc6_34 [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %.loc6_35.2: %u16 = converted %int.convert_checked.loc6_34, %.loc6_35.1 [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %b: %u16 = bind_name b, %.loc6_35.2
// CHECK:STDOUT:   %Int32ToInt16.ref.loc8: %Int32ToInt16.type = name_ref Int32ToInt16, imports.%import_ref.10 [template = constants.%Int32ToInt16]
// CHECK:STDOUT:   %int_32767.loc8: Core.IntLiteral = int_value 32767 [template = constants.%int_32767]
// CHECK:STDOUT:   %impl.elem0.loc8: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc8: <bound method> = bound_method %int_32767.loc8, %impl.elem0.loc8 [symbolic = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc8: <specific function> = specific_function %Convert.bound.loc8, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc8_27: init %iN = call %Convert.specific_fn.loc8(%int_32767.loc8) [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %.loc8_27.1: %i32 = value_of_initializer %int.convert_checked.loc8_27 [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %.loc8_27.2: %i32 = converted %int_32767.loc8, %.loc8_27.1 [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %int.convert_checked.loc8_33: init %i16 = call %Int32ToInt16.ref.loc8(%.loc8_27.2) [symbolic = constants.%int.convert_checked.6]
// CHECK:STDOUT:   %.loc8_34.1: %i16 = value_of_initializer %int.convert_checked.loc8_33 [symbolic = constants.%int.convert_checked.6]
// CHECK:STDOUT:   %.loc8_34.2: %i16 = converted %int.convert_checked.loc8_33, %.loc8_34.1 [symbolic = constants.%int.convert_checked.6]
// CHECK:STDOUT:   %c: %i16 = bind_name c, %.loc8_34.2
// CHECK:STDOUT:   %Int32ToInt16.ref.loc9: %Int32ToInt16.type = name_ref Int32ToInt16, imports.%import_ref.10 [template = constants.%Int32ToInt16]
// CHECK:STDOUT:   %NegateI32.ref.loc9: %NegateI32.type = name_ref NegateI32, imports.%import_ref.1 [template = constants.%NegateI32]
// CHECK:STDOUT:   %int_32768.loc9: Core.IntLiteral = int_value 32768 [template = constants.%int_32768]
// CHECK:STDOUT:   %impl.elem0.loc9: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc9: <bound method> = bound_method %int_32768.loc9, %impl.elem0.loc9 [symbolic = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc9: <specific function> = specific_function %Convert.bound.loc9, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc9_37: init %iN = call %Convert.specific_fn.loc9(%int_32768.loc9) [symbolic = constants.%int.convert_checked.7]
// CHECK:STDOUT:   %.loc9_37.1: %i32 = value_of_initializer %int.convert_checked.loc9_37 [symbolic = constants.%int.convert_checked.7]
// CHECK:STDOUT:   %.loc9_37.2: %i32 = converted %int_32768.loc9, %.loc9_37.1 [symbolic = constants.%int.convert_checked.7]
// CHECK:STDOUT:   %int.snegate.loc9: init %i32 = call %NegateI32.ref.loc9(%.loc9_37.2)
// CHECK:STDOUT:   %.loc9_43.1: %i32 = value_of_initializer %int.snegate.loc9
// CHECK:STDOUT:   %.loc9_43.2: %i32 = converted %int.snegate.loc9, %.loc9_43.1
// CHECK:STDOUT:   %int.convert_checked.loc9_44: init %i16 = call %Int32ToInt16.ref.loc9(%.loc9_43.2)
// CHECK:STDOUT:   %.loc9_45.1: %i16 = value_of_initializer %int.convert_checked.loc9_44
// CHECK:STDOUT:   %.loc9_45.2: %i16 = converted %int.convert_checked.loc9_44, %.loc9_45.1
// CHECK:STDOUT:   %d: %i16 = bind_name d, %.loc9_45.2
// CHECK:STDOUT:   %Uint32ToUint16.ref.loc11: %Uint32ToUint16.type = name_ref Uint32ToUint16, imports.%import_ref.13 [template = constants.%Uint32ToUint16]
// CHECK:STDOUT:   %Int32ToUint32.ref.loc11: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %int_0.loc11: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %impl.elem0.loc11: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc11: <bound method> = bound_method %int_0.loc11, %impl.elem0.loc11 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11: <specific function> = specific_function %Convert.bound.loc11, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_43: init %iN = call %Convert.specific_fn.loc11(%int_0.loc11) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc11_43.1: %i32 = value_of_initializer %int.convert_checked.loc11_43 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc11_43.2: %i32 = converted %int_0.loc11, %.loc11_43.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_44: init %u32 = call %Int32ToUint32.ref.loc11(%.loc11_43.2) [symbolic = constants.%int.convert_checked.8]
// CHECK:STDOUT:   %.loc11_44.1: %u32 = value_of_initializer %int.convert_checked.loc11_44 [symbolic = constants.%int.convert_checked.8]
// CHECK:STDOUT:   %.loc11_44.2: %u32 = converted %int.convert_checked.loc11_44, %.loc11_44.1 [symbolic = constants.%int.convert_checked.8]
// CHECK:STDOUT:   %int.convert_checked.loc11_45: init %u16 = call %Uint32ToUint16.ref.loc11(%.loc11_44.2) [symbolic = constants.%int.convert_checked.9]
// CHECK:STDOUT:   %.loc11_46.1: %u16 = value_of_initializer %int.convert_checked.loc11_45 [symbolic = constants.%int.convert_checked.9]
// CHECK:STDOUT:   %.loc11_46.2: %u16 = converted %int.convert_checked.loc11_45, %.loc11_46.1 [symbolic = constants.%int.convert_checked.9]
// CHECK:STDOUT:   %e: %u16 = bind_name e, %.loc11_46.2
// CHECK:STDOUT:   %Uint32ToUint16.ref.loc12: %Uint32ToUint16.type = name_ref Uint32ToUint16, imports.%import_ref.13 [template = constants.%Uint32ToUint16]
// CHECK:STDOUT:   %Int32ToUint32.ref.loc12: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %int_65535.loc12: Core.IntLiteral = int_value 65535 [template = constants.%int_65535]
// CHECK:STDOUT:   %impl.elem0.loc12: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc12: <bound method> = bound_method %int_65535.loc12, %impl.elem0.loc12 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc12: <specific function> = specific_function %Convert.bound.loc12, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc12_43: init %iN = call %Convert.specific_fn.loc12(%int_65535.loc12) [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc12_43.1: %i32 = value_of_initializer %int.convert_checked.loc12_43 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc12_43.2: %i32 = converted %int_65535.loc12, %.loc12_43.1 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %int.convert_checked.loc12_49: init %u32 = call %Int32ToUint32.ref.loc12(%.loc12_43.2) [symbolic = constants.%int.convert_checked.10]
// CHECK:STDOUT:   %.loc12_49.1: %u32 = value_of_initializer %int.convert_checked.loc12_49 [symbolic = constants.%int.convert_checked.10]
// CHECK:STDOUT:   %.loc12_49.2: %u32 = converted %int.convert_checked.loc12_49, %.loc12_49.1 [symbolic = constants.%int.convert_checked.10]
// CHECK:STDOUT:   %int.convert_checked.loc12_50: init %u16 = call %Uint32ToUint16.ref.loc12(%.loc12_49.2) [symbolic = constants.%int.convert_checked.11]
// CHECK:STDOUT:   %.loc12_51.1: %u16 = value_of_initializer %int.convert_checked.loc12_50 [symbolic = constants.%int.convert_checked.11]
// CHECK:STDOUT:   %.loc12_51.2: %u16 = converted %int.convert_checked.loc12_50, %.loc12_51.1 [symbolic = constants.%int.convert_checked.11]
// CHECK:STDOUT:   %f: %u16 = bind_name f, %.loc12_51.2
// CHECK:STDOUT:   %Uint32ToInt16.ref.loc14: %Uint32ToInt16.type = name_ref Uint32ToInt16, imports.%import_ref.12 [template = constants.%Uint32ToInt16]
// CHECK:STDOUT:   %Int32ToUint32.ref.loc14: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %int_0.loc14: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %impl.elem0.loc14: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc14: <bound method> = bound_method %int_0.loc14, %impl.elem0.loc14 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc14: <specific function> = specific_function %Convert.bound.loc14, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc14_42: init %iN = call %Convert.specific_fn.loc14(%int_0.loc14) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc14_42.1: %i32 = value_of_initializer %int.convert_checked.loc14_42 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc14_42.2: %i32 = converted %int_0.loc14, %.loc14_42.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %int.convert_checked.loc14_43: init %u32 = call %Int32ToUint32.ref.loc14(%.loc14_42.2) [symbolic = constants.%int.convert_checked.8]
// CHECK:STDOUT:   %.loc14_43.1: %u32 = value_of_initializer %int.convert_checked.loc14_43 [symbolic = constants.%int.convert_checked.8]
// CHECK:STDOUT:   %.loc14_43.2: %u32 = converted %int.convert_checked.loc14_43, %.loc14_43.1 [symbolic = constants.%int.convert_checked.8]
// CHECK:STDOUT:   %int.convert_checked.loc14_44: init %i16 = call %Uint32ToInt16.ref.loc14(%.loc14_43.2) [symbolic = constants.%int.convert_checked.12]
// CHECK:STDOUT:   %.loc14_45.1: %i16 = value_of_initializer %int.convert_checked.loc14_44 [symbolic = constants.%int.convert_checked.12]
// CHECK:STDOUT:   %.loc14_45.2: %i16 = converted %int.convert_checked.loc14_44, %.loc14_45.1 [symbolic = constants.%int.convert_checked.12]
// CHECK:STDOUT:   %g: %i16 = bind_name g, %.loc14_45.2
// CHECK:STDOUT:   %Uint32ToInt16.ref.loc15: %Uint32ToInt16.type = name_ref Uint32ToInt16, imports.%import_ref.12 [template = constants.%Uint32ToInt16]
// CHECK:STDOUT:   %Int32ToUint32.ref.loc15: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %int_32767.loc15: Core.IntLiteral = int_value 32767 [template = constants.%int_32767]
// CHECK:STDOUT:   %impl.elem0.loc15: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc15: <bound method> = bound_method %int_32767.loc15, %impl.elem0.loc15 [symbolic = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc15: <specific function> = specific_function %Convert.bound.loc15, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc15_42: init %iN = call %Convert.specific_fn.loc15(%int_32767.loc15) [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %.loc15_42.1: %i32 = value_of_initializer %int.convert_checked.loc15_42 [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %.loc15_42.2: %i32 = converted %int_32767.loc15, %.loc15_42.1 [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %int.convert_checked.loc15_48: init %u32 = call %Int32ToUint32.ref.loc15(%.loc15_42.2) [symbolic = constants.%int.convert_checked.13]
// CHECK:STDOUT:   %.loc15_48.1: %u32 = value_of_initializer %int.convert_checked.loc15_48 [symbolic = constants.%int.convert_checked.13]
// CHECK:STDOUT:   %.loc15_48.2: %u32 = converted %int.convert_checked.loc15_48, %.loc15_48.1 [symbolic = constants.%int.convert_checked.13]
// CHECK:STDOUT:   %int.convert_checked.loc15_49: init %i16 = call %Uint32ToInt16.ref.loc15(%.loc15_48.2) [symbolic = constants.%int.convert_checked.14]
// CHECK:STDOUT:   %.loc15_50.1: %i16 = value_of_initializer %int.convert_checked.loc15_49 [symbolic = constants.%int.convert_checked.14]
// CHECK:STDOUT:   %.loc15_50.2: %i16 = converted %int.convert_checked.loc15_49, %.loc15_50.1 [symbolic = constants.%int.convert_checked.14]
// CHECK:STDOUT:   %h: %i16 = bind_name h, %.loc15_50.2
// CHECK:STDOUT:   %IntLiteralToInt16.ref.loc17: %IntLiteralToInt16.type = name_ref IntLiteralToInt16, imports.%import_ref.14 [template = constants.%IntLiteralToInt16]
// CHECK:STDOUT:   %Int32ToIntLiteral.ref.loc17: %Int32ToIntLiteral.type = name_ref Int32ToIntLiteral, imports.%import_ref.20 [template = constants.%Int32ToIntLiteral]
// CHECK:STDOUT:   %NegateI32.ref.loc17: %NegateI32.type = name_ref NegateI32, imports.%import_ref.1 [template = constants.%NegateI32]
// CHECK:STDOUT:   %int_32768.loc17: Core.IntLiteral = int_value 32768 [template = constants.%int_32768]
// CHECK:STDOUT:   %impl.elem0.loc17: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc17: <bound method> = bound_method %int_32768.loc17, %impl.elem0.loc17 [symbolic = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc17: <specific function> = specific_function %Convert.bound.loc17, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc17_70: init %iN = call %Convert.specific_fn.loc17(%int_32768.loc17) [symbolic = constants.%int.convert_checked.7]
// CHECK:STDOUT:   %.loc17_70.1: %i32 = value_of_initializer %int.convert_checked.loc17_70 [symbolic = constants.%int.convert_checked.7]
// CHECK:STDOUT:   %.loc17_70.2: %i32 = converted %int_32768.loc17, %.loc17_70.1 [symbolic = constants.%int.convert_checked.7]
// CHECK:STDOUT:   %int.snegate.loc17: init %i32 = call %NegateI32.ref.loc17(%.loc17_70.2)
// CHECK:STDOUT:   %.loc17_76.1: %i32 = value_of_initializer %int.snegate.loc17
// CHECK:STDOUT:   %.loc17_76.2: %i32 = converted %int.snegate.loc17, %.loc17_76.1
// CHECK:STDOUT:   %int.convert_checked.loc17_77: init Core.IntLiteral = call %Int32ToIntLiteral.ref.loc17(%.loc17_76.2)
// CHECK:STDOUT:   %.loc17_77.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc17_77
// CHECK:STDOUT:   %.loc17_77.2: Core.IntLiteral = converted %int.convert_checked.loc17_77, %.loc17_77.1
// CHECK:STDOUT:   %int.convert_checked.loc17_78: init %i16 = call %IntLiteralToInt16.ref.loc17(%.loc17_77.2)
// CHECK:STDOUT:   %.loc17_79.1: %i16 = value_of_initializer %int.convert_checked.loc17_78
// CHECK:STDOUT:   %.loc17_79.2: %i16 = converted %int.convert_checked.loc17_78, %.loc17_79.1
// CHECK:STDOUT:   %lit_i16_min: %i16 = bind_name lit_i16_min, %.loc17_79.2
// CHECK:STDOUT:   %IntLiteralToInt16.ref.loc18: %IntLiteralToInt16.type = name_ref IntLiteralToInt16, imports.%import_ref.14 [template = constants.%IntLiteralToInt16]
// CHECK:STDOUT:   %Int32ToIntLiteral.ref.loc18: %Int32ToIntLiteral.type = name_ref Int32ToIntLiteral, imports.%import_ref.20 [template = constants.%Int32ToIntLiteral]
// CHECK:STDOUT:   %int_32767.loc18: Core.IntLiteral = int_value 32767 [template = constants.%int_32767]
// CHECK:STDOUT:   %impl.elem0.loc18: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc18: <bound method> = bound_method %int_32767.loc18, %impl.elem0.loc18 [symbolic = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc18: <specific function> = specific_function %Convert.bound.loc18, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc18_60: init %iN = call %Convert.specific_fn.loc18(%int_32767.loc18) [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %.loc18_60.1: %i32 = value_of_initializer %int.convert_checked.loc18_60 [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %.loc18_60.2: %i32 = converted %int_32767.loc18, %.loc18_60.1 [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %int.convert_checked.loc18_66: init Core.IntLiteral = call %Int32ToIntLiteral.ref.loc18(%.loc18_60.2) [symbolic = constants.%int.convert_checked.15]
// CHECK:STDOUT:   %.loc18_66.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc18_66 [symbolic = constants.%int.convert_checked.15]
// CHECK:STDOUT:   %.loc18_66.2: Core.IntLiteral = converted %int.convert_checked.loc18_66, %.loc18_66.1 [symbolic = constants.%int.convert_checked.15]
// CHECK:STDOUT:   %int.convert_checked.loc18_67: init %i16 = call %IntLiteralToInt16.ref.loc18(%.loc18_66.2) [symbolic = constants.%int.convert_checked.16]
// CHECK:STDOUT:   %.loc18_68.1: %i16 = value_of_initializer %int.convert_checked.loc18_67 [symbolic = constants.%int.convert_checked.16]
// CHECK:STDOUT:   %.loc18_68.2: %i16 = converted %int.convert_checked.loc18_67, %.loc18_68.1 [symbolic = constants.%int.convert_checked.16]
// CHECK:STDOUT:   %lit_i16_max: %i16 = bind_name lit_i16_max, %.loc18_68.2
// CHECK:STDOUT:   %IntLiteralToUint16.ref.loc20: %IntLiteralToUint16.type = name_ref IntLiteralToUint16, imports.%import_ref.15 [template = constants.%IntLiteralToUint16]
// CHECK:STDOUT:   %Int32ToIntLiteral.ref.loc20: %Int32ToIntLiteral.type = name_ref Int32ToIntLiteral, imports.%import_ref.20 [template = constants.%Int32ToIntLiteral]
// CHECK:STDOUT:   %int_0.loc20: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %impl.elem0.loc20: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc20: <bound method> = bound_method %int_0.loc20, %impl.elem0.loc20 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc20: <specific function> = specific_function %Convert.bound.loc20, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc20_61: init %iN = call %Convert.specific_fn.loc20(%int_0.loc20) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc20_61.1: %i32 = value_of_initializer %int.convert_checked.loc20_61 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc20_61.2: %i32 = converted %int_0.loc20, %.loc20_61.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %int.convert_checked.loc20_62: init Core.IntLiteral = call %Int32ToIntLiteral.ref.loc20(%.loc20_61.2) [symbolic = constants.%int.convert_checked.17]
// CHECK:STDOUT:   %.loc20_62.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc20_62 [symbolic = constants.%int.convert_checked.17]
// CHECK:STDOUT:   %.loc20_62.2: Core.IntLiteral = converted %int.convert_checked.loc20_62, %.loc20_62.1 [symbolic = constants.%int.convert_checked.17]
// CHECK:STDOUT:   %int.convert_checked.loc20_63: init %u16 = call %IntLiteralToUint16.ref.loc20(%.loc20_62.2) [symbolic = constants.%int.convert_checked.18]
// CHECK:STDOUT:   %.loc20_64.1: %u16 = value_of_initializer %int.convert_checked.loc20_63 [symbolic = constants.%int.convert_checked.18]
// CHECK:STDOUT:   %.loc20_64.2: %u16 = converted %int.convert_checked.loc20_63, %.loc20_64.1 [symbolic = constants.%int.convert_checked.18]
// CHECK:STDOUT:   %lit_u16_min: %u16 = bind_name lit_u16_min, %.loc20_64.2
// CHECK:STDOUT:   %IntLiteralToUint16.ref.loc21: %IntLiteralToUint16.type = name_ref IntLiteralToUint16, imports.%import_ref.15 [template = constants.%IntLiteralToUint16]
// CHECK:STDOUT:   %Int32ToIntLiteral.ref.loc21: %Int32ToIntLiteral.type = name_ref Int32ToIntLiteral, imports.%import_ref.20 [template = constants.%Int32ToIntLiteral]
// CHECK:STDOUT:   %int_65535.loc21: Core.IntLiteral = int_value 65535 [template = constants.%int_65535]
// CHECK:STDOUT:   %impl.elem0.loc21: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc21: <bound method> = bound_method %int_65535.loc21, %impl.elem0.loc21 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc21: <specific function> = specific_function %Convert.bound.loc21, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc21_61: init %iN = call %Convert.specific_fn.loc21(%int_65535.loc21) [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc21_61.1: %i32 = value_of_initializer %int.convert_checked.loc21_61 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc21_61.2: %i32 = converted %int_65535.loc21, %.loc21_61.1 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %int.convert_checked.loc21_67: init Core.IntLiteral = call %Int32ToIntLiteral.ref.loc21(%.loc21_61.2) [symbolic = constants.%int.convert_checked.19]
// CHECK:STDOUT:   %.loc21_67.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc21_67 [symbolic = constants.%int.convert_checked.19]
// CHECK:STDOUT:   %.loc21_67.2: Core.IntLiteral = converted %int.convert_checked.loc21_67, %.loc21_67.1 [symbolic = constants.%int.convert_checked.19]
// CHECK:STDOUT:   %int.convert_checked.loc21_68: init %u16 = call %IntLiteralToUint16.ref.loc21(%.loc21_67.2) [symbolic = constants.%int.convert_checked.20]
// CHECK:STDOUT:   %.loc21_69.1: %u16 = value_of_initializer %int.convert_checked.loc21_68 [symbolic = constants.%int.convert_checked.20]
// CHECK:STDOUT:   %.loc21_69.2: %u16 = converted %int.convert_checked.loc21_68, %.loc21_69.1 [symbolic = constants.%int.convert_checked.20]
// CHECK:STDOUT:   %lit_u16_max: %u16 = bind_name lit_u16_max, %.loc21_69.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- zero_extend.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %u64: type = int_type unsigned, %int_64 [template]
// CHECK:STDOUT:   %Uint32ToUint64.type: type = fn_type @Uint32ToUint64 [template]
// CHECK:STDOUT:   %Uint32ToUint64: %Uint32ToUint64.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %u32: type = int_type unsigned, %int_32 [template]
// CHECK:STDOUT:   %Int32ToUint32.type: type = fn_type @Int32ToUint32 [template]
// CHECK:STDOUT:   %Int32ToUint32: %Int32ToUint32.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Convert.3) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_0, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %iN = call %Convert.specific_fn.1(%int_0) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %u32 = call %Int32ToUint32(%int.convert_checked.1) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.3: init %u64 = call %Uint32ToUint64(%int.convert_checked.2) [symbolic]
// CHECK:STDOUT:   %AddU32.type: type = fn_type @AddU32 [template]
// CHECK:STDOUT:   %AddU32: %AddU32.type = struct_value () [template]
// CHECK:STDOUT:   %int_2147483647: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2147483647, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.4: init %iN = call %Convert.specific_fn.2(%int_2147483647) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.5: init %u32 = call %Int32ToUint32(%int.convert_checked.4) [symbolic]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_1, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.6: init %iN = call %Convert.specific_fn.3(%int_1) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.7: init %u32 = call %Int32ToUint32(%int.convert_checked.6) [symbolic]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i64: type = int_type signed, %int_64 [template]
// CHECK:STDOUT:   %Uint32ToInt64.type: type = fn_type @Uint32ToInt64 [template]
// CHECK:STDOUT:   %Uint32ToInt64: %Uint32ToInt64.type = struct_value () [template]
// CHECK:STDOUT:   %int.convert_checked.8: init %i64 = call %Uint32ToInt64(%int.convert_checked.2) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//int_ops, NegateI32, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//int_ops, SubI32, unloaded
// CHECK:STDOUT:   %import_ref.3: %AddU32.type = import_ref Main//int_ops, AddU32, loaded [template = constants.%AddU32]
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//int_ops, IntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//int_ops, Int32ToInt32, unloaded
// CHECK:STDOUT:   %import_ref.6: %Int32ToUint32.type = import_ref Main//int_ops, Int32ToUint32, loaded [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//int_ops, Uint32ToInt32, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, Uint32ToUint32, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//int_ops, IntLiteralToIntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//int_ops, Int32ToInt16, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//int_ops, Int32ToUint16, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//int_ops, Uint32ToInt16, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//int_ops, Uint32ToUint16, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//int_ops, IntLiteralToInt16, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//int_ops, IntLiteralToUint16, unloaded
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//int_ops, Int32ToInt64, unloaded
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//int_ops, Int32ToUint64, unloaded
// CHECK:STDOUT:   %import_ref.18: %Uint32ToInt64.type = import_ref Main//int_ops, Uint32ToInt64, loaded [template = constants.%Uint32ToInt64]
// CHECK:STDOUT:   %import_ref.19: %Uint32ToUint64.type = import_ref Main//int_ops, Uint32ToUint64, loaded [template = constants.%Uint32ToUint64]
// CHECK:STDOUT:   %import_ref.20 = import_ref Main//int_ops, Int32ToIntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, Uint32ToUintLiteral, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .UInt = %import_ref.22
// CHECK:STDOUT:     .ImplicitAs = %import_ref.23
// CHECK:STDOUT:     .Int = %import_ref.59
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NegateI32 = imports.%import_ref.1
// CHECK:STDOUT:     .SubI32 = imports.%import_ref.2
// CHECK:STDOUT:     .AddU32 = imports.%import_ref.3
// CHECK:STDOUT:     .IntLiteral = imports.%import_ref.4
// CHECK:STDOUT:     .Int32ToInt32 = imports.%import_ref.5
// CHECK:STDOUT:     .Int32ToUint32 = imports.%import_ref.6
// CHECK:STDOUT:     .Uint32ToInt32 = imports.%import_ref.7
// CHECK:STDOUT:     .Uint32ToUint32 = imports.%import_ref.8
// CHECK:STDOUT:     .IntLiteralToIntLiteral = imports.%import_ref.9
// CHECK:STDOUT:     .Int32ToInt16 = imports.%import_ref.10
// CHECK:STDOUT:     .Int32ToUint16 = imports.%import_ref.11
// CHECK:STDOUT:     .Uint32ToInt16 = imports.%import_ref.12
// CHECK:STDOUT:     .Uint32ToUint16 = imports.%import_ref.13
// CHECK:STDOUT:     .IntLiteralToInt16 = imports.%import_ref.14
// CHECK:STDOUT:     .IntLiteralToUint16 = imports.%import_ref.15
// CHECK:STDOUT:     .Int32ToInt64 = imports.%import_ref.16
// CHECK:STDOUT:     .Int32ToUint64 = imports.%import_ref.17
// CHECK:STDOUT:     .Uint32ToInt64 = imports.%import_ref.18
// CHECK:STDOUT:     .Uint32ToUint64 = imports.%import_ref.19
// CHECK:STDOUT:     .Int32ToIntLiteral = imports.%import_ref.20
// CHECK:STDOUT:     .Uint32ToUintLiteral = imports.%import_ref.21
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:     .c = @__global_init.%c
// CHECK:STDOUT:     .d = @__global_init.%d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %int_64.loc5: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:   %int.make_type_unsigned.loc5: init type = call constants.%UInt(%int_64.loc5) [template = constants.%u64]
// CHECK:STDOUT:   %.loc5_8.1: type = value_of_initializer %int.make_type_unsigned.loc5 [template = constants.%u64]
// CHECK:STDOUT:   %.loc5_8.2: type = converted %int.make_type_unsigned.loc5, %.loc5_8.1 [template = constants.%u64]
// CHECK:STDOUT:   %int_64.loc6: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:   %int.make_type_unsigned.loc6: init type = call constants.%UInt(%int_64.loc6) [template = constants.%u64]
// CHECK:STDOUT:   %.loc6_8.1: type = value_of_initializer %int.make_type_unsigned.loc6 [template = constants.%u64]
// CHECK:STDOUT:   %.loc6_8.2: type = converted %int.make_type_unsigned.loc6, %.loc6_8.1 [template = constants.%u64]
// CHECK:STDOUT:   %int_64.loc11: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:   %int.make_type_signed.loc11: init type = call constants.%Int(%int_64.loc11) [template = constants.%i64]
// CHECK:STDOUT:   %.loc11_8.1: type = value_of_initializer %int.make_type_signed.loc11 [template = constants.%i64]
// CHECK:STDOUT:   %.loc11_8.2: type = converted %int.make_type_signed.loc11, %.loc11_8.1 [template = constants.%i64]
// CHECK:STDOUT:   %int_64.loc12: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%int_64.loc12) [template = constants.%i64]
// CHECK:STDOUT:   %.loc12_8.1: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i64]
// CHECK:STDOUT:   %.loc12_8.2: type = converted %int.make_type_signed.loc12, %.loc12_8.1 [template = constants.%i64]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToUint64(%a.param_patt: %u32) -> %u64 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint32(%a.param_patt: %i32) -> %u32 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddU32(%a.param_patt: %u32, %b.param_patt: %u32) -> %u32 = "int.uadd" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToInt64(%a.param_patt: %u32) -> %i64 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Uint32ToUint64.ref.loc5: %Uint32ToUint64.type = name_ref Uint32ToUint64, imports.%import_ref.19 [template = constants.%Uint32ToUint64]
// CHECK:STDOUT:   %Int32ToUint32.ref.loc5: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %int_0.loc5: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %impl.elem0.loc5: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc5: <bound method> = bound_method %int_0.loc5, %impl.elem0.loc5 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc5: <specific function> = specific_function %Convert.bound.loc5, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc5_43: init %iN = call %Convert.specific_fn.loc5(%int_0.loc5) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc5_43.1: %i32 = value_of_initializer %int.convert_checked.loc5_43 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc5_43.2: %i32 = converted %int_0.loc5, %.loc5_43.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %int.convert_checked.loc5_44: init %u32 = call %Int32ToUint32.ref.loc5(%.loc5_43.2) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc5_44.1: %u32 = value_of_initializer %int.convert_checked.loc5_44 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc5_44.2: %u32 = converted %int.convert_checked.loc5_44, %.loc5_44.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.convert_checked.loc5_45: init %u64 = call %Uint32ToUint64.ref.loc5(%.loc5_44.2) [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc5_46.1: %u64 = value_of_initializer %int.convert_checked.loc5_45 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc5_46.2: %u64 = converted %int.convert_checked.loc5_45, %.loc5_46.1 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %a: %u64 = bind_name a, %.loc5_46.2
// CHECK:STDOUT:   %Uint32ToUint64.ref.loc6: %Uint32ToUint64.type = name_ref Uint32ToUint64, imports.%import_ref.19 [template = constants.%Uint32ToUint64]
// CHECK:STDOUT:   %AddU32.ref.loc7: %AddU32.type = name_ref AddU32, imports.%import_ref.3 [template = constants.%AddU32]
// CHECK:STDOUT:   %AddU32.ref.loc8: %AddU32.type = name_ref AddU32, imports.%import_ref.3 [template = constants.%AddU32]
// CHECK:STDOUT:   %Int32ToUint32.ref.loc8_12: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %int_2147483647.loc8_26: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %impl.elem0.loc8_26: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc8_26: <bound method> = bound_method %int_2147483647.loc8_26, %impl.elem0.loc8_26 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc8_26: <specific function> = specific_function %Convert.bound.loc8_26, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc8_26: init %iN = call %Convert.specific_fn.loc8_26(%int_2147483647.loc8_26) [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %.loc8_26.1: %i32 = value_of_initializer %int.convert_checked.loc8_26 [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %.loc8_26.2: %i32 = converted %int_2147483647.loc8_26, %.loc8_26.1 [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %int.convert_checked.loc8_37: init %u32 = call %Int32ToUint32.ref.loc8_12(%.loc8_26.2) [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %Int32ToUint32.ref.loc8_40: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %int_2147483647.loc8_54: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %impl.elem0.loc8_54: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc8_54: <bound method> = bound_method %int_2147483647.loc8_54, %impl.elem0.loc8_54 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc8_54: <specific function> = specific_function %Convert.bound.loc8_54, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc8_54: init %iN = call %Convert.specific_fn.loc8_54(%int_2147483647.loc8_54) [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %.loc8_54.1: %i32 = value_of_initializer %int.convert_checked.loc8_54 [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %.loc8_54.2: %i32 = converted %int_2147483647.loc8_54, %.loc8_54.1 [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %int.convert_checked.loc8_65: init %u32 = call %Int32ToUint32.ref.loc8_40(%.loc8_54.2) [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %.loc8_37.1: %u32 = value_of_initializer %int.convert_checked.loc8_37 [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %.loc8_37.2: %u32 = converted %int.convert_checked.loc8_37, %.loc8_37.1 [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %.loc8_65.1: %u32 = value_of_initializer %int.convert_checked.loc8_65 [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %.loc8_65.2: %u32 = converted %int.convert_checked.loc8_65, %.loc8_65.1 [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %int.uadd.loc8: init %u32 = call %AddU32.ref.loc8(%.loc8_37.2, %.loc8_65.2)
// CHECK:STDOUT:   %Int32ToUint32.ref.loc9: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc9: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc9: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9 [symbolic = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc9: <specific function> = specific_function %Convert.bound.loc9, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc9_19: init %iN = call %Convert.specific_fn.loc9(%int_1.loc9) [symbolic = constants.%int.convert_checked.6]
// CHECK:STDOUT:   %.loc9_19.1: %i32 = value_of_initializer %int.convert_checked.loc9_19 [symbolic = constants.%int.convert_checked.6]
// CHECK:STDOUT:   %.loc9_19.2: %i32 = converted %int_1.loc9, %.loc9_19.1 [symbolic = constants.%int.convert_checked.6]
// CHECK:STDOUT:   %int.convert_checked.loc9_20: init %u32 = call %Int32ToUint32.ref.loc9(%.loc9_19.2) [symbolic = constants.%int.convert_checked.7]
// CHECK:STDOUT:   %.loc8_66.1: %u32 = value_of_initializer %int.uadd.loc8
// CHECK:STDOUT:   %.loc8_66.2: %u32 = converted %int.uadd.loc8, %.loc8_66.1
// CHECK:STDOUT:   %.loc9_20.1: %u32 = value_of_initializer %int.convert_checked.loc9_20 [symbolic = constants.%int.convert_checked.7]
// CHECK:STDOUT:   %.loc9_20.2: %u32 = converted %int.convert_checked.loc9_20, %.loc9_20.1 [symbolic = constants.%int.convert_checked.7]
// CHECK:STDOUT:   %int.uadd.loc9: init %u32 = call %AddU32.ref.loc7(%.loc8_66.2, %.loc9_20.2)
// CHECK:STDOUT:   %.loc9_21.1: %u32 = value_of_initializer %int.uadd.loc9
// CHECK:STDOUT:   %.loc9_21.2: %u32 = converted %int.uadd.loc9, %.loc9_21.1
// CHECK:STDOUT:   %int.convert_checked.loc9_22: init %u64 = call %Uint32ToUint64.ref.loc6(%.loc9_21.2)
// CHECK:STDOUT:   %.loc9_23.1: %u64 = value_of_initializer %int.convert_checked.loc9_22
// CHECK:STDOUT:   %.loc9_23.2: %u64 = converted %int.convert_checked.loc9_22, %.loc9_23.1
// CHECK:STDOUT:   %b: %u64 = bind_name b, %.loc9_23.2
// CHECK:STDOUT:   %Uint32ToInt64.ref.loc11: %Uint32ToInt64.type = name_ref Uint32ToInt64, imports.%import_ref.18 [template = constants.%Uint32ToInt64]
// CHECK:STDOUT:   %Int32ToUint32.ref.loc11: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %int_0.loc11: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %impl.elem0.loc11: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc11: <bound method> = bound_method %int_0.loc11, %impl.elem0.loc11 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11: <specific function> = specific_function %Convert.bound.loc11, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_42: init %iN = call %Convert.specific_fn.loc11(%int_0.loc11) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc11_42.1: %i32 = value_of_initializer %int.convert_checked.loc11_42 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc11_42.2: %i32 = converted %int_0.loc11, %.loc11_42.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_43: init %u32 = call %Int32ToUint32.ref.loc11(%.loc11_42.2) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc11_43.1: %u32 = value_of_initializer %int.convert_checked.loc11_43 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc11_43.2: %u32 = converted %int.convert_checked.loc11_43, %.loc11_43.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.convert_checked.loc11_44: init %i64 = call %Uint32ToInt64.ref.loc11(%.loc11_43.2) [symbolic = constants.%int.convert_checked.8]
// CHECK:STDOUT:   %.loc11_45.1: %i64 = value_of_initializer %int.convert_checked.loc11_44 [symbolic = constants.%int.convert_checked.8]
// CHECK:STDOUT:   %.loc11_45.2: %i64 = converted %int.convert_checked.loc11_44, %.loc11_45.1 [symbolic = constants.%int.convert_checked.8]
// CHECK:STDOUT:   %c: %i64 = bind_name c, %.loc11_45.2
// CHECK:STDOUT:   %Uint32ToInt64.ref.loc12: %Uint32ToInt64.type = name_ref Uint32ToInt64, imports.%import_ref.18 [template = constants.%Uint32ToInt64]
// CHECK:STDOUT:   %AddU32.ref.loc13: %AddU32.type = name_ref AddU32, imports.%import_ref.3 [template = constants.%AddU32]
// CHECK:STDOUT:   %AddU32.ref.loc14: %AddU32.type = name_ref AddU32, imports.%import_ref.3 [template = constants.%AddU32]
// CHECK:STDOUT:   %Int32ToUint32.ref.loc14_12: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %int_2147483647.loc14_26: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %impl.elem0.loc14_26: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc14_26: <bound method> = bound_method %int_2147483647.loc14_26, %impl.elem0.loc14_26 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_26: <specific function> = specific_function %Convert.bound.loc14_26, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc14_26: init %iN = call %Convert.specific_fn.loc14_26(%int_2147483647.loc14_26) [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %.loc14_26.1: %i32 = value_of_initializer %int.convert_checked.loc14_26 [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %.loc14_26.2: %i32 = converted %int_2147483647.loc14_26, %.loc14_26.1 [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %int.convert_checked.loc14_37: init %u32 = call %Int32ToUint32.ref.loc14_12(%.loc14_26.2) [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %Int32ToUint32.ref.loc14_40: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %int_2147483647.loc14_54: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %impl.elem0.loc14_54: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc14_54: <bound method> = bound_method %int_2147483647.loc14_54, %impl.elem0.loc14_54 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_54: <specific function> = specific_function %Convert.bound.loc14_54, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc14_54: init %iN = call %Convert.specific_fn.loc14_54(%int_2147483647.loc14_54) [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %.loc14_54.1: %i32 = value_of_initializer %int.convert_checked.loc14_54 [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %.loc14_54.2: %i32 = converted %int_2147483647.loc14_54, %.loc14_54.1 [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %int.convert_checked.loc14_65: init %u32 = call %Int32ToUint32.ref.loc14_40(%.loc14_54.2) [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %.loc14_37.1: %u32 = value_of_initializer %int.convert_checked.loc14_37 [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %.loc14_37.2: %u32 = converted %int.convert_checked.loc14_37, %.loc14_37.1 [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %.loc14_65.1: %u32 = value_of_initializer %int.convert_checked.loc14_65 [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %.loc14_65.2: %u32 = converted %int.convert_checked.loc14_65, %.loc14_65.1 [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %int.uadd.loc14: init %u32 = call %AddU32.ref.loc14(%.loc14_37.2, %.loc14_65.2)
// CHECK:STDOUT:   %Int32ToUint32.ref.loc15: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc15: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc15: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15 [symbolic = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc15: <specific function> = specific_function %Convert.bound.loc15, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc15_19: init %iN = call %Convert.specific_fn.loc15(%int_1.loc15) [symbolic = constants.%int.convert_checked.6]
// CHECK:STDOUT:   %.loc15_19.1: %i32 = value_of_initializer %int.convert_checked.loc15_19 [symbolic = constants.%int.convert_checked.6]
// CHECK:STDOUT:   %.loc15_19.2: %i32 = converted %int_1.loc15, %.loc15_19.1 [symbolic = constants.%int.convert_checked.6]
// CHECK:STDOUT:   %int.convert_checked.loc15_20: init %u32 = call %Int32ToUint32.ref.loc15(%.loc15_19.2) [symbolic = constants.%int.convert_checked.7]
// CHECK:STDOUT:   %.loc14_66.1: %u32 = value_of_initializer %int.uadd.loc14
// CHECK:STDOUT:   %.loc14_66.2: %u32 = converted %int.uadd.loc14, %.loc14_66.1
// CHECK:STDOUT:   %.loc15_20.1: %u32 = value_of_initializer %int.convert_checked.loc15_20 [symbolic = constants.%int.convert_checked.7]
// CHECK:STDOUT:   %.loc15_20.2: %u32 = converted %int.convert_checked.loc15_20, %.loc15_20.1 [symbolic = constants.%int.convert_checked.7]
// CHECK:STDOUT:   %int.uadd.loc15: init %u32 = call %AddU32.ref.loc13(%.loc14_66.2, %.loc15_20.2)
// CHECK:STDOUT:   %.loc15_21.1: %u32 = value_of_initializer %int.uadd.loc15
// CHECK:STDOUT:   %.loc15_21.2: %u32 = converted %int.uadd.loc15, %.loc15_21.1
// CHECK:STDOUT:   %int.convert_checked.loc15_22: init %i64 = call %Uint32ToInt64.ref.loc12(%.loc15_21.2)
// CHECK:STDOUT:   %.loc15_23.1: %i64 = value_of_initializer %int.convert_checked.loc15_22
// CHECK:STDOUT:   %.loc15_23.2: %i64 = converted %int.convert_checked.loc15_22, %.loc15_23.1
// CHECK:STDOUT:   %d: %i64 = bind_name d, %.loc15_23.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- sign_extend.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %u64: type = int_type unsigned, %int_64 [template]
// CHECK:STDOUT:   %Int32ToUint64.type: type = fn_type @Int32ToUint64 [template]
// CHECK:STDOUT:   %Int32ToUint64: %Int32ToUint64.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Convert.3) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_0, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %iN = call %Convert.specific_fn.1(%int_0) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %u64 = call %Int32ToUint64(%int.convert_checked.1) [symbolic]
// CHECK:STDOUT:   %int_2147483647: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2147483647, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.3: init %iN = call %Convert.specific_fn.2(%int_2147483647) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.4: init %u64 = call %Int32ToUint64(%int.convert_checked.3) [symbolic]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i64: type = int_type signed, %int_64 [template]
// CHECK:STDOUT:   %Int32ToInt64.type: type = fn_type @Int32ToInt64 [template]
// CHECK:STDOUT:   %Int32ToInt64: %Int32ToInt64.type = struct_value () [template]
// CHECK:STDOUT:   %SubI32.type: type = fn_type @SubI32 [template]
// CHECK:STDOUT:   %SubI32: %SubI32.type = struct_value () [template]
// CHECK:STDOUT:   %NegateI32.type: type = fn_type @NegateI32 [template]
// CHECK:STDOUT:   %NegateI32: %NegateI32.type = struct_value () [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_1, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.5: init %iN = call %Convert.specific_fn.3(%int_1) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.6: init %i64 = call %Int32ToInt64(%int.convert_checked.3) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %NegateI32.type = import_ref Main//int_ops, NegateI32, loaded [template = constants.%NegateI32]
// CHECK:STDOUT:   %import_ref.2: %SubI32.type = import_ref Main//int_ops, SubI32, loaded [template = constants.%SubI32]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//int_ops, AddU32, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//int_ops, IntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//int_ops, Int32ToInt32, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//int_ops, Int32ToUint32, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//int_ops, Uint32ToInt32, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, Uint32ToUint32, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//int_ops, IntLiteralToIntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//int_ops, Int32ToInt16, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//int_ops, Int32ToUint16, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//int_ops, Uint32ToInt16, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//int_ops, Uint32ToUint16, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//int_ops, IntLiteralToInt16, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//int_ops, IntLiteralToUint16, unloaded
// CHECK:STDOUT:   %import_ref.16: %Int32ToInt64.type = import_ref Main//int_ops, Int32ToInt64, loaded [template = constants.%Int32ToInt64]
// CHECK:STDOUT:   %import_ref.17: %Int32ToUint64.type = import_ref Main//int_ops, Int32ToUint64, loaded [template = constants.%Int32ToUint64]
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//int_ops, Uint32ToInt64, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//int_ops, Uint32ToUint64, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Main//int_ops, Int32ToIntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, Uint32ToUintLiteral, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .UInt = %import_ref.22
// CHECK:STDOUT:     .ImplicitAs = %import_ref.23
// CHECK:STDOUT:     .Int = %import_ref.59
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NegateI32 = imports.%import_ref.1
// CHECK:STDOUT:     .SubI32 = imports.%import_ref.2
// CHECK:STDOUT:     .AddU32 = imports.%import_ref.3
// CHECK:STDOUT:     .IntLiteral = imports.%import_ref.4
// CHECK:STDOUT:     .Int32ToInt32 = imports.%import_ref.5
// CHECK:STDOUT:     .Int32ToUint32 = imports.%import_ref.6
// CHECK:STDOUT:     .Uint32ToInt32 = imports.%import_ref.7
// CHECK:STDOUT:     .Uint32ToUint32 = imports.%import_ref.8
// CHECK:STDOUT:     .IntLiteralToIntLiteral = imports.%import_ref.9
// CHECK:STDOUT:     .Int32ToInt16 = imports.%import_ref.10
// CHECK:STDOUT:     .Int32ToUint16 = imports.%import_ref.11
// CHECK:STDOUT:     .Uint32ToInt16 = imports.%import_ref.12
// CHECK:STDOUT:     .Uint32ToUint16 = imports.%import_ref.13
// CHECK:STDOUT:     .IntLiteralToInt16 = imports.%import_ref.14
// CHECK:STDOUT:     .IntLiteralToUint16 = imports.%import_ref.15
// CHECK:STDOUT:     .Int32ToInt64 = imports.%import_ref.16
// CHECK:STDOUT:     .Int32ToUint64 = imports.%import_ref.17
// CHECK:STDOUT:     .Uint32ToInt64 = imports.%import_ref.18
// CHECK:STDOUT:     .Uint32ToUint64 = imports.%import_ref.19
// CHECK:STDOUT:     .Int32ToIntLiteral = imports.%import_ref.20
// CHECK:STDOUT:     .Uint32ToUintLiteral = imports.%import_ref.21
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:     .c = @__global_init.%c
// CHECK:STDOUT:     .d = @__global_init.%d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %int_64.loc5: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:   %int.make_type_unsigned.loc5: init type = call constants.%UInt(%int_64.loc5) [template = constants.%u64]
// CHECK:STDOUT:   %.loc5_8.1: type = value_of_initializer %int.make_type_unsigned.loc5 [template = constants.%u64]
// CHECK:STDOUT:   %.loc5_8.2: type = converted %int.make_type_unsigned.loc5, %.loc5_8.1 [template = constants.%u64]
// CHECK:STDOUT:   %int_64.loc6: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:   %int.make_type_unsigned.loc6: init type = call constants.%UInt(%int_64.loc6) [template = constants.%u64]
// CHECK:STDOUT:   %.loc6_8.1: type = value_of_initializer %int.make_type_unsigned.loc6 [template = constants.%u64]
// CHECK:STDOUT:   %.loc6_8.2: type = converted %int.make_type_unsigned.loc6, %.loc6_8.1 [template = constants.%u64]
// CHECK:STDOUT:   %int_64.loc8: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:   %int.make_type_signed.loc8: init type = call constants.%Int(%int_64.loc8) [template = constants.%i64]
// CHECK:STDOUT:   %.loc8_8.1: type = value_of_initializer %int.make_type_signed.loc8 [template = constants.%i64]
// CHECK:STDOUT:   %.loc8_8.2: type = converted %int.make_type_signed.loc8, %.loc8_8.1 [template = constants.%i64]
// CHECK:STDOUT:   %int_64.loc9: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:   %int.make_type_signed.loc9: init type = call constants.%Int(%int_64.loc9) [template = constants.%i64]
// CHECK:STDOUT:   %.loc9_8.1: type = value_of_initializer %int.make_type_signed.loc9 [template = constants.%i64]
// CHECK:STDOUT:   %.loc9_8.2: type = converted %int.make_type_signed.loc9, %.loc9_8.1 [template = constants.%i64]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint64(%a.param_patt: %i32) -> %u64 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToInt64(%a.param_patt: %i32) -> %i64 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubI32(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.ssub" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NegateI32(%a.param_patt: %i32) -> %i32 = "int.snegate" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Int32ToUint64.ref.loc5: %Int32ToUint64.type = name_ref Int32ToUint64, imports.%import_ref.17 [template = constants.%Int32ToUint64]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %impl.elem0.loc5: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc5: <bound method> = bound_method %int_0, %impl.elem0.loc5 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc5: <specific function> = specific_function %Convert.bound.loc5, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc5_28: init %iN = call %Convert.specific_fn.loc5(%int_0) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc5_28.1: %i32 = value_of_initializer %int.convert_checked.loc5_28 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc5_28.2: %i32 = converted %int_0, %.loc5_28.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %int.convert_checked.loc5_29: init %u64 = call %Int32ToUint64.ref.loc5(%.loc5_28.2) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc5_30.1: %u64 = value_of_initializer %int.convert_checked.loc5_29 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc5_30.2: %u64 = converted %int.convert_checked.loc5_29, %.loc5_30.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %a: %u64 = bind_name a, %.loc5_30.2
// CHECK:STDOUT:   %Int32ToUint64.ref.loc6: %Int32ToUint64.type = name_ref Int32ToUint64, imports.%import_ref.17 [template = constants.%Int32ToUint64]
// CHECK:STDOUT:   %int_2147483647.loc6: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %impl.elem0.loc6: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc6: <bound method> = bound_method %int_2147483647.loc6, %impl.elem0.loc6 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc6: <specific function> = specific_function %Convert.bound.loc6, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc6_28: init %iN = call %Convert.specific_fn.loc6(%int_2147483647.loc6) [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc6_28.1: %i32 = value_of_initializer %int.convert_checked.loc6_28 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc6_28.2: %i32 = converted %int_2147483647.loc6, %.loc6_28.1 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %int.convert_checked.loc6_39: init %u64 = call %Int32ToUint64.ref.loc6(%.loc6_28.2) [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %.loc6_40.1: %u64 = value_of_initializer %int.convert_checked.loc6_39 [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %.loc6_40.2: %u64 = converted %int.convert_checked.loc6_39, %.loc6_40.1 [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %b: %u64 = bind_name b, %.loc6_40.2
// CHECK:STDOUT:   %Int32ToInt64.ref.loc8: %Int32ToInt64.type = name_ref Int32ToInt64, imports.%import_ref.16 [template = constants.%Int32ToInt64]
// CHECK:STDOUT:   %SubI32.ref: %SubI32.type = name_ref SubI32, imports.%import_ref.2 [template = constants.%SubI32]
// CHECK:STDOUT:   %NegateI32.ref: %NegateI32.type = name_ref NegateI32, imports.%import_ref.1 [template = constants.%NegateI32]
// CHECK:STDOUT:   %int_2147483647.loc8: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %impl.elem0.loc8_44: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc8_44: <bound method> = bound_method %int_2147483647.loc8, %impl.elem0.loc8_44 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc8_44: <specific function> = specific_function %Convert.bound.loc8_44, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc8_44: init %iN = call %Convert.specific_fn.loc8_44(%int_2147483647.loc8) [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc8_44.1: %i32 = value_of_initializer %int.convert_checked.loc8_44 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc8_44.2: %i32 = converted %int_2147483647.loc8, %.loc8_44.1 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %int.snegate: init %i32 = call %NegateI32.ref(%.loc8_44.2)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc8_55.1: %i32 = value_of_initializer %int.snegate
// CHECK:STDOUT:   %.loc8_55.2: %i32 = converted %int.snegate, %.loc8_55.1
// CHECK:STDOUT:   %impl.elem0.loc8_58: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc8_58: <bound method> = bound_method %int_1, %impl.elem0.loc8_58 [symbolic = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc8_58: <specific function> = specific_function %Convert.bound.loc8_58, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc8_58: init %iN = call %Convert.specific_fn.loc8_58(%int_1) [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %.loc8_58.1: %i32 = value_of_initializer %int.convert_checked.loc8_58 [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %.loc8_58.2: %i32 = converted %int_1, %.loc8_58.1 [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %int.ssub: init %i32 = call %SubI32.ref(%.loc8_55.2, %.loc8_58.2)
// CHECK:STDOUT:   %.loc8_59.1: %i32 = value_of_initializer %int.ssub
// CHECK:STDOUT:   %.loc8_59.2: %i32 = converted %int.ssub, %.loc8_59.1
// CHECK:STDOUT:   %int.convert_checked.loc8_60: init %i64 = call %Int32ToInt64.ref.loc8(%.loc8_59.2)
// CHECK:STDOUT:   %.loc8_61.1: %i64 = value_of_initializer %int.convert_checked.loc8_60
// CHECK:STDOUT:   %.loc8_61.2: %i64 = converted %int.convert_checked.loc8_60, %.loc8_61.1
// CHECK:STDOUT:   %c: %i64 = bind_name c, %.loc8_61.2
// CHECK:STDOUT:   %Int32ToInt64.ref.loc9: %Int32ToInt64.type = name_ref Int32ToInt64, imports.%import_ref.16 [template = constants.%Int32ToInt64]
// CHECK:STDOUT:   %int_2147483647.loc9: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %impl.elem0.loc9: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc9: <bound method> = bound_method %int_2147483647.loc9, %impl.elem0.loc9 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc9: <specific function> = specific_function %Convert.bound.loc9, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc9_27: init %iN = call %Convert.specific_fn.loc9(%int_2147483647.loc9) [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc9_27.1: %i32 = value_of_initializer %int.convert_checked.loc9_27 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc9_27.2: %i32 = converted %int_2147483647.loc9, %.loc9_27.1 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %int.convert_checked.loc9_38: init %i64 = call %Int32ToInt64.ref.loc9(%.loc9_27.2) [symbolic = constants.%int.convert_checked.6]
// CHECK:STDOUT:   %.loc9_39.1: %i64 = value_of_initializer %int.convert_checked.loc9_38 [symbolic = constants.%int.convert_checked.6]
// CHECK:STDOUT:   %.loc9_39.2: %i64 = converted %int.convert_checked.loc9_38, %.loc9_39.1 [symbolic = constants.%int.convert_checked.6]
// CHECK:STDOUT:   %d: %i64 = bind_name d, %.loc9_39.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_too_large_u32_for_i32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Uint32ToInt32.type: type = fn_type @Uint32ToInt32 [template]
// CHECK:STDOUT:   %Uint32ToInt32: %Uint32ToInt32.type = struct_value () [template]
// CHECK:STDOUT:   %u32: type = int_type unsigned, %int_32 [template]
// CHECK:STDOUT:   %AddU32.type: type = fn_type @AddU32 [template]
// CHECK:STDOUT:   %AddU32: %AddU32.type = struct_value () [template]
// CHECK:STDOUT:   %Int32ToUint32.type: type = fn_type @Int32ToUint32 [template]
// CHECK:STDOUT:   %Int32ToUint32: %Int32ToUint32.type = struct_value () [template]
// CHECK:STDOUT:   %int_2147483647: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Convert.3) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_2147483647, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %iN = call %Convert.specific_fn.1(%int_2147483647) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %u32 = call %Int32ToUint32(%int.convert_checked.1) [symbolic]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_1, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.3: init %iN = call %Convert.specific_fn.2(%int_1) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.4: init %u32 = call %Int32ToUint32(%int.convert_checked.3) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//int_ops, NegateI32, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//int_ops, SubI32, unloaded
// CHECK:STDOUT:   %import_ref.3: %AddU32.type = import_ref Main//int_ops, AddU32, loaded [template = constants.%AddU32]
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//int_ops, IntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//int_ops, Int32ToInt32, unloaded
// CHECK:STDOUT:   %import_ref.6: %Int32ToUint32.type = import_ref Main//int_ops, Int32ToUint32, loaded [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %import_ref.7: %Uint32ToInt32.type = import_ref Main//int_ops, Uint32ToInt32, loaded [template = constants.%Uint32ToInt32]
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, Uint32ToUint32, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//int_ops, IntLiteralToIntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//int_ops, Int32ToInt16, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//int_ops, Int32ToUint16, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//int_ops, Uint32ToInt16, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//int_ops, Uint32ToUint16, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//int_ops, IntLiteralToInt16, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//int_ops, IntLiteralToUint16, unloaded
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//int_ops, Int32ToInt64, unloaded
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//int_ops, Int32ToUint64, unloaded
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//int_ops, Uint32ToInt64, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//int_ops, Uint32ToUint64, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Main//int_ops, Int32ToIntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, Uint32ToUintLiteral, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.22
// CHECK:STDOUT:     .ImplicitAs = %import_ref.23
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NegateI32 = imports.%import_ref.1
// CHECK:STDOUT:     .SubI32 = imports.%import_ref.2
// CHECK:STDOUT:     .AddU32 = imports.%import_ref.3
// CHECK:STDOUT:     .IntLiteral = imports.%import_ref.4
// CHECK:STDOUT:     .Int32ToInt32 = imports.%import_ref.5
// CHECK:STDOUT:     .Int32ToUint32 = imports.%import_ref.6
// CHECK:STDOUT:     .Uint32ToInt32 = imports.%import_ref.7
// CHECK:STDOUT:     .Uint32ToUint32 = imports.%import_ref.8
// CHECK:STDOUT:     .IntLiteralToIntLiteral = imports.%import_ref.9
// CHECK:STDOUT:     .Int32ToInt16 = imports.%import_ref.10
// CHECK:STDOUT:     .Int32ToUint16 = imports.%import_ref.11
// CHECK:STDOUT:     .Uint32ToInt16 = imports.%import_ref.12
// CHECK:STDOUT:     .Uint32ToUint16 = imports.%import_ref.13
// CHECK:STDOUT:     .IntLiteralToInt16 = imports.%import_ref.14
// CHECK:STDOUT:     .IntLiteralToUint16 = imports.%import_ref.15
// CHECK:STDOUT:     .Int32ToInt64 = imports.%import_ref.16
// CHECK:STDOUT:     .Int32ToUint64 = imports.%import_ref.17
// CHECK:STDOUT:     .Uint32ToInt64 = imports.%import_ref.18
// CHECK:STDOUT:     .Uint32ToUint64 = imports.%import_ref.19
// CHECK:STDOUT:     .Int32ToIntLiteral = imports.%import_ref.20
// CHECK:STDOUT:     .Uint32ToUintLiteral = imports.%import_ref.21
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .max_plus_one = @__global_init.%max_plus_one
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_19.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_19.2: type = converted %int.make_type_signed, %.loc5_19.1 [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToInt32(%a.param_patt: %u32) -> %i32 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddU32(%a.param_patt: %u32, %b.param_patt: %u32) -> %u32 = "int.uadd" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint32(%a.param_patt: %i32) -> %u32 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Uint32ToInt32.ref: %Uint32ToInt32.type = name_ref Uint32ToInt32, imports.%import_ref.7 [template = constants.%Uint32ToInt32]
// CHECK:STDOUT:   %AddU32.ref: %AddU32.type = name_ref AddU32, imports.%import_ref.3 [template = constants.%AddU32]
// CHECK:STDOUT:   %Int32ToUint32.ref.loc11: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %int_2147483647: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %impl.elem0.loc11: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc11: <bound method> = bound_method %int_2147483647, %impl.elem0.loc11 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11: <specific function> = specific_function %Convert.bound.loc11, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_26: init %iN = call %Convert.specific_fn.loc11(%int_2147483647) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc11_26.1: %i32 = value_of_initializer %int.convert_checked.loc11_26 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc11_26.2: %i32 = converted %int_2147483647, %.loc11_26.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_37: init %u32 = call %Int32ToUint32.ref.loc11(%.loc11_26.2) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %Int32ToUint32.ref.loc12: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc12: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc12: <bound method> = bound_method %int_1, %impl.elem0.loc12 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc12: <specific function> = specific_function %Convert.bound.loc12, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc12_26: init %iN = call %Convert.specific_fn.loc12(%int_1) [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc12_26.1: %i32 = value_of_initializer %int.convert_checked.loc12_26 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc12_26.2: %i32 = converted %int_1, %.loc12_26.1 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %int.convert_checked.loc12_27: init %u32 = call %Int32ToUint32.ref.loc12(%.loc12_26.2) [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %.loc11_37.1: %u32 = value_of_initializer %int.convert_checked.loc11_37 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc11_37.2: %u32 = converted %int.convert_checked.loc11_37, %.loc11_37.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc12_27.1: %u32 = value_of_initializer %int.convert_checked.loc12_27 [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %.loc12_27.2: %u32 = converted %int.convert_checked.loc12_27, %.loc12_27.1 [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %int.uadd: init %u32 = call %AddU32.ref(%.loc11_37.2, %.loc12_27.2)
// CHECK:STDOUT:   %.loc12_28.1: %u32 = value_of_initializer %int.uadd
// CHECK:STDOUT:   %.loc12_28.2: %u32 = converted %int.uadd, %.loc12_28.1
// CHECK:STDOUT:   %int.convert_checked.loc12_29: init %i32 = call %Uint32ToInt32.ref(%.loc12_28.2)
// CHECK:STDOUT:   %.loc12_30.1: %i32 = value_of_initializer %int.convert_checked.loc12_29
// CHECK:STDOUT:   %.loc12_30.2: %i32 = converted %int.convert_checked.loc12_29, %.loc12_30.1
// CHECK:STDOUT:   %max_plus_one: %i32 = bind_name max_plus_one, %.loc12_30.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_too_large_i32_for_i16.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i16: type = int_type signed, %int_16 [template]
// CHECK:STDOUT:   %Int32ToInt16.type: type = fn_type @Int32ToInt16 [template]
// CHECK:STDOUT:   %Int32ToInt16: %Int32ToInt16.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %int_32768: Core.IntLiteral = int_value 32768 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Convert.3) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_32768, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %iN = call %Convert.specific_fn(%int_32768) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %i16 = call %Int32ToInt16(%int.convert_checked.1) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//int_ops, NegateI32, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//int_ops, SubI32, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//int_ops, AddU32, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//int_ops, IntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//int_ops, Int32ToInt32, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//int_ops, Int32ToUint32, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//int_ops, Uint32ToInt32, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, Uint32ToUint32, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//int_ops, IntLiteralToIntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.10: %Int32ToInt16.type = import_ref Main//int_ops, Int32ToInt16, loaded [template = constants.%Int32ToInt16]
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//int_ops, Int32ToUint16, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//int_ops, Uint32ToInt16, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//int_ops, Uint32ToUint16, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//int_ops, IntLiteralToInt16, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//int_ops, IntLiteralToUint16, unloaded
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//int_ops, Int32ToInt64, unloaded
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//int_ops, Int32ToUint64, unloaded
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//int_ops, Uint32ToInt64, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//int_ops, Uint32ToUint64, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Main//int_ops, Int32ToIntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, Uint32ToUintLiteral, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.22
// CHECK:STDOUT:     .ImplicitAs = %import_ref.23
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NegateI32 = imports.%import_ref.1
// CHECK:STDOUT:     .SubI32 = imports.%import_ref.2
// CHECK:STDOUT:     .AddU32 = imports.%import_ref.3
// CHECK:STDOUT:     .IntLiteral = imports.%import_ref.4
// CHECK:STDOUT:     .Int32ToInt32 = imports.%import_ref.5
// CHECK:STDOUT:     .Int32ToUint32 = imports.%import_ref.6
// CHECK:STDOUT:     .Uint32ToInt32 = imports.%import_ref.7
// CHECK:STDOUT:     .Uint32ToUint32 = imports.%import_ref.8
// CHECK:STDOUT:     .IntLiteralToIntLiteral = imports.%import_ref.9
// CHECK:STDOUT:     .Int32ToInt16 = imports.%import_ref.10
// CHECK:STDOUT:     .Int32ToUint16 = imports.%import_ref.11
// CHECK:STDOUT:     .Uint32ToInt16 = imports.%import_ref.12
// CHECK:STDOUT:     .Uint32ToUint16 = imports.%import_ref.13
// CHECK:STDOUT:     .IntLiteralToInt16 = imports.%import_ref.14
// CHECK:STDOUT:     .IntLiteralToUint16 = imports.%import_ref.15
// CHECK:STDOUT:     .Int32ToInt64 = imports.%import_ref.16
// CHECK:STDOUT:     .Int32ToUint64 = imports.%import_ref.17
// CHECK:STDOUT:     .Uint32ToInt64 = imports.%import_ref.18
// CHECK:STDOUT:     .Uint32ToUint64 = imports.%import_ref.19
// CHECK:STDOUT:     .Int32ToIntLiteral = imports.%import_ref.20
// CHECK:STDOUT:     .Uint32ToUintLiteral = imports.%import_ref.21
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .max_plus_one = @__global_init.%max_plus_one
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [template = constants.%int_16]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_16) [template = constants.%i16]
// CHECK:STDOUT:   %.loc9_19.1: type = value_of_initializer %int.make_type_signed [template = constants.%i16]
// CHECK:STDOUT:   %.loc9_19.2: type = converted %int.make_type_signed, %.loc9_19.1 [template = constants.%i16]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToInt16(%a.param_patt: %i32) -> %i16 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Int32ToInt16.ref: %Int32ToInt16.type = name_ref Int32ToInt16, imports.%import_ref.10 [template = constants.%Int32ToInt16]
// CHECK:STDOUT:   %int_32768: Core.IntLiteral = int_value 32768 [template = constants.%int_32768]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_32768, %impl.elem0 [symbolic = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc9_38: init %iN = call %Convert.specific_fn(%int_32768) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc9_38.1: %i32 = value_of_initializer %int.convert_checked.loc9_38 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc9_38.2: %i32 = converted %int_32768, %.loc9_38.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %int.convert_checked.loc9_44: init %i16 = call %Int32ToInt16.ref(%.loc9_38.2) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc9_45.1: %i16 = value_of_initializer %int.convert_checked.loc9_44 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc9_45.2: %i16 = converted %int.convert_checked.loc9_44, %.loc9_45.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %max_plus_one: %i16 = bind_name max_plus_one, %.loc9_45.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_too_large_i32_for_u16.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %u16: type = int_type unsigned, %int_16 [template]
// CHECK:STDOUT:   %Int32ToUint16.type: type = fn_type @Int32ToUint16 [template]
// CHECK:STDOUT:   %Int32ToUint16: %Int32ToUint16.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %int_65536: Core.IntLiteral = int_value 65536 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Convert.3) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_65536, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %iN = call %Convert.specific_fn(%int_65536) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %u16 = call %Int32ToUint16(%int.convert_checked.1) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//int_ops, NegateI32, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//int_ops, SubI32, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//int_ops, AddU32, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//int_ops, IntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//int_ops, Int32ToInt32, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//int_ops, Int32ToUint32, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//int_ops, Uint32ToInt32, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, Uint32ToUint32, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//int_ops, IntLiteralToIntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//int_ops, Int32ToInt16, unloaded
// CHECK:STDOUT:   %import_ref.11: %Int32ToUint16.type = import_ref Main//int_ops, Int32ToUint16, loaded [template = constants.%Int32ToUint16]
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//int_ops, Uint32ToInt16, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//int_ops, Uint32ToUint16, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//int_ops, IntLiteralToInt16, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//int_ops, IntLiteralToUint16, unloaded
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//int_ops, Int32ToInt64, unloaded
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//int_ops, Int32ToUint64, unloaded
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//int_ops, Uint32ToInt64, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//int_ops, Uint32ToUint64, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Main//int_ops, Int32ToIntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, Uint32ToUintLiteral, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .UInt = %import_ref.22
// CHECK:STDOUT:     .ImplicitAs = %import_ref.23
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NegateI32 = imports.%import_ref.1
// CHECK:STDOUT:     .SubI32 = imports.%import_ref.2
// CHECK:STDOUT:     .AddU32 = imports.%import_ref.3
// CHECK:STDOUT:     .IntLiteral = imports.%import_ref.4
// CHECK:STDOUT:     .Int32ToInt32 = imports.%import_ref.5
// CHECK:STDOUT:     .Int32ToUint32 = imports.%import_ref.6
// CHECK:STDOUT:     .Uint32ToInt32 = imports.%import_ref.7
// CHECK:STDOUT:     .Uint32ToUint32 = imports.%import_ref.8
// CHECK:STDOUT:     .IntLiteralToIntLiteral = imports.%import_ref.9
// CHECK:STDOUT:     .Int32ToInt16 = imports.%import_ref.10
// CHECK:STDOUT:     .Int32ToUint16 = imports.%import_ref.11
// CHECK:STDOUT:     .Uint32ToInt16 = imports.%import_ref.12
// CHECK:STDOUT:     .Uint32ToUint16 = imports.%import_ref.13
// CHECK:STDOUT:     .IntLiteralToInt16 = imports.%import_ref.14
// CHECK:STDOUT:     .IntLiteralToUint16 = imports.%import_ref.15
// CHECK:STDOUT:     .Int32ToInt64 = imports.%import_ref.16
// CHECK:STDOUT:     .Int32ToUint64 = imports.%import_ref.17
// CHECK:STDOUT:     .Uint32ToInt64 = imports.%import_ref.18
// CHECK:STDOUT:     .Uint32ToUint64 = imports.%import_ref.19
// CHECK:STDOUT:     .Int32ToIntLiteral = imports.%import_ref.20
// CHECK:STDOUT:     .Uint32ToUintLiteral = imports.%import_ref.21
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .max_plus_one = @__global_init.%max_plus_one
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [template = constants.%int_16]
// CHECK:STDOUT:   %int.make_type_unsigned: init type = call constants.%UInt(%int_16) [template = constants.%u16]
// CHECK:STDOUT:   %.loc9_19.1: type = value_of_initializer %int.make_type_unsigned [template = constants.%u16]
// CHECK:STDOUT:   %.loc9_19.2: type = converted %int.make_type_unsigned, %.loc9_19.1 [template = constants.%u16]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint16(%a.param_patt: %i32) -> %u16 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Int32ToUint16.ref: %Int32ToUint16.type = name_ref Int32ToUint16, imports.%import_ref.11 [template = constants.%Int32ToUint16]
// CHECK:STDOUT:   %int_65536: Core.IntLiteral = int_value 65536 [template = constants.%int_65536]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_65536, %impl.elem0 [symbolic = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc9_39: init %iN = call %Convert.specific_fn(%int_65536) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc9_39.1: %i32 = value_of_initializer %int.convert_checked.loc9_39 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc9_39.2: %i32 = converted %int_65536, %.loc9_39.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %int.convert_checked.loc9_47: init %u16 = call %Int32ToUint16.ref(%.loc9_39.2) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc9_48.1: %u16 = value_of_initializer %int.convert_checked.loc9_47 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc9_48.2: %u16 = converted %int.convert_checked.loc9_47, %.loc9_48.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %max_plus_one: %u16 = bind_name max_plus_one, %.loc9_48.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_too_large_u32_for_i16.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i16: type = int_type signed, %int_16 [template]
// CHECK:STDOUT:   %Uint32ToInt16.type: type = fn_type @Uint32ToInt16 [template]
// CHECK:STDOUT:   %Uint32ToInt16: %Uint32ToInt16.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %u32: type = int_type unsigned, %int_32 [template]
// CHECK:STDOUT:   %Int32ToUint32.type: type = fn_type @Int32ToUint32 [template]
// CHECK:STDOUT:   %Int32ToUint32: %Int32ToUint32.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %int_32768: Core.IntLiteral = int_value 32768 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Convert.3) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_32768, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %iN = call %Convert.specific_fn(%int_32768) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %u32 = call %Int32ToUint32(%int.convert_checked.1) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.3: init %i16 = call %Uint32ToInt16(%int.convert_checked.2) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//int_ops, NegateI32, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//int_ops, SubI32, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//int_ops, AddU32, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//int_ops, IntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//int_ops, Int32ToInt32, unloaded
// CHECK:STDOUT:   %import_ref.6: %Int32ToUint32.type = import_ref Main//int_ops, Int32ToUint32, loaded [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//int_ops, Uint32ToInt32, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, Uint32ToUint32, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//int_ops, IntLiteralToIntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//int_ops, Int32ToInt16, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//int_ops, Int32ToUint16, unloaded
// CHECK:STDOUT:   %import_ref.12: %Uint32ToInt16.type = import_ref Main//int_ops, Uint32ToInt16, loaded [template = constants.%Uint32ToInt16]
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//int_ops, Uint32ToUint16, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//int_ops, IntLiteralToInt16, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//int_ops, IntLiteralToUint16, unloaded
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//int_ops, Int32ToInt64, unloaded
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//int_ops, Int32ToUint64, unloaded
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//int_ops, Uint32ToInt64, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//int_ops, Uint32ToUint64, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Main//int_ops, Int32ToIntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, Uint32ToUintLiteral, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.22
// CHECK:STDOUT:     .ImplicitAs = %import_ref.23
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NegateI32 = imports.%import_ref.1
// CHECK:STDOUT:     .SubI32 = imports.%import_ref.2
// CHECK:STDOUT:     .AddU32 = imports.%import_ref.3
// CHECK:STDOUT:     .IntLiteral = imports.%import_ref.4
// CHECK:STDOUT:     .Int32ToInt32 = imports.%import_ref.5
// CHECK:STDOUT:     .Int32ToUint32 = imports.%import_ref.6
// CHECK:STDOUT:     .Uint32ToInt32 = imports.%import_ref.7
// CHECK:STDOUT:     .Uint32ToUint32 = imports.%import_ref.8
// CHECK:STDOUT:     .IntLiteralToIntLiteral = imports.%import_ref.9
// CHECK:STDOUT:     .Int32ToInt16 = imports.%import_ref.10
// CHECK:STDOUT:     .Int32ToUint16 = imports.%import_ref.11
// CHECK:STDOUT:     .Uint32ToInt16 = imports.%import_ref.12
// CHECK:STDOUT:     .Uint32ToUint16 = imports.%import_ref.13
// CHECK:STDOUT:     .IntLiteralToInt16 = imports.%import_ref.14
// CHECK:STDOUT:     .IntLiteralToUint16 = imports.%import_ref.15
// CHECK:STDOUT:     .Int32ToInt64 = imports.%import_ref.16
// CHECK:STDOUT:     .Int32ToUint64 = imports.%import_ref.17
// CHECK:STDOUT:     .Uint32ToInt64 = imports.%import_ref.18
// CHECK:STDOUT:     .Uint32ToUint64 = imports.%import_ref.19
// CHECK:STDOUT:     .Int32ToIntLiteral = imports.%import_ref.20
// CHECK:STDOUT:     .Uint32ToUintLiteral = imports.%import_ref.21
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .max_plus_one = @__global_init.%max_plus_one
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [template = constants.%int_16]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_16) [template = constants.%i16]
// CHECK:STDOUT:   %.loc9_19.1: type = value_of_initializer %int.make_type_signed [template = constants.%i16]
// CHECK:STDOUT:   %.loc9_19.2: type = converted %int.make_type_signed, %.loc9_19.1 [template = constants.%i16]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToInt16(%a.param_patt: %u32) -> %i16 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint32(%a.param_patt: %i32) -> %u32 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Uint32ToInt16.ref: %Uint32ToInt16.type = name_ref Uint32ToInt16, imports.%import_ref.12 [template = constants.%Uint32ToInt16]
// CHECK:STDOUT:   %Int32ToUint32.ref: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %int_32768: Core.IntLiteral = int_value 32768 [template = constants.%int_32768]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_32768, %impl.elem0 [symbolic = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc9_53: init %iN = call %Convert.specific_fn(%int_32768) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc9_53.1: %i32 = value_of_initializer %int.convert_checked.loc9_53 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc9_53.2: %i32 = converted %int_32768, %.loc9_53.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %int.convert_checked.loc9_59: init %u32 = call %Int32ToUint32.ref(%.loc9_53.2) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc9_59.1: %u32 = value_of_initializer %int.convert_checked.loc9_59 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc9_59.2: %u32 = converted %int.convert_checked.loc9_59, %.loc9_59.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.convert_checked.loc9_60: init %i16 = call %Uint32ToInt16.ref(%.loc9_59.2) [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc9_61.1: %i16 = value_of_initializer %int.convert_checked.loc9_60 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc9_61.2: %i16 = converted %int.convert_checked.loc9_60, %.loc9_61.1 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %max_plus_one: %i16 = bind_name max_plus_one, %.loc9_61.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_too_large_u32_for_u16.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %u16: type = int_type unsigned, %int_16 [template]
// CHECK:STDOUT:   %Uint32ToUint16.type: type = fn_type @Uint32ToUint16 [template]
// CHECK:STDOUT:   %Uint32ToUint16: %Uint32ToUint16.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %u32: type = int_type unsigned, %int_32 [template]
// CHECK:STDOUT:   %Int32ToUint32.type: type = fn_type @Int32ToUint32 [template]
// CHECK:STDOUT:   %Int32ToUint32: %Int32ToUint32.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %int_65536: Core.IntLiteral = int_value 65536 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Convert.3) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_65536, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %iN = call %Convert.specific_fn(%int_65536) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %u32 = call %Int32ToUint32(%int.convert_checked.1) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.3: init %u16 = call %Uint32ToUint16(%int.convert_checked.2) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//int_ops, NegateI32, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//int_ops, SubI32, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//int_ops, AddU32, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//int_ops, IntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//int_ops, Int32ToInt32, unloaded
// CHECK:STDOUT:   %import_ref.6: %Int32ToUint32.type = import_ref Main//int_ops, Int32ToUint32, loaded [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//int_ops, Uint32ToInt32, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, Uint32ToUint32, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//int_ops, IntLiteralToIntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//int_ops, Int32ToInt16, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//int_ops, Int32ToUint16, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//int_ops, Uint32ToInt16, unloaded
// CHECK:STDOUT:   %import_ref.13: %Uint32ToUint16.type = import_ref Main//int_ops, Uint32ToUint16, loaded [template = constants.%Uint32ToUint16]
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//int_ops, IntLiteralToInt16, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//int_ops, IntLiteralToUint16, unloaded
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//int_ops, Int32ToInt64, unloaded
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//int_ops, Int32ToUint64, unloaded
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//int_ops, Uint32ToInt64, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//int_ops, Uint32ToUint64, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Main//int_ops, Int32ToIntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, Uint32ToUintLiteral, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .UInt = %import_ref.22
// CHECK:STDOUT:     .ImplicitAs = %import_ref.23
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NegateI32 = imports.%import_ref.1
// CHECK:STDOUT:     .SubI32 = imports.%import_ref.2
// CHECK:STDOUT:     .AddU32 = imports.%import_ref.3
// CHECK:STDOUT:     .IntLiteral = imports.%import_ref.4
// CHECK:STDOUT:     .Int32ToInt32 = imports.%import_ref.5
// CHECK:STDOUT:     .Int32ToUint32 = imports.%import_ref.6
// CHECK:STDOUT:     .Uint32ToInt32 = imports.%import_ref.7
// CHECK:STDOUT:     .Uint32ToUint32 = imports.%import_ref.8
// CHECK:STDOUT:     .IntLiteralToIntLiteral = imports.%import_ref.9
// CHECK:STDOUT:     .Int32ToInt16 = imports.%import_ref.10
// CHECK:STDOUT:     .Int32ToUint16 = imports.%import_ref.11
// CHECK:STDOUT:     .Uint32ToInt16 = imports.%import_ref.12
// CHECK:STDOUT:     .Uint32ToUint16 = imports.%import_ref.13
// CHECK:STDOUT:     .IntLiteralToInt16 = imports.%import_ref.14
// CHECK:STDOUT:     .IntLiteralToUint16 = imports.%import_ref.15
// CHECK:STDOUT:     .Int32ToInt64 = imports.%import_ref.16
// CHECK:STDOUT:     .Int32ToUint64 = imports.%import_ref.17
// CHECK:STDOUT:     .Uint32ToInt64 = imports.%import_ref.18
// CHECK:STDOUT:     .Uint32ToUint64 = imports.%import_ref.19
// CHECK:STDOUT:     .Int32ToIntLiteral = imports.%import_ref.20
// CHECK:STDOUT:     .Uint32ToUintLiteral = imports.%import_ref.21
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .max_plus_one = @__global_init.%max_plus_one
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [template = constants.%int_16]
// CHECK:STDOUT:   %int.make_type_unsigned: init type = call constants.%UInt(%int_16) [template = constants.%u16]
// CHECK:STDOUT:   %.loc9_19.1: type = value_of_initializer %int.make_type_unsigned [template = constants.%u16]
// CHECK:STDOUT:   %.loc9_19.2: type = converted %int.make_type_unsigned, %.loc9_19.1 [template = constants.%u16]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToUint16(%a.param_patt: %u32) -> %u16 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint32(%a.param_patt: %i32) -> %u32 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Uint32ToUint16.ref: %Uint32ToUint16.type = name_ref Uint32ToUint16, imports.%import_ref.13 [template = constants.%Uint32ToUint16]
// CHECK:STDOUT:   %Int32ToUint32.ref: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %int_65536: Core.IntLiteral = int_value 65536 [template = constants.%int_65536]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_65536, %impl.elem0 [symbolic = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc9_54: init %iN = call %Convert.specific_fn(%int_65536) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc9_54.1: %i32 = value_of_initializer %int.convert_checked.loc9_54 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc9_54.2: %i32 = converted %int_65536, %.loc9_54.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %int.convert_checked.loc9_62: init %u32 = call %Int32ToUint32.ref(%.loc9_54.2) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc9_62.1: %u32 = value_of_initializer %int.convert_checked.loc9_62 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc9_62.2: %u32 = converted %int.convert_checked.loc9_62, %.loc9_62.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.convert_checked.loc9_63: init %u16 = call %Uint32ToUint16.ref(%.loc9_62.2) [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc9_64.1: %u16 = value_of_initializer %int.convert_checked.loc9_63 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc9_64.2: %u16 = converted %int.convert_checked.loc9_63, %.loc9_64.1 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %max_plus_one: %u16 = bind_name max_plus_one, %.loc9_64.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_negative_i32_to_u16.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %u16: type = int_type unsigned, %int_16 [template]
// CHECK:STDOUT:   %Int32ToUint16.type: type = fn_type @Int32ToUint16 [template]
// CHECK:STDOUT:   %Int32ToUint16: %Int32ToUint16.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %SubI32.type: type = fn_type @SubI32 [template]
// CHECK:STDOUT:   %SubI32: %SubI32.type = struct_value () [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Convert.3) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_0, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %iN = call %Convert.specific_fn.1(%int_0) [symbolic]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_1, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %iN = call %Convert.specific_fn.2(%int_1) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//int_ops, NegateI32, unloaded
// CHECK:STDOUT:   %import_ref.2: %SubI32.type = import_ref Main//int_ops, SubI32, loaded [template = constants.%SubI32]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//int_ops, AddU32, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//int_ops, IntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//int_ops, Int32ToInt32, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//int_ops, Int32ToUint32, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//int_ops, Uint32ToInt32, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, Uint32ToUint32, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//int_ops, IntLiteralToIntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//int_ops, Int32ToInt16, unloaded
// CHECK:STDOUT:   %import_ref.11: %Int32ToUint16.type = import_ref Main//int_ops, Int32ToUint16, loaded [template = constants.%Int32ToUint16]
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//int_ops, Uint32ToInt16, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//int_ops, Uint32ToUint16, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//int_ops, IntLiteralToInt16, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//int_ops, IntLiteralToUint16, unloaded
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//int_ops, Int32ToInt64, unloaded
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//int_ops, Int32ToUint64, unloaded
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//int_ops, Uint32ToInt64, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//int_ops, Uint32ToUint64, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Main//int_ops, Int32ToIntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, Uint32ToUintLiteral, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .UInt = %import_ref.22
// CHECK:STDOUT:     .ImplicitAs = %import_ref.23
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NegateI32 = imports.%import_ref.1
// CHECK:STDOUT:     .SubI32 = imports.%import_ref.2
// CHECK:STDOUT:     .AddU32 = imports.%import_ref.3
// CHECK:STDOUT:     .IntLiteral = imports.%import_ref.4
// CHECK:STDOUT:     .Int32ToInt32 = imports.%import_ref.5
// CHECK:STDOUT:     .Int32ToUint32 = imports.%import_ref.6
// CHECK:STDOUT:     .Uint32ToInt32 = imports.%import_ref.7
// CHECK:STDOUT:     .Uint32ToUint32 = imports.%import_ref.8
// CHECK:STDOUT:     .IntLiteralToIntLiteral = imports.%import_ref.9
// CHECK:STDOUT:     .Int32ToInt16 = imports.%import_ref.10
// CHECK:STDOUT:     .Int32ToUint16 = imports.%import_ref.11
// CHECK:STDOUT:     .Uint32ToInt16 = imports.%import_ref.12
// CHECK:STDOUT:     .Uint32ToUint16 = imports.%import_ref.13
// CHECK:STDOUT:     .IntLiteralToInt16 = imports.%import_ref.14
// CHECK:STDOUT:     .IntLiteralToUint16 = imports.%import_ref.15
// CHECK:STDOUT:     .Int32ToInt64 = imports.%import_ref.16
// CHECK:STDOUT:     .Int32ToUint64 = imports.%import_ref.17
// CHECK:STDOUT:     .Uint32ToInt64 = imports.%import_ref.18
// CHECK:STDOUT:     .Uint32ToUint64 = imports.%import_ref.19
// CHECK:STDOUT:     .Int32ToIntLiteral = imports.%import_ref.20
// CHECK:STDOUT:     .Uint32ToUintLiteral = imports.%import_ref.21
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .minus_one_to_u16 = @__global_init.%minus_one_to_u16
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [template = constants.%int_16]
// CHECK:STDOUT:   %int.make_type_unsigned: init type = call constants.%UInt(%int_16) [template = constants.%u16]
// CHECK:STDOUT:   %.loc9_23.1: type = value_of_initializer %int.make_type_unsigned [template = constants.%u16]
// CHECK:STDOUT:   %.loc9_23.2: type = converted %int.make_type_unsigned, %.loc9_23.1 [template = constants.%u16]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint16(%a.param_patt: %i32) -> %u16 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubI32(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.ssub" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Int32ToUint16.ref: %Int32ToUint16.type = name_ref Int32ToUint16, imports.%import_ref.11 [template = constants.%Int32ToUint16]
// CHECK:STDOUT:   %SubI32.ref: %SubI32.type = name_ref SubI32, imports.%import_ref.2 [template = constants.%SubI32]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc9_50: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc9_50: <bound method> = bound_method %int_0, %impl.elem0.loc9_50 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc9_50: <specific function> = specific_function %Convert.bound.loc9_50, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc9_50: init %iN = call %Convert.specific_fn.loc9_50(%int_0) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc9_50.1: %i32 = value_of_initializer %int.convert_checked.loc9_50 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc9_50.2: %i32 = converted %int_0, %.loc9_50.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc9_53: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc9_53: <bound method> = bound_method %int_1, %impl.elem0.loc9_53 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc9_53: <specific function> = specific_function %Convert.bound.loc9_53, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc9_53: init %iN = call %Convert.specific_fn.loc9_53(%int_1) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc9_53.1: %i32 = value_of_initializer %int.convert_checked.loc9_53 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc9_53.2: %i32 = converted %int_1, %.loc9_53.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.ssub: init %i32 = call %SubI32.ref(%.loc9_50.2, %.loc9_53.2)
// CHECK:STDOUT:   %.loc9_54.1: %i32 = value_of_initializer %int.ssub
// CHECK:STDOUT:   %.loc9_54.2: %i32 = converted %int.ssub, %.loc9_54.1
// CHECK:STDOUT:   %int.convert_checked.loc9_55: init %u16 = call %Int32ToUint16.ref(%.loc9_54.2)
// CHECK:STDOUT:   %.loc9_56.1: %u16 = value_of_initializer %int.convert_checked.loc9_55
// CHECK:STDOUT:   %.loc9_56.2: %u16 = converted %int.convert_checked.loc9_55, %.loc9_56.1
// CHECK:STDOUT:   %minus_one_to_u16: %u16 = bind_name minus_one_to_u16, %.loc9_56.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_negative_i32_to_u32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %u32: type = int_type unsigned, %int_32 [template]
// CHECK:STDOUT:   %Int32ToUint32.type: type = fn_type @Int32ToUint32 [template]
// CHECK:STDOUT:   %Int32ToUint32: %Int32ToUint32.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %SubI32.type: type = fn_type @SubI32 [template]
// CHECK:STDOUT:   %SubI32: %SubI32.type = struct_value () [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Convert.3) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_0, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %iN = call %Convert.specific_fn.1(%int_0) [symbolic]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_1, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %iN = call %Convert.specific_fn.2(%int_1) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//int_ops, NegateI32, unloaded
// CHECK:STDOUT:   %import_ref.2: %SubI32.type = import_ref Main//int_ops, SubI32, loaded [template = constants.%SubI32]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//int_ops, AddU32, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//int_ops, IntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//int_ops, Int32ToInt32, unloaded
// CHECK:STDOUT:   %import_ref.6: %Int32ToUint32.type = import_ref Main//int_ops, Int32ToUint32, loaded [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//int_ops, Uint32ToInt32, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, Uint32ToUint32, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//int_ops, IntLiteralToIntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//int_ops, Int32ToInt16, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//int_ops, Int32ToUint16, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//int_ops, Uint32ToInt16, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//int_ops, Uint32ToUint16, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//int_ops, IntLiteralToInt16, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//int_ops, IntLiteralToUint16, unloaded
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//int_ops, Int32ToInt64, unloaded
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//int_ops, Int32ToUint64, unloaded
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//int_ops, Uint32ToInt64, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//int_ops, Uint32ToUint64, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Main//int_ops, Int32ToIntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, Uint32ToUintLiteral, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .UInt = %import_ref.22
// CHECK:STDOUT:     .ImplicitAs = %import_ref.23
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NegateI32 = imports.%import_ref.1
// CHECK:STDOUT:     .SubI32 = imports.%import_ref.2
// CHECK:STDOUT:     .AddU32 = imports.%import_ref.3
// CHECK:STDOUT:     .IntLiteral = imports.%import_ref.4
// CHECK:STDOUT:     .Int32ToInt32 = imports.%import_ref.5
// CHECK:STDOUT:     .Int32ToUint32 = imports.%import_ref.6
// CHECK:STDOUT:     .Uint32ToInt32 = imports.%import_ref.7
// CHECK:STDOUT:     .Uint32ToUint32 = imports.%import_ref.8
// CHECK:STDOUT:     .IntLiteralToIntLiteral = imports.%import_ref.9
// CHECK:STDOUT:     .Int32ToInt16 = imports.%import_ref.10
// CHECK:STDOUT:     .Int32ToUint16 = imports.%import_ref.11
// CHECK:STDOUT:     .Uint32ToInt16 = imports.%import_ref.12
// CHECK:STDOUT:     .Uint32ToUint16 = imports.%import_ref.13
// CHECK:STDOUT:     .IntLiteralToInt16 = imports.%import_ref.14
// CHECK:STDOUT:     .IntLiteralToUint16 = imports.%import_ref.15
// CHECK:STDOUT:     .Int32ToInt64 = imports.%import_ref.16
// CHECK:STDOUT:     .Int32ToUint64 = imports.%import_ref.17
// CHECK:STDOUT:     .Uint32ToInt64 = imports.%import_ref.18
// CHECK:STDOUT:     .Uint32ToUint64 = imports.%import_ref.19
// CHECK:STDOUT:     .Int32ToIntLiteral = imports.%import_ref.20
// CHECK:STDOUT:     .Uint32ToUintLiteral = imports.%import_ref.21
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .minus_one_to_u32 = @__global_init.%minus_one_to_u32
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_unsigned: init type = call constants.%UInt(%int_32) [template = constants.%u32]
// CHECK:STDOUT:   %.loc9_23.1: type = value_of_initializer %int.make_type_unsigned [template = constants.%u32]
// CHECK:STDOUT:   %.loc9_23.2: type = converted %int.make_type_unsigned, %.loc9_23.1 [template = constants.%u32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint32(%a.param_patt: %i32) -> %u32 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubI32(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.ssub" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Int32ToUint32.ref: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %SubI32.ref: %SubI32.type = name_ref SubI32, imports.%import_ref.2 [template = constants.%SubI32]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc9_50: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc9_50: <bound method> = bound_method %int_0, %impl.elem0.loc9_50 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc9_50: <specific function> = specific_function %Convert.bound.loc9_50, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc9_50: init %iN = call %Convert.specific_fn.loc9_50(%int_0) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc9_50.1: %i32 = value_of_initializer %int.convert_checked.loc9_50 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc9_50.2: %i32 = converted %int_0, %.loc9_50.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc9_53: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc9_53: <bound method> = bound_method %int_1, %impl.elem0.loc9_53 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc9_53: <specific function> = specific_function %Convert.bound.loc9_53, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc9_53: init %iN = call %Convert.specific_fn.loc9_53(%int_1) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc9_53.1: %i32 = value_of_initializer %int.convert_checked.loc9_53 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc9_53.2: %i32 = converted %int_1, %.loc9_53.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.ssub: init %i32 = call %SubI32.ref(%.loc9_50.2, %.loc9_53.2)
// CHECK:STDOUT:   %.loc9_54.1: %i32 = value_of_initializer %int.ssub
// CHECK:STDOUT:   %.loc9_54.2: %i32 = converted %int.ssub, %.loc9_54.1
// CHECK:STDOUT:   %int.convert_checked.loc9_55: init %u32 = call %Int32ToUint32.ref(%.loc9_54.2)
// CHECK:STDOUT:   %.loc9_56.1: %u32 = value_of_initializer %int.convert_checked.loc9_55
// CHECK:STDOUT:   %.loc9_56.2: %u32 = converted %int.convert_checked.loc9_55, %.loc9_56.1
// CHECK:STDOUT:   %minus_one_to_u32: %u32 = bind_name minus_one_to_u32, %.loc9_56.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_negative_i32_to_u64.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %u64: type = int_type unsigned, %int_64 [template]
// CHECK:STDOUT:   %Int32ToUint64.type: type = fn_type @Int32ToUint64 [template]
// CHECK:STDOUT:   %Int32ToUint64: %Int32ToUint64.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %SubI32.type: type = fn_type @SubI32 [template]
// CHECK:STDOUT:   %SubI32: %SubI32.type = struct_value () [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Convert.3) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_0, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %iN = call %Convert.specific_fn.1(%int_0) [symbolic]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_1, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %iN = call %Convert.specific_fn.2(%int_1) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//int_ops, NegateI32, unloaded
// CHECK:STDOUT:   %import_ref.2: %SubI32.type = import_ref Main//int_ops, SubI32, loaded [template = constants.%SubI32]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//int_ops, AddU32, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//int_ops, IntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//int_ops, Int32ToInt32, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//int_ops, Int32ToUint32, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//int_ops, Uint32ToInt32, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, Uint32ToUint32, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//int_ops, IntLiteralToIntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//int_ops, Int32ToInt16, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//int_ops, Int32ToUint16, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//int_ops, Uint32ToInt16, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//int_ops, Uint32ToUint16, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//int_ops, IntLiteralToInt16, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//int_ops, IntLiteralToUint16, unloaded
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//int_ops, Int32ToInt64, unloaded
// CHECK:STDOUT:   %import_ref.17: %Int32ToUint64.type = import_ref Main//int_ops, Int32ToUint64, loaded [template = constants.%Int32ToUint64]
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//int_ops, Uint32ToInt64, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//int_ops, Uint32ToUint64, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Main//int_ops, Int32ToIntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, Uint32ToUintLiteral, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .UInt = %import_ref.22
// CHECK:STDOUT:     .ImplicitAs = %import_ref.23
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NegateI32 = imports.%import_ref.1
// CHECK:STDOUT:     .SubI32 = imports.%import_ref.2
// CHECK:STDOUT:     .AddU32 = imports.%import_ref.3
// CHECK:STDOUT:     .IntLiteral = imports.%import_ref.4
// CHECK:STDOUT:     .Int32ToInt32 = imports.%import_ref.5
// CHECK:STDOUT:     .Int32ToUint32 = imports.%import_ref.6
// CHECK:STDOUT:     .Uint32ToInt32 = imports.%import_ref.7
// CHECK:STDOUT:     .Uint32ToUint32 = imports.%import_ref.8
// CHECK:STDOUT:     .IntLiteralToIntLiteral = imports.%import_ref.9
// CHECK:STDOUT:     .Int32ToInt16 = imports.%import_ref.10
// CHECK:STDOUT:     .Int32ToUint16 = imports.%import_ref.11
// CHECK:STDOUT:     .Uint32ToInt16 = imports.%import_ref.12
// CHECK:STDOUT:     .Uint32ToUint16 = imports.%import_ref.13
// CHECK:STDOUT:     .IntLiteralToInt16 = imports.%import_ref.14
// CHECK:STDOUT:     .IntLiteralToUint16 = imports.%import_ref.15
// CHECK:STDOUT:     .Int32ToInt64 = imports.%import_ref.16
// CHECK:STDOUT:     .Int32ToUint64 = imports.%import_ref.17
// CHECK:STDOUT:     .Uint32ToInt64 = imports.%import_ref.18
// CHECK:STDOUT:     .Uint32ToUint64 = imports.%import_ref.19
// CHECK:STDOUT:     .Int32ToIntLiteral = imports.%import_ref.20
// CHECK:STDOUT:     .Uint32ToUintLiteral = imports.%import_ref.21
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .minus_one_to_u64 = @__global_init.%minus_one_to_u64
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:   %int.make_type_unsigned: init type = call constants.%UInt(%int_64) [template = constants.%u64]
// CHECK:STDOUT:   %.loc9_23.1: type = value_of_initializer %int.make_type_unsigned [template = constants.%u64]
// CHECK:STDOUT:   %.loc9_23.2: type = converted %int.make_type_unsigned, %.loc9_23.1 [template = constants.%u64]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint64(%a.param_patt: %i32) -> %u64 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubI32(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.ssub" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Int32ToUint64.ref: %Int32ToUint64.type = name_ref Int32ToUint64, imports.%import_ref.17 [template = constants.%Int32ToUint64]
// CHECK:STDOUT:   %SubI32.ref: %SubI32.type = name_ref SubI32, imports.%import_ref.2 [template = constants.%SubI32]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc9_50: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc9_50: <bound method> = bound_method %int_0, %impl.elem0.loc9_50 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc9_50: <specific function> = specific_function %Convert.bound.loc9_50, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc9_50: init %iN = call %Convert.specific_fn.loc9_50(%int_0) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc9_50.1: %i32 = value_of_initializer %int.convert_checked.loc9_50 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc9_50.2: %i32 = converted %int_0, %.loc9_50.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc9_53: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc9_53: <bound method> = bound_method %int_1, %impl.elem0.loc9_53 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc9_53: <specific function> = specific_function %Convert.bound.loc9_53, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc9_53: init %iN = call %Convert.specific_fn.loc9_53(%int_1) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc9_53.1: %i32 = value_of_initializer %int.convert_checked.loc9_53 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc9_53.2: %i32 = converted %int_1, %.loc9_53.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.ssub: init %i32 = call %SubI32.ref(%.loc9_50.2, %.loc9_53.2)
// CHECK:STDOUT:   %.loc9_54.1: %i32 = value_of_initializer %int.ssub
// CHECK:STDOUT:   %.loc9_54.2: %i32 = converted %int.ssub, %.loc9_54.1
// CHECK:STDOUT:   %int.convert_checked.loc9_55: init %u64 = call %Int32ToUint64.ref(%.loc9_54.2)
// CHECK:STDOUT:   %.loc9_56.1: %u64 = value_of_initializer %int.convert_checked.loc9_55
// CHECK:STDOUT:   %.loc9_56.2: %u64 = converted %int.convert_checked.loc9_55, %.loc9_56.1
// CHECK:STDOUT:   %minus_one_to_u64: %u64 = bind_name minus_one_to_u64, %.loc9_56.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_too_small_i32_for_i16.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i16: type = int_type signed, %int_16 [template]
// CHECK:STDOUT:   %Int32ToInt16.type: type = fn_type @Int32ToInt16 [template]
// CHECK:STDOUT:   %Int32ToInt16: %Int32ToInt16.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %NegateI32.type: type = fn_type @NegateI32 [template]
// CHECK:STDOUT:   %NegateI32: %NegateI32.type = struct_value () [template]
// CHECK:STDOUT:   %int_32769: Core.IntLiteral = int_value 32769 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Convert.3) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_32769, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init %iN = call %Convert.specific_fn(%int_32769) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %NegateI32.type = import_ref Main//int_ops, NegateI32, loaded [template = constants.%NegateI32]
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//int_ops, SubI32, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//int_ops, AddU32, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//int_ops, IntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//int_ops, Int32ToInt32, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//int_ops, Int32ToUint32, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//int_ops, Uint32ToInt32, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, Uint32ToUint32, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//int_ops, IntLiteralToIntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.10: %Int32ToInt16.type = import_ref Main//int_ops, Int32ToInt16, loaded [template = constants.%Int32ToInt16]
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//int_ops, Int32ToUint16, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//int_ops, Uint32ToInt16, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//int_ops, Uint32ToUint16, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//int_ops, IntLiteralToInt16, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//int_ops, IntLiteralToUint16, unloaded
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//int_ops, Int32ToInt64, unloaded
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//int_ops, Int32ToUint64, unloaded
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//int_ops, Uint32ToInt64, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//int_ops, Uint32ToUint64, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Main//int_ops, Int32ToIntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, Uint32ToUintLiteral, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.22
// CHECK:STDOUT:     .ImplicitAs = %import_ref.23
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NegateI32 = imports.%import_ref.1
// CHECK:STDOUT:     .SubI32 = imports.%import_ref.2
// CHECK:STDOUT:     .AddU32 = imports.%import_ref.3
// CHECK:STDOUT:     .IntLiteral = imports.%import_ref.4
// CHECK:STDOUT:     .Int32ToInt32 = imports.%import_ref.5
// CHECK:STDOUT:     .Int32ToUint32 = imports.%import_ref.6
// CHECK:STDOUT:     .Uint32ToInt32 = imports.%import_ref.7
// CHECK:STDOUT:     .Uint32ToUint32 = imports.%import_ref.8
// CHECK:STDOUT:     .IntLiteralToIntLiteral = imports.%import_ref.9
// CHECK:STDOUT:     .Int32ToInt16 = imports.%import_ref.10
// CHECK:STDOUT:     .Int32ToUint16 = imports.%import_ref.11
// CHECK:STDOUT:     .Uint32ToInt16 = imports.%import_ref.12
// CHECK:STDOUT:     .Uint32ToUint16 = imports.%import_ref.13
// CHECK:STDOUT:     .IntLiteralToInt16 = imports.%import_ref.14
// CHECK:STDOUT:     .IntLiteralToUint16 = imports.%import_ref.15
// CHECK:STDOUT:     .Int32ToInt64 = imports.%import_ref.16
// CHECK:STDOUT:     .Int32ToUint64 = imports.%import_ref.17
// CHECK:STDOUT:     .Uint32ToInt64 = imports.%import_ref.18
// CHECK:STDOUT:     .Uint32ToUint64 = imports.%import_ref.19
// CHECK:STDOUT:     .Int32ToIntLiteral = imports.%import_ref.20
// CHECK:STDOUT:     .Uint32ToUintLiteral = imports.%import_ref.21
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .min_minus_one = @__global_init.%min_minus_one
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [template = constants.%int_16]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_16) [template = constants.%i16]
// CHECK:STDOUT:   %.loc9_20.1: type = value_of_initializer %int.make_type_signed [template = constants.%i16]
// CHECK:STDOUT:   %.loc9_20.2: type = converted %int.make_type_signed, %.loc9_20.1 [template = constants.%i16]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToInt16(%a.param_patt: %i32) -> %i16 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NegateI32(%a.param_patt: %i32) -> %i32 = "int.snegate" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Int32ToInt16.ref: %Int32ToInt16.type = name_ref Int32ToInt16, imports.%import_ref.10 [template = constants.%Int32ToInt16]
// CHECK:STDOUT:   %NegateI32.ref: %NegateI32.type = name_ref NegateI32, imports.%import_ref.1 [template = constants.%NegateI32]
// CHECK:STDOUT:   %int_32769: Core.IntLiteral = int_value 32769 [template = constants.%int_32769]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_32769, %impl.elem0 [symbolic = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc9_49: init %iN = call %Convert.specific_fn(%int_32769) [symbolic = constants.%int.convert_checked]
// CHECK:STDOUT:   %.loc9_49.1: %i32 = value_of_initializer %int.convert_checked.loc9_49 [symbolic = constants.%int.convert_checked]
// CHECK:STDOUT:   %.loc9_49.2: %i32 = converted %int_32769, %.loc9_49.1 [symbolic = constants.%int.convert_checked]
// CHECK:STDOUT:   %int.snegate: init %i32 = call %NegateI32.ref(%.loc9_49.2)
// CHECK:STDOUT:   %.loc9_55.1: %i32 = value_of_initializer %int.snegate
// CHECK:STDOUT:   %.loc9_55.2: %i32 = converted %int.snegate, %.loc9_55.1
// CHECK:STDOUT:   %int.convert_checked.loc9_56: init %i16 = call %Int32ToInt16.ref(%.loc9_55.2)
// CHECK:STDOUT:   %.loc9_57.1: %i16 = value_of_initializer %int.convert_checked.loc9_56
// CHECK:STDOUT:   %.loc9_57.2: %i16 = converted %int.convert_checked.loc9_56, %.loc9_57.1
// CHECK:STDOUT:   %min_minus_one: %i16 = bind_name min_minus_one, %.loc9_57.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_not_constant.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Convert.3) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init %iN = call %Convert.specific_fn(%int_0) [symbolic]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [template]
// CHECK:STDOUT:   %i16: type = int_type signed, %int_16 [template]
// CHECK:STDOUT:   %Int32ToInt16.type: type = fn_type @Int32ToInt16 [template]
// CHECK:STDOUT:   %Int32ToInt16: %Int32ToInt16.type = struct_value () [template]
// CHECK:STDOUT:   %Int32ToInt32.type: type = fn_type @Int32ToInt32 [template]
// CHECK:STDOUT:   %Int32ToInt32: %Int32ToInt32.type = struct_value () [template]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %i64: type = int_type signed, %int_64 [template]
// CHECK:STDOUT:   %Int32ToInt64.type: type = fn_type @Int32ToInt64 [template]
// CHECK:STDOUT:   %Int32ToInt64: %Int32ToInt64.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//int_ops, NegateI32, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//int_ops, SubI32, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//int_ops, AddU32, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//int_ops, IntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.5: %Int32ToInt32.type = import_ref Main//int_ops, Int32ToInt32, loaded [template = constants.%Int32ToInt32]
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//int_ops, Int32ToUint32, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//int_ops, Uint32ToInt32, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, Uint32ToUint32, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//int_ops, IntLiteralToIntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.10: %Int32ToInt16.type = import_ref Main//int_ops, Int32ToInt16, loaded [template = constants.%Int32ToInt16]
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//int_ops, Int32ToUint16, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//int_ops, Uint32ToInt16, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//int_ops, Uint32ToUint16, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//int_ops, IntLiteralToInt16, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//int_ops, IntLiteralToUint16, unloaded
// CHECK:STDOUT:   %import_ref.16: %Int32ToInt64.type = import_ref Main//int_ops, Int32ToInt64, loaded [template = constants.%Int32ToInt64]
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//int_ops, Int32ToUint64, unloaded
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//int_ops, Uint32ToInt64, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//int_ops, Uint32ToUint64, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Main//int_ops, Int32ToIntLiteral, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, Uint32ToUintLiteral, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.22
// CHECK:STDOUT:     .ImplicitAs = %import_ref.23
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NegateI32 = imports.%import_ref.1
// CHECK:STDOUT:     .SubI32 = imports.%import_ref.2
// CHECK:STDOUT:     .AddU32 = imports.%import_ref.3
// CHECK:STDOUT:     .IntLiteral = imports.%import_ref.4
// CHECK:STDOUT:     .Int32ToInt32 = imports.%import_ref.5
// CHECK:STDOUT:     .Int32ToUint32 = imports.%import_ref.6
// CHECK:STDOUT:     .Uint32ToInt32 = imports.%import_ref.7
// CHECK:STDOUT:     .Uint32ToUint32 = imports.%import_ref.8
// CHECK:STDOUT:     .IntLiteralToIntLiteral = imports.%import_ref.9
// CHECK:STDOUT:     .Int32ToInt16 = imports.%import_ref.10
// CHECK:STDOUT:     .Int32ToUint16 = imports.%import_ref.11
// CHECK:STDOUT:     .Uint32ToInt16 = imports.%import_ref.12
// CHECK:STDOUT:     .Uint32ToUint16 = imports.%import_ref.13
// CHECK:STDOUT:     .IntLiteralToInt16 = imports.%import_ref.14
// CHECK:STDOUT:     .IntLiteralToUint16 = imports.%import_ref.15
// CHECK:STDOUT:     .Int32ToInt64 = imports.%import_ref.16
// CHECK:STDOUT:     .Int32ToUint64 = imports.%import_ref.17
// CHECK:STDOUT:     .Uint32ToInt64 = imports.%import_ref.18
// CHECK:STDOUT:     .Uint32ToUint64 = imports.%import_ref.19
// CHECK:STDOUT:     .Int32ToIntLiteral = imports.%import_ref.20
// CHECK:STDOUT:     .Uint32ToUintLiteral = imports.%import_ref.21
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .not_constant = @__global_init.%not_constant
// CHECK:STDOUT:     .convert_not_constant_narrow = @__global_init.%convert_not_constant_narrow
// CHECK:STDOUT:     .convert_not_constant_same = @__global_init.%convert_not_constant_same
// CHECK:STDOUT:     .convert_not_constant_widen = @__global_init.%convert_not_constant_widen
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %int_32.loc5: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc5: init type = call constants.%Int(%int_32.loc5) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_19.1: type = value_of_initializer %int.make_type_signed.loc5 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_19.2: type = converted %int.make_type_signed.loc5, %.loc5_19.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [template = constants.%int_16]
// CHECK:STDOUT:   %int.make_type_signed.loc15: init type = call constants.%Int(%int_16) [template = constants.%i16]
// CHECK:STDOUT:   %.loc15_34.1: type = value_of_initializer %int.make_type_signed.loc15 [template = constants.%i16]
// CHECK:STDOUT:   %.loc15_34.2: type = converted %int.make_type_signed.loc15, %.loc15_34.1 [template = constants.%i16]
// CHECK:STDOUT:   %int_32.loc25: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc25: init type = call constants.%Int(%int_32.loc25) [template = constants.%i32]
// CHECK:STDOUT:   %.loc25_32.1: type = value_of_initializer %int.make_type_signed.loc25 [template = constants.%i32]
// CHECK:STDOUT:   %.loc25_32.2: type = converted %int.make_type_signed.loc25, %.loc25_32.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:   %int.make_type_signed.loc34: init type = call constants.%Int(%int_64) [template = constants.%i64]
// CHECK:STDOUT:   %.loc34_33.1: type = value_of_initializer %int.make_type_signed.loc34 [template = constants.%i64]
// CHECK:STDOUT:   %.loc34_33.2: type = converted %int.make_type_signed.loc34, %.loc34_33.1 [template = constants.%i64]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToInt16(%a.param_patt: %i32) -> %i16 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToInt32(%a.param_patt: %i32) -> %i32 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToInt64(%a.param_patt: %i32) -> %i64 = "int.convert_checked" [from "int_ops.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0, %impl.elem0 [symbolic = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc5: init %iN = call %Convert.specific_fn(%int_0) [symbolic = constants.%int.convert_checked]
// CHECK:STDOUT:   %.loc5_26.1: %i32 = value_of_initializer %int.convert_checked.loc5 [symbolic = constants.%int.convert_checked]
// CHECK:STDOUT:   %.loc5_26.2: %i32 = converted %int_0, %.loc5_26.1 [symbolic = constants.%int.convert_checked]
// CHECK:STDOUT:   %not_constant: %i32 = bind_name not_constant, %.loc5_26.2
// CHECK:STDOUT:   %Int32ToInt16.ref: %Int32ToInt16.type = name_ref Int32ToInt16, imports.%import_ref.10 [template = constants.%Int32ToInt16]
// CHECK:STDOUT:   %not_constant.ref.loc15: %i32 = name_ref not_constant, %not_constant
// CHECK:STDOUT:   %int.convert_checked.loc15: init %i16 = call %Int32ToInt16.ref(%not_constant.ref.loc15)
// CHECK:STDOUT:   %.loc15_66.1: %i16 = value_of_initializer %int.convert_checked.loc15
// CHECK:STDOUT:   %.loc15_66.2: %i16 = converted %int.convert_checked.loc15, %.loc15_66.1
// CHECK:STDOUT:   %convert_not_constant_narrow: %i16 = bind_name convert_not_constant_narrow, %.loc15_66.2
// CHECK:STDOUT:   %Int32ToInt32.ref: %Int32ToInt32.type = name_ref Int32ToInt32, imports.%import_ref.5 [template = constants.%Int32ToInt32]
// CHECK:STDOUT:   %not_constant.ref.loc25: %i32 = name_ref not_constant, %not_constant
// CHECK:STDOUT:   %int.convert_checked.loc25: init %i32 = call %Int32ToInt32.ref(%not_constant.ref.loc25)
// CHECK:STDOUT:   %.loc25_64.1: %i32 = value_of_initializer %int.convert_checked.loc25
// CHECK:STDOUT:   %.loc25_64.2: %i32 = converted %int.convert_checked.loc25, %.loc25_64.1
// CHECK:STDOUT:   %convert_not_constant_same: %i32 = bind_name convert_not_constant_same, %.loc25_64.2
// CHECK:STDOUT:   %Int32ToInt64.ref: %Int32ToInt64.type = name_ref Int32ToInt64, imports.%import_ref.16 [template = constants.%Int32ToInt64]
// CHECK:STDOUT:   %not_constant.ref.loc34: %i32 = name_ref not_constant, %not_constant
// CHECK:STDOUT:   %int.convert_checked.loc34: init %i64 = call %Int32ToInt64.ref(%not_constant.ref.loc34)
// CHECK:STDOUT:   %.loc34_65.1: %i64 = value_of_initializer %int.convert_checked.loc34
// CHECK:STDOUT:   %.loc34_65.2: %i64 = converted %int.convert_checked.loc34, %.loc34_65.1
// CHECK:STDOUT:   %convert_not_constant_widen: %i64 = bind_name convert_not_constant_widen, %.loc34_65.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
