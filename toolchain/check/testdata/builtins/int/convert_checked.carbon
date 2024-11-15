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
// CHECK:STDERR:                         ^~~~~~~~~~~~~
// CHECK:STDERR:
let max_plus_one: i16 = Int32ToInt16(0x8000);

// --- fail_too_large_i32_for_u16.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: fail_too_large_i32_for_u16.carbon:[[@LINE+4]]:25: error: integer value 65536 too large for type `u16` [IntTooLargeForType]
// CHECK:STDERR: let max_plus_one: u16 = Int32ToUint16(0x1_0000);
// CHECK:STDERR:                         ^~~~~~~~~~~~~~
// CHECK:STDERR:
let max_plus_one: u16 = Int32ToUint16(0x1_0000);

// --- fail_too_large_u32_for_i16.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: fail_too_large_u32_for_i16.carbon:[[@LINE+4]]:25: error: integer value 32768 too large for type `i16` [IntTooLargeForType]
// CHECK:STDERR: let max_plus_one: i16 = Uint32ToInt16(Int32ToUint32(0x8000));
// CHECK:STDERR:                         ^~~~~~~~~~~~~~
// CHECK:STDERR:
let max_plus_one: i16 = Uint32ToInt16(Int32ToUint32(0x8000));

// --- fail_too_large_u32_for_u16.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: fail_too_large_u32_for_u16.carbon:[[@LINE+4]]:25: error: integer value 65536 too large for type `u16` [IntTooLargeForType]
// CHECK:STDERR: let max_plus_one: u16 = Uint32ToUint16(Int32ToUint32(0x1_0000));
// CHECK:STDERR:                         ^~~~~~~~~~~~~~~
// CHECK:STDERR:
let max_plus_one: u16 = Uint32ToUint16(Int32ToUint32(0x1_0000));

// --- fail_negative_i32_to_u16.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: fail_negative_i32_to_u16.carbon:[[@LINE+4]]:29: error: negative integer value -1 converted to unsigned type `u16` [NegativeIntInUnsignedType]
// CHECK:STDERR: let minus_one_to_u16: u16 = Int32ToUint16(SubI32(0, 1));
// CHECK:STDERR:                             ^~~~~~~~~~~~~~
// CHECK:STDERR:
let minus_one_to_u16: u16 = Int32ToUint16(SubI32(0, 1));

// --- fail_negative_i32_to_u32.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: fail_negative_i32_to_u32.carbon:[[@LINE+4]]:29: error: negative integer value -1 converted to unsigned type `u32` [NegativeIntInUnsignedType]
// CHECK:STDERR: let minus_one_to_u32: u32 = Int32ToUint32(SubI32(0, 1));
// CHECK:STDERR:                             ^~~~~~~~~~~~~~
// CHECK:STDERR:
let minus_one_to_u32: u32 = Int32ToUint32(SubI32(0, 1));

// --- fail_negative_i32_to_u64.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: fail_negative_i32_to_u64.carbon:[[@LINE+4]]:29: error: negative integer value -1 converted to unsigned type `u64` [NegativeIntInUnsignedType]
// CHECK:STDERR: let minus_one_to_u64: u64 = Int32ToUint64(SubI32(0, 1));
// CHECK:STDERR:                             ^~~~~~~~~~~~~~
// CHECK:STDERR:
let minus_one_to_u64: u64 = Int32ToUint64(SubI32(0, 1));

// --- fail_too_small_i32_for_i16.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

// CHECK:STDERR: fail_too_small_i32_for_i16.carbon:[[@LINE+4]]:26: error: integer value -32769 too large for type `i16` [IntTooLargeForType]
// CHECK:STDERR: let min_minus_one: i16 = Int32ToInt16(NegateI32(0x8001));
// CHECK:STDERR:                          ^~~~~~~~~~~~~
// CHECK:STDERR:
let min_minus_one: i16 = Int32ToInt16(NegateI32(0x8001));

// --- fail_not_constant.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

let not_constant: i32 = 0;

// CHECK:STDERR: fail_not_constant.carbon:[[@LINE+8]]:40: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let convert_not_constant_narrow: i16 = Int32ToInt16(not_constant);
// CHECK:STDERR:                                        ^~~~~~~~~~~~~
// CHECK:STDERR: fail_not_constant.carbon:[[@LINE-7]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:18:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn Int32ToInt16(a: i32) -> i16 = "int.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let convert_not_constant_narrow: i16 = Int32ToInt16(not_constant);

// CHECK:STDERR: fail_not_constant.carbon:[[@LINE+8]]:38: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let convert_not_constant_same: i32 = Int32ToInt32(not_constant);
// CHECK:STDERR:                                      ^~~~~~~~~~~~~
// CHECK:STDERR: fail_not_constant.carbon:[[@LINE-17]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:10:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn Int32ToInt32(a: i32) -> i32 = "int.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let convert_not_constant_same: i32 = Int32ToInt32(not_constant);

// CHECK:STDERR: fail_not_constant.carbon:[[@LINE+7]]:39: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let convert_not_constant_widen: i64 = Int32ToInt64(not_constant);
// CHECK:STDERR:                                       ^~~~~~~~~~~~~
// CHECK:STDERR: fail_not_constant.carbon:[[@LINE-27]]:1: in import [InImport]
// CHECK:STDERR: int_ops.carbon:26:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn Int32ToInt64(a: i32) -> i64 = "int.convert_checked";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
let convert_not_constant_widen: i64 = Int32ToInt64(not_constant);

// CHECK:STDOUT: --- int_ops.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %NegateI32.type: type = fn_type @NegateI32 [template]
// CHECK:STDOUT:   %NegateI32: %NegateI32.type = struct_value () [template]
// CHECK:STDOUT:   %SubI32.type: type = fn_type @SubI32 [template]
// CHECK:STDOUT:   %SubI32: %SubI32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type unsigned, %.1 [template]
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
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 16 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = int_type signed, %.3 [template]
// CHECK:STDOUT:   %Int32ToInt16.type: type = fn_type @Int32ToInt16 [template]
// CHECK:STDOUT:   %Int32ToInt16: %Int32ToInt16.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = int_type unsigned, %.3 [template]
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
// CHECK:STDOUT:   %.6: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %.7: type = int_type signed, %.6 [template]
// CHECK:STDOUT:   %Int32ToInt64.type: type = fn_type @Int32ToInt64 [template]
// CHECK:STDOUT:   %Int32ToInt64: %Int32ToInt64.type = struct_value () [template]
// CHECK:STDOUT:   %.8: type = int_type unsigned, %.6 [template]
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
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .UInt = %import_ref.2
// CHECK:STDOUT:     .Int = %import_ref.3
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
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc4_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_17.1: type = value_of_initializer %int.make_type_32.loc4_17 [template = i32]
// CHECK:STDOUT:     %.loc4_17.2: type = converted %int.make_type_32.loc4_17, %.loc4_17.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc4_25: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_25.1: type = value_of_initializer %int.make_type_32.loc4_25 [template = i32]
// CHECK:STDOUT:     %.loc4_25.2: type = converted %int.make_type_32.loc4_25, %.loc4_25.1 [template = i32]
// CHECK:STDOUT:     %a.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SubI32.decl: %SubI32.type = fn_decl @SubI32 [template = constants.%SubI32] {
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc5_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_14.1: type = value_of_initializer %int.make_type_32.loc5_14 [template = i32]
// CHECK:STDOUT:     %.loc5_14.2: type = converted %int.make_type_32.loc5_14, %.loc5_14.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc5_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_22.1: type = value_of_initializer %int.make_type_32.loc5_22 [template = i32]
// CHECK:STDOUT:     %.loc5_22.2: type = converted %int.make_type_32.loc5_22, %.loc5_22.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc5_30: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_30.1: type = value_of_initializer %int.make_type_32.loc5_30 [template = i32]
// CHECK:STDOUT:     %.loc5_30.2: type = converted %int.make_type_32.loc5_30, %.loc5_30.1 [template = i32]
// CHECK:STDOUT:     %a.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddU32.decl: %AddU32.type = fn_decl @AddU32 [template = constants.%AddU32] {
// CHECK:STDOUT:     %a.patt: %.2 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %.2 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %.2 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %.2 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.2 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_14.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_unsigned.loc6_14: init type = call constants.%UInt(%.loc6_14.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc6_14.2: type = value_of_initializer %int.make_type_unsigned.loc6_14 [template = constants.%.2]
// CHECK:STDOUT:     %.loc6_14.3: type = converted %int.make_type_unsigned.loc6_14, %.loc6_14.2 [template = constants.%.2]
// CHECK:STDOUT:     %.loc6_22.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_unsigned.loc6_22: init type = call constants.%UInt(%.loc6_22.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc6_22.2: type = value_of_initializer %int.make_type_unsigned.loc6_22 [template = constants.%.2]
// CHECK:STDOUT:     %.loc6_22.3: type = converted %int.make_type_unsigned.loc6_22, %.loc6_22.2 [template = constants.%.2]
// CHECK:STDOUT:     %.loc6_30.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_unsigned.loc6_30: init type = call constants.%UInt(%.loc6_30.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc6_30.2: type = value_of_initializer %int.make_type_unsigned.loc6_30 [template = constants.%.2]
// CHECK:STDOUT:     %.loc6_30.3: type = converted %int.make_type_unsigned.loc6_30, %.loc6_30.2 [template = constants.%.2]
// CHECK:STDOUT:     %a.param: %.2 = value_param runtime_param0
// CHECK:STDOUT:     %a: %.2 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %.2 = value_param runtime_param1
// CHECK:STDOUT:     %b: %.2 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %.2 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntLiteral.decl: %IntLiteral.type = fn_decl @IntLiteral [template = constants.%IntLiteral] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int32ToInt32.decl: %Int32ToInt32.type = fn_decl @Int32ToInt32 [template = constants.%Int32ToInt32] {
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc10_20: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc10_20.1: type = value_of_initializer %int.make_type_32.loc10_20 [template = i32]
// CHECK:STDOUT:     %.loc10_20.2: type = converted %int.make_type_32.loc10_20, %.loc10_20.1 [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc10_28: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc10_28.1: type = value_of_initializer %int.make_type_32.loc10_28 [template = i32]
// CHECK:STDOUT:     %.loc10_28.2: type = converted %int.make_type_32.loc10_28, %.loc10_28.1 [template = i32]
// CHECK:STDOUT:     %a.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int32ToUint32.decl: %Int32ToUint32.type = fn_decl @Int32ToUint32 [template = constants.%Int32ToUint32] {
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.2 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_21.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc11_21.2: type = converted %int.make_type_32, %.loc11_21.1 [template = i32]
// CHECK:STDOUT:     %.loc11_29.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_unsigned: init type = call constants.%UInt(%.loc11_29.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_29.2: type = value_of_initializer %int.make_type_unsigned [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_29.3: type = converted %int.make_type_unsigned, %.loc11_29.2 [template = constants.%.2]
// CHECK:STDOUT:     %a.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %.2 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Uint32ToInt32.decl: %Uint32ToInt32.type = fn_decl @Uint32ToInt32 [template = constants.%Uint32ToInt32] {
// CHECK:STDOUT:     %a.patt: %.2 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %.2 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_21.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_unsigned: init type = call constants.%UInt(%.loc12_21.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc12_21.2: type = value_of_initializer %int.make_type_unsigned [template = constants.%.2]
// CHECK:STDOUT:     %.loc12_21.3: type = converted %int.make_type_unsigned, %.loc12_21.2 [template = constants.%.2]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc12_29.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc12_29.2: type = converted %int.make_type_32, %.loc12_29.1 [template = i32]
// CHECK:STDOUT:     %a.param: %.2 = value_param runtime_param0
// CHECK:STDOUT:     %a: %.2 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Uint32ToUint32.decl: %Uint32ToUint32.type = fn_decl @Uint32ToUint32 [template = constants.%Uint32ToUint32] {
// CHECK:STDOUT:     %a.patt: %.2 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %.2 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.2 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_22.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_unsigned.loc13_22: init type = call constants.%UInt(%.loc13_22.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc13_22.2: type = value_of_initializer %int.make_type_unsigned.loc13_22 [template = constants.%.2]
// CHECK:STDOUT:     %.loc13_22.3: type = converted %int.make_type_unsigned.loc13_22, %.loc13_22.2 [template = constants.%.2]
// CHECK:STDOUT:     %.loc13_30.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_unsigned.loc13_30: init type = call constants.%UInt(%.loc13_30.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc13_30.2: type = value_of_initializer %int.make_type_unsigned.loc13_30 [template = constants.%.2]
// CHECK:STDOUT:     %.loc13_30.3: type = converted %int.make_type_unsigned.loc13_30, %.loc13_30.2 [template = constants.%.2]
// CHECK:STDOUT:     %a.param: %.2 = value_param runtime_param0
// CHECK:STDOUT:     %a: %.2 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %.2 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntLiteralToIntLiteral.decl: %IntLiteralToIntLiteral.type = fn_decl @IntLiteralToIntLiteral [template = constants.%IntLiteralToIntLiteral] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref.loc14_30: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc14_40: init type = call %IntLiteral.ref.loc14_30() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc14_41.1: type = value_of_initializer %int_literal.make_type.loc14_40 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc14_41.2: type = converted %int_literal.make_type.loc14_40, %.loc14_41.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %IntLiteral.ref.loc14_47: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc14_57: init type = call %IntLiteral.ref.loc14_47() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc14_58.1: type = value_of_initializer %int_literal.make_type.loc14_57 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc14_58.2: type = converted %int_literal.make_type.loc14_57, %.loc14_58.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int32ToInt16.decl: %Int32ToInt16.type = fn_decl @Int32ToInt16 [template = constants.%Int32ToInt16] {
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.4 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.4 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc18_20.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc18_20.2: type = converted %int.make_type_32, %.loc18_20.1 [template = i32]
// CHECK:STDOUT:     %.loc18_28.1: Core.IntLiteral = int_value 16 [template = constants.%.3]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc18_28.1) [template = constants.%.4]
// CHECK:STDOUT:     %.loc18_28.2: type = value_of_initializer %int.make_type_signed [template = constants.%.4]
// CHECK:STDOUT:     %.loc18_28.3: type = converted %int.make_type_signed, %.loc18_28.2 [template = constants.%.4]
// CHECK:STDOUT:     %a.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %.4 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.4 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int32ToUint16.decl: %Int32ToUint16.type = fn_decl @Int32ToUint16 [template = constants.%Int32ToUint16] {
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.5 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.5 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc19_21.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc19_21.2: type = converted %int.make_type_32, %.loc19_21.1 [template = i32]
// CHECK:STDOUT:     %.loc19_29.1: Core.IntLiteral = int_value 16 [template = constants.%.3]
// CHECK:STDOUT:     %int.make_type_unsigned: init type = call constants.%UInt(%.loc19_29.1) [template = constants.%.5]
// CHECK:STDOUT:     %.loc19_29.2: type = value_of_initializer %int.make_type_unsigned [template = constants.%.5]
// CHECK:STDOUT:     %.loc19_29.3: type = converted %int.make_type_unsigned, %.loc19_29.2 [template = constants.%.5]
// CHECK:STDOUT:     %a.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %.5 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.5 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Uint32ToInt16.decl: %Uint32ToInt16.type = fn_decl @Uint32ToInt16 [template = constants.%Uint32ToInt16] {
// CHECK:STDOUT:     %a.patt: %.2 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %.2 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.4 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.4 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc20_21.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_unsigned: init type = call constants.%UInt(%.loc20_21.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc20_21.2: type = value_of_initializer %int.make_type_unsigned [template = constants.%.2]
// CHECK:STDOUT:     %.loc20_21.3: type = converted %int.make_type_unsigned, %.loc20_21.2 [template = constants.%.2]
// CHECK:STDOUT:     %.loc20_29.1: Core.IntLiteral = int_value 16 [template = constants.%.3]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc20_29.1) [template = constants.%.4]
// CHECK:STDOUT:     %.loc20_29.2: type = value_of_initializer %int.make_type_signed [template = constants.%.4]
// CHECK:STDOUT:     %.loc20_29.3: type = converted %int.make_type_signed, %.loc20_29.2 [template = constants.%.4]
// CHECK:STDOUT:     %a.param: %.2 = value_param runtime_param0
// CHECK:STDOUT:     %a: %.2 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %.4 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.4 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Uint32ToUint16.decl: %Uint32ToUint16.type = fn_decl @Uint32ToUint16 [template = constants.%Uint32ToUint16] {
// CHECK:STDOUT:     %a.patt: %.2 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %.2 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.5 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.5 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc21_22.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_unsigned.loc21_22: init type = call constants.%UInt(%.loc21_22.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc21_22.2: type = value_of_initializer %int.make_type_unsigned.loc21_22 [template = constants.%.2]
// CHECK:STDOUT:     %.loc21_22.3: type = converted %int.make_type_unsigned.loc21_22, %.loc21_22.2 [template = constants.%.2]
// CHECK:STDOUT:     %.loc21_30.1: Core.IntLiteral = int_value 16 [template = constants.%.3]
// CHECK:STDOUT:     %int.make_type_unsigned.loc21_30: init type = call constants.%UInt(%.loc21_30.1) [template = constants.%.5]
// CHECK:STDOUT:     %.loc21_30.2: type = value_of_initializer %int.make_type_unsigned.loc21_30 [template = constants.%.5]
// CHECK:STDOUT:     %.loc21_30.3: type = converted %int.make_type_unsigned.loc21_30, %.loc21_30.2 [template = constants.%.5]
// CHECK:STDOUT:     %a.param: %.2 = value_param runtime_param0
// CHECK:STDOUT:     %a: %.2 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %.5 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.5 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntLiteralToInt16.decl: %IntLiteralToInt16.type = fn_decl @IntLiteralToInt16 [template = constants.%IntLiteralToInt16] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.4 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.4 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc22_36.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc22_36.2: type = converted %int_literal.make_type, %.loc22_36.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc22_42.1: Core.IntLiteral = int_value 16 [template = constants.%.3]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc22_42.1) [template = constants.%.4]
// CHECK:STDOUT:     %.loc22_42.2: type = value_of_initializer %int.make_type_signed [template = constants.%.4]
// CHECK:STDOUT:     %.loc22_42.3: type = converted %int.make_type_signed, %.loc22_42.2 [template = constants.%.4]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %.4 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.4 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntLiteralToUint16.decl: %IntLiteralToUint16.type = fn_decl @IntLiteralToUint16 [template = constants.%IntLiteralToUint16] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.5 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.5 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc23_37.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc23_37.2: type = converted %int_literal.make_type, %.loc23_37.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc23_43.1: Core.IntLiteral = int_value 16 [template = constants.%.3]
// CHECK:STDOUT:     %int.make_type_unsigned: init type = call constants.%UInt(%.loc23_43.1) [template = constants.%.5]
// CHECK:STDOUT:     %.loc23_43.2: type = value_of_initializer %int.make_type_unsigned [template = constants.%.5]
// CHECK:STDOUT:     %.loc23_43.3: type = converted %int.make_type_unsigned, %.loc23_43.2 [template = constants.%.5]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %.5 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.5 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int32ToInt64.decl: %Int32ToInt64.type = fn_decl @Int32ToInt64 [template = constants.%Int32ToInt64] {
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.7 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.7 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc26_20.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc26_20.2: type = converted %int.make_type_32, %.loc26_20.1 [template = i32]
// CHECK:STDOUT:     %.loc26_28.1: Core.IntLiteral = int_value 64 [template = constants.%.6]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc26_28.1) [template = constants.%.7]
// CHECK:STDOUT:     %.loc26_28.2: type = value_of_initializer %int.make_type_signed [template = constants.%.7]
// CHECK:STDOUT:     %.loc26_28.3: type = converted %int.make_type_signed, %.loc26_28.2 [template = constants.%.7]
// CHECK:STDOUT:     %a.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %.7 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.7 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int32ToUint64.decl: %Int32ToUint64.type = fn_decl @Int32ToUint64 [template = constants.%Int32ToUint64] {
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.8 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.8 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc27_21.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc27_21.2: type = converted %int.make_type_32, %.loc27_21.1 [template = i32]
// CHECK:STDOUT:     %.loc27_29.1: Core.IntLiteral = int_value 64 [template = constants.%.6]
// CHECK:STDOUT:     %int.make_type_unsigned: init type = call constants.%UInt(%.loc27_29.1) [template = constants.%.8]
// CHECK:STDOUT:     %.loc27_29.2: type = value_of_initializer %int.make_type_unsigned [template = constants.%.8]
// CHECK:STDOUT:     %.loc27_29.3: type = converted %int.make_type_unsigned, %.loc27_29.2 [template = constants.%.8]
// CHECK:STDOUT:     %a.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %.8 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.8 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Uint32ToInt64.decl: %Uint32ToInt64.type = fn_decl @Uint32ToInt64 [template = constants.%Uint32ToInt64] {
// CHECK:STDOUT:     %a.patt: %.2 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %.2 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.7 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.7 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc28_21.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_unsigned: init type = call constants.%UInt(%.loc28_21.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc28_21.2: type = value_of_initializer %int.make_type_unsigned [template = constants.%.2]
// CHECK:STDOUT:     %.loc28_21.3: type = converted %int.make_type_unsigned, %.loc28_21.2 [template = constants.%.2]
// CHECK:STDOUT:     %.loc28_29.1: Core.IntLiteral = int_value 64 [template = constants.%.6]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc28_29.1) [template = constants.%.7]
// CHECK:STDOUT:     %.loc28_29.2: type = value_of_initializer %int.make_type_signed [template = constants.%.7]
// CHECK:STDOUT:     %.loc28_29.3: type = converted %int.make_type_signed, %.loc28_29.2 [template = constants.%.7]
// CHECK:STDOUT:     %a.param: %.2 = value_param runtime_param0
// CHECK:STDOUT:     %a: %.2 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %.7 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.7 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Uint32ToUint64.decl: %Uint32ToUint64.type = fn_decl @Uint32ToUint64 [template = constants.%Uint32ToUint64] {
// CHECK:STDOUT:     %a.patt: %.2 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %.2 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.8 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.8 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc29_22.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_unsigned.loc29_22: init type = call constants.%UInt(%.loc29_22.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc29_22.2: type = value_of_initializer %int.make_type_unsigned.loc29_22 [template = constants.%.2]
// CHECK:STDOUT:     %.loc29_22.3: type = converted %int.make_type_unsigned.loc29_22, %.loc29_22.2 [template = constants.%.2]
// CHECK:STDOUT:     %.loc29_30.1: Core.IntLiteral = int_value 64 [template = constants.%.6]
// CHECK:STDOUT:     %int.make_type_unsigned.loc29_30: init type = call constants.%UInt(%.loc29_30.1) [template = constants.%.8]
// CHECK:STDOUT:     %.loc29_30.2: type = value_of_initializer %int.make_type_unsigned.loc29_30 [template = constants.%.8]
// CHECK:STDOUT:     %.loc29_30.3: type = converted %int.make_type_unsigned.loc29_30, %.loc29_30.2 [template = constants.%.8]
// CHECK:STDOUT:     %a.param: %.2 = value_param runtime_param0
// CHECK:STDOUT:     %a: %.2 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %.8 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.8 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int32ToIntLiteral.decl: %Int32ToIntLiteral.type = fn_decl @Int32ToIntLiteral [template = constants.%Int32ToIntLiteral] {
// CHECK:STDOUT:     %a.patt: i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc30_25.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc30_25.2: type = converted %int.make_type_32, %.loc30_25.1 [template = i32]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc30_44.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc30_44.2: type = converted %int_literal.make_type, %.loc30_44.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Uint32ToUintLiteral.decl: %Uint32ToUintLiteral.type = fn_decl @Uint32ToUintLiteral [template = constants.%Uint32ToUintLiteral] {
// CHECK:STDOUT:     %a.patt: %.2 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %.2 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc31_27.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_unsigned: init type = call constants.%UInt(%.loc31_27.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc31_27.2: type = value_of_initializer %int.make_type_unsigned [template = constants.%.2]
// CHECK:STDOUT:     %.loc31_27.3: type = converted %int.make_type_unsigned, %.loc31_27.2 [template = constants.%.2]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc31_46.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc31_46.2: type = converted %int_literal.make_type, %.loc31_46.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: %.2 = value_param runtime_param0
// CHECK:STDOUT:     %a: %.2 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NegateI32(%a.param_patt: i32) -> i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubI32(%a.param_patt: i32, %b.param_patt: i32) -> i32 = "int.ssub";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddU32(%a.param_patt: %.2, %b.param_patt: %.2) -> %.2 = "int.uadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToInt32(%a.param_patt: i32) -> i32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint32(%a.param_patt: i32) -> %.2 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToInt32(%a.param_patt: %.2) -> i32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToUint32(%a.param_patt: %.2) -> %.2 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteralToIntLiteral(%a.param_patt: Core.IntLiteral) -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToInt16(%a.param_patt: i32) -> %.4 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint16(%a.param_patt: i32) -> %.5 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToInt16(%a.param_patt: %.2) -> %.4 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToUint16(%a.param_patt: %.2) -> %.5 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteralToInt16(%a.param_patt: Core.IntLiteral) -> %.4 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteralToUint16(%a.param_patt: Core.IntLiteral) -> %.5 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToInt64(%a.param_patt: i32) -> %.7 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint64(%a.param_patt: i32) -> %.8 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToInt64(%a.param_patt: %.2) -> %.7 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToUint64(%a.param_patt: %.2) -> %.8 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToIntLiteral(%a.param_patt: i32) -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToUintLiteral(%a.param_patt: %.2) -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: --- identity.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Int32ToInt32.type: type = fn_type @Int32ToInt32 [template]
// CHECK:STDOUT:   %Int32ToInt32: %Int32ToInt32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: i32 = int_value 0 [template]
// CHECK:STDOUT:   %.2: i32 = int_value 2147483647 [template]
// CHECK:STDOUT:   %SubI32.type: type = fn_type @SubI32 [template]
// CHECK:STDOUT:   %SubI32: %SubI32.type = struct_value () [template]
// CHECK:STDOUT:   %NegateI32.type: type = fn_type @NegateI32 [template]
// CHECK:STDOUT:   %NegateI32: %NegateI32.type = struct_value () [template]
// CHECK:STDOUT:   %.3: i32 = int_value -2147483647 [template]
// CHECK:STDOUT:   %.4: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_value -2147483648 [template]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %IntLiteralToIntLiteral.type: type = fn_type @IntLiteralToIntLiteral [template]
// CHECK:STDOUT:   %IntLiteralToIntLiteral: %IntLiteralToIntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %Int32ToIntLiteral.type: type = fn_type @Int32ToIntLiteral [template]
// CHECK:STDOUT:   %Int32ToIntLiteral: %Int32ToIntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %.6: i32 = int_value -1 [template]
// CHECK:STDOUT:   %.7: Core.IntLiteral = int_value -1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %NegateI32.type = import_ref Main//int_ops, inst+25, loaded [template = constants.%NegateI32]
// CHECK:STDOUT:   %import_ref.2: %SubI32.type = import_ref Main//int_ops, inst+49, loaded [template = constants.%SubI32]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//int_ops, inst+89, unloaded
// CHECK:STDOUT:   %import_ref.4: %IntLiteral.type = import_ref Main//int_ops, inst+96, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT:   %import_ref.5: %Int32ToInt32.type = import_ref Main//int_ops, inst+113, loaded [template = constants.%Int32ToInt32]
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//int_ops, inst+131, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//int_ops, inst+149, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, inst+168, unloaded
// CHECK:STDOUT:   %import_ref.9: %IntLiteralToIntLiteral.type = import_ref Main//int_ops, inst+187, loaded [template = constants.%IntLiteralToIntLiteral]
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//int_ops, inst+218, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//int_ops, inst+237, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//int_ops, inst+256, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//int_ops, inst+275, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//int_ops, inst+294, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//int_ops, inst+313, unloaded
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//int_ops, inst+333, unloaded
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//int_ops, inst+352, unloaded
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//int_ops, inst+371, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//int_ops, inst+390, unloaded
// CHECK:STDOUT:   %import_ref.20: %Int32ToIntLiteral.type = import_ref Main//int_ops, inst+408, loaded [template = constants.%Int32ToIntLiteral]
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, inst+427, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.22
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
// CHECK:STDOUT:   %int.make_type_32.loc5: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_8.1: type = value_of_initializer %int.make_type_32.loc5 [template = i32]
// CHECK:STDOUT:   %.loc5_8.2: type = converted %int.make_type_32.loc5, %.loc5_8.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc6: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc6_8.1: type = value_of_initializer %int.make_type_32.loc6 [template = i32]
// CHECK:STDOUT:   %.loc6_8.2: type = converted %int.make_type_32.loc6, %.loc6_8.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc7: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc7_8.1: type = value_of_initializer %int.make_type_32.loc7 [template = i32]
// CHECK:STDOUT:   %.loc7_8.2: type = converted %int.make_type_32.loc7, %.loc7_8.1 [template = i32]
// CHECK:STDOUT:   %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.4 [template = constants.%IntLiteral]
// CHECK:STDOUT:   %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:   %.loc8_19.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:   %.loc8_19.2: type = converted %int_literal.make_type, %.loc8_19.1 [template = Core.IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToInt32(%a.param_patt: i32) -> i32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubI32(%a.param_patt: i32, %b.param_patt: i32) -> i32 = "int.ssub";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NegateI32(%a.param_patt: i32) -> i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteralToIntLiteral(%a.param_patt: Core.IntLiteral) -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToIntLiteral(%a.param_patt: i32) -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Int32ToInt32.ref.loc5: %Int32ToInt32.type = name_ref Int32ToInt32, imports.%import_ref.5 [template = constants.%Int32ToInt32]
// CHECK:STDOUT:   %.loc5_27: i32 = int_value 0 [template = constants.%.1]
// CHECK:STDOUT:   %int.convert_checked.loc5: init i32 = call %Int32ToInt32.ref.loc5(%.loc5_27) [template = constants.%.1]
// CHECK:STDOUT:   %.loc5_29.1: i32 = value_of_initializer %int.convert_checked.loc5 [template = constants.%.1]
// CHECK:STDOUT:   %.loc5_29.2: i32 = converted %int.convert_checked.loc5, %.loc5_29.1 [template = constants.%.1]
// CHECK:STDOUT:   %a: i32 = bind_name a, %.loc5_29.2
// CHECK:STDOUT:   %Int32ToInt32.ref.loc6: %Int32ToInt32.type = name_ref Int32ToInt32, imports.%import_ref.5 [template = constants.%Int32ToInt32]
// CHECK:STDOUT:   %.loc6_27: i32 = int_value 2147483647 [template = constants.%.2]
// CHECK:STDOUT:   %int.convert_checked.loc6: init i32 = call %Int32ToInt32.ref.loc6(%.loc6_27) [template = constants.%.2]
// CHECK:STDOUT:   %.loc6_39.1: i32 = value_of_initializer %int.convert_checked.loc6 [template = constants.%.2]
// CHECK:STDOUT:   %.loc6_39.2: i32 = converted %int.convert_checked.loc6, %.loc6_39.1 [template = constants.%.2]
// CHECK:STDOUT:   %b: i32 = bind_name b, %.loc6_39.2
// CHECK:STDOUT:   %Int32ToInt32.ref.loc7: %Int32ToInt32.type = name_ref Int32ToInt32, imports.%import_ref.5 [template = constants.%Int32ToInt32]
// CHECK:STDOUT:   %SubI32.ref: %SubI32.type = name_ref SubI32, imports.%import_ref.2 [template = constants.%SubI32]
// CHECK:STDOUT:   %NegateI32.ref.loc7: %NegateI32.type = name_ref NegateI32, imports.%import_ref.1 [template = constants.%NegateI32]
// CHECK:STDOUT:   %.loc7_44: i32 = int_value 2147483647 [template = constants.%.2]
// CHECK:STDOUT:   %int.snegate.loc7: init i32 = call %NegateI32.ref.loc7(%.loc7_44) [template = constants.%.3]
// CHECK:STDOUT:   %.loc7_58: i32 = int_value 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_43.1: i32 = value_of_initializer %int.snegate.loc7 [template = constants.%.3]
// CHECK:STDOUT:   %.loc7_43.2: i32 = converted %int.snegate.loc7, %.loc7_43.1 [template = constants.%.3]
// CHECK:STDOUT:   %int.ssub: init i32 = call %SubI32.ref(%.loc7_43.2, %.loc7_58) [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_33.1: i32 = value_of_initializer %int.ssub [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_33.2: i32 = converted %int.ssub, %.loc7_33.1 [template = constants.%.5]
// CHECK:STDOUT:   %int.convert_checked.loc7: init i32 = call %Int32ToInt32.ref.loc7(%.loc7_33.2) [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_61.1: i32 = value_of_initializer %int.convert_checked.loc7 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_61.2: i32 = converted %int.convert_checked.loc7, %.loc7_61.1 [template = constants.%.5]
// CHECK:STDOUT:   %c: i32 = bind_name c, %.loc7_61.2
// CHECK:STDOUT:   %IntLiteralToIntLiteral.ref: %IntLiteralToIntLiteral.type = name_ref IntLiteralToIntLiteral, imports.%import_ref.9 [template = constants.%IntLiteralToIntLiteral]
// CHECK:STDOUT:   %Int32ToIntLiteral.ref: %Int32ToIntLiteral.type = name_ref Int32ToIntLiteral, imports.%import_ref.20 [template = constants.%Int32ToIntLiteral]
// CHECK:STDOUT:   %NegateI32.ref.loc8: %NegateI32.type = name_ref NegateI32, imports.%import_ref.1 [template = constants.%NegateI32]
// CHECK:STDOUT:   %.loc8_74: i32 = int_value 1 [template = constants.%.4]
// CHECK:STDOUT:   %int.snegate.loc8: init i32 = call %NegateI32.ref.loc8(%.loc8_74) [template = constants.%.6]
// CHECK:STDOUT:   %.loc8_73.1: i32 = value_of_initializer %int.snegate.loc8 [template = constants.%.6]
// CHECK:STDOUT:   %.loc8_73.2: i32 = converted %int.snegate.loc8, %.loc8_73.1 [template = constants.%.6]
// CHECK:STDOUT:   %int.convert_checked.loc8_63: init Core.IntLiteral = call %Int32ToIntLiteral.ref(%.loc8_73.2) [template = constants.%.7]
// CHECK:STDOUT:   %.loc8_63.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc8_63 [template = constants.%.7]
// CHECK:STDOUT:   %.loc8_63.2: Core.IntLiteral = converted %int.convert_checked.loc8_63, %.loc8_63.1 [template = constants.%.7]
// CHECK:STDOUT:   %int.convert_checked.loc8_45: init Core.IntLiteral = call %IntLiteralToIntLiteral.ref(%.loc8_63.2) [template = constants.%.7]
// CHECK:STDOUT:   %.loc8_78.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc8_45 [template = constants.%.7]
// CHECK:STDOUT:   %.loc8_78.2: Core.IntLiteral = converted %int.convert_checked.loc8_45, %.loc8_78.1 [template = constants.%.7]
// CHECK:STDOUT:   %d: Core.IntLiteral = bind_name d, %.loc8_78.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- same_size.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type unsigned, %.1 [template]
// CHECK:STDOUT:   %Int32ToUint32.type: type = fn_type @Int32ToUint32 [template]
// CHECK:STDOUT:   %Int32ToUint32: %Int32ToUint32.type = struct_value () [template]
// CHECK:STDOUT:   %.3: i32 = int_value 2147483647 [template]
// CHECK:STDOUT:   %.4: %.2 = int_value 2147483647 [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Uint32ToInt32.type: type = fn_type @Uint32ToInt32 [template]
// CHECK:STDOUT:   %Uint32ToInt32: %Uint32ToInt32.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//int_ops, inst+25, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//int_ops, inst+49, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//int_ops, inst+89, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//int_ops, inst+96, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//int_ops, inst+113, unloaded
// CHECK:STDOUT:   %import_ref.6: %Int32ToUint32.type = import_ref Main//int_ops, inst+131, loaded [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %import_ref.7: %Uint32ToInt32.type = import_ref Main//int_ops, inst+149, loaded [template = constants.%Uint32ToInt32]
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, inst+168, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//int_ops, inst+187, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//int_ops, inst+218, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//int_ops, inst+237, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//int_ops, inst+256, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//int_ops, inst+275, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//int_ops, inst+294, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//int_ops, inst+313, unloaded
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//int_ops, inst+333, unloaded
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//int_ops, inst+352, unloaded
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//int_ops, inst+371, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//int_ops, inst+390, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Main//int_ops, inst+408, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, inst+427, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .UInt = %import_ref.22
// CHECK:STDOUT:     .Int32 = %import_ref.23
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
// CHECK:STDOUT:   %.loc5_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_unsigned: init type = call constants.%UInt(%.loc5_10.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc5_10.2: type = value_of_initializer %int.make_type_unsigned [template = constants.%.2]
// CHECK:STDOUT:   %.loc5_10.3: type = converted %int.make_type_unsigned, %.loc5_10.2 [template = constants.%.2]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc6_20.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc6_20.2: type = converted %int.make_type_32, %.loc6_20.1 [template = i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint32(%a.param_patt: i32) -> %.2 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToInt32(%a.param_patt: %.2) -> i32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Int32ToUint32.ref.loc5: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %.loc5_30: i32 = int_value 2147483647 [template = constants.%.3]
// CHECK:STDOUT:   %int.convert_checked.loc5: init %.2 = call %Int32ToUint32.ref.loc5(%.loc5_30) [template = constants.%.4]
// CHECK:STDOUT:   %.loc5_42.1: %.2 = value_of_initializer %int.convert_checked.loc5 [template = constants.%.4]
// CHECK:STDOUT:   %.loc5_42.2: %.2 = converted %int.convert_checked.loc5, %.loc5_42.1 [template = constants.%.4]
// CHECK:STDOUT:   %max: %.2 = bind_name max, %.loc5_42.2
// CHECK:STDOUT:   %Uint32ToInt32.ref: %Uint32ToInt32.type = name_ref Uint32ToInt32, imports.%import_ref.7 [template = constants.%Uint32ToInt32]
// CHECK:STDOUT:   %Int32ToUint32.ref.loc6: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %.loc6_54: i32 = int_value 2147483647 [template = constants.%.3]
// CHECK:STDOUT:   %int.convert_checked.loc6_53: init %.2 = call %Int32ToUint32.ref.loc6(%.loc6_54) [template = constants.%.4]
// CHECK:STDOUT:   %.loc6_53.1: %.2 = value_of_initializer %int.convert_checked.loc6_53 [template = constants.%.4]
// CHECK:STDOUT:   %.loc6_53.2: %.2 = converted %int.convert_checked.loc6_53, %.loc6_53.1 [template = constants.%.4]
// CHECK:STDOUT:   %int.convert_checked.loc6_39: init i32 = call %Uint32ToInt32.ref(%.loc6_53.2) [template = constants.%.3]
// CHECK:STDOUT:   %.loc6_67.1: i32 = value_of_initializer %int.convert_checked.loc6_39 [template = constants.%.3]
// CHECK:STDOUT:   %.loc6_67.2: i32 = converted %int.convert_checked.loc6_39, %.loc6_67.1 [template = constants.%.3]
// CHECK:STDOUT:   %max_roundtrip: i32 = bind_name max_roundtrip, %.loc6_67.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- truncate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 16 [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type unsigned, %.1 [template]
// CHECK:STDOUT:   %Int32ToUint16.type: type = fn_type @Int32ToUint16 [template]
// CHECK:STDOUT:   %Int32ToUint16: %Int32ToUint16.type = struct_value () [template]
// CHECK:STDOUT:   %.3: i32 = int_value 0 [template]
// CHECK:STDOUT:   %.4: %.2 = int_value 0 [template]
// CHECK:STDOUT:   %.5: i32 = int_value 65535 [template]
// CHECK:STDOUT:   %.6: %.2 = int_value 65535 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.7: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %Int32ToInt16.type: type = fn_type @Int32ToInt16 [template]
// CHECK:STDOUT:   %Int32ToInt16: %Int32ToInt16.type = struct_value () [template]
// CHECK:STDOUT:   %.8: i32 = int_value 32767 [template]
// CHECK:STDOUT:   %.9: %.7 = int_value 32767 [template]
// CHECK:STDOUT:   %NegateI32.type: type = fn_type @NegateI32 [template]
// CHECK:STDOUT:   %NegateI32: %NegateI32.type = struct_value () [template]
// CHECK:STDOUT:   %.10: i32 = int_value 32768 [template]
// CHECK:STDOUT:   %.11: i32 = int_value -32768 [template]
// CHECK:STDOUT:   %.12: %.7 = int_value -32768 [template]
// CHECK:STDOUT:   %Uint32ToUint16.type: type = fn_type @Uint32ToUint16 [template]
// CHECK:STDOUT:   %Uint32ToUint16: %Uint32ToUint16.type = struct_value () [template]
// CHECK:STDOUT:   %.13: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %.14: type = int_type unsigned, %.13 [template]
// CHECK:STDOUT:   %Int32ToUint32.type: type = fn_type @Int32ToUint32 [template]
// CHECK:STDOUT:   %Int32ToUint32: %Int32ToUint32.type = struct_value () [template]
// CHECK:STDOUT:   %.15: %.14 = int_value 0 [template]
// CHECK:STDOUT:   %.16: %.14 = int_value 65535 [template]
// CHECK:STDOUT:   %Uint32ToInt16.type: type = fn_type @Uint32ToInt16 [template]
// CHECK:STDOUT:   %Uint32ToInt16: %Uint32ToInt16.type = struct_value () [template]
// CHECK:STDOUT:   %.17: %.7 = int_value 0 [template]
// CHECK:STDOUT:   %.18: %.14 = int_value 32767 [template]
// CHECK:STDOUT:   %IntLiteralToInt16.type: type = fn_type @IntLiteralToInt16 [template]
// CHECK:STDOUT:   %IntLiteralToInt16: %IntLiteralToInt16.type = struct_value () [template]
// CHECK:STDOUT:   %Int32ToIntLiteral.type: type = fn_type @Int32ToIntLiteral [template]
// CHECK:STDOUT:   %Int32ToIntLiteral: %Int32ToIntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %.19: Core.IntLiteral = int_value -32768 [template]
// CHECK:STDOUT:   %.20: Core.IntLiteral = int_value 32767 [template]
// CHECK:STDOUT:   %IntLiteralToUint16.type: type = fn_type @IntLiteralToUint16 [template]
// CHECK:STDOUT:   %IntLiteralToUint16: %IntLiteralToUint16.type = struct_value () [template]
// CHECK:STDOUT:   %.21: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %.22: Core.IntLiteral = int_value 65535 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %NegateI32.type = import_ref Main//int_ops, inst+25, loaded [template = constants.%NegateI32]
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//int_ops, inst+49, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//int_ops, inst+89, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//int_ops, inst+96, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//int_ops, inst+113, unloaded
// CHECK:STDOUT:   %import_ref.6: %Int32ToUint32.type = import_ref Main//int_ops, inst+131, loaded [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//int_ops, inst+149, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, inst+168, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//int_ops, inst+187, unloaded
// CHECK:STDOUT:   %import_ref.10: %Int32ToInt16.type = import_ref Main//int_ops, inst+218, loaded [template = constants.%Int32ToInt16]
// CHECK:STDOUT:   %import_ref.11: %Int32ToUint16.type = import_ref Main//int_ops, inst+237, loaded [template = constants.%Int32ToUint16]
// CHECK:STDOUT:   %import_ref.12: %Uint32ToInt16.type = import_ref Main//int_ops, inst+256, loaded [template = constants.%Uint32ToInt16]
// CHECK:STDOUT:   %import_ref.13: %Uint32ToUint16.type = import_ref Main//int_ops, inst+275, loaded [template = constants.%Uint32ToUint16]
// CHECK:STDOUT:   %import_ref.14: %IntLiteralToInt16.type = import_ref Main//int_ops, inst+294, loaded [template = constants.%IntLiteralToInt16]
// CHECK:STDOUT:   %import_ref.15: %IntLiteralToUint16.type = import_ref Main//int_ops, inst+313, loaded [template = constants.%IntLiteralToUint16]
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//int_ops, inst+333, unloaded
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//int_ops, inst+352, unloaded
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//int_ops, inst+371, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//int_ops, inst+390, unloaded
// CHECK:STDOUT:   %import_ref.20: %Int32ToIntLiteral.type = import_ref Main//int_ops, inst+408, loaded [template = constants.%Int32ToIntLiteral]
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, inst+427, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .UInt = %import_ref.22
// CHECK:STDOUT:     .Int = %import_ref.23
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
// CHECK:STDOUT:   %.loc5_8.1: Core.IntLiteral = int_value 16 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_unsigned.loc5: init type = call constants.%UInt(%.loc5_8.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc5_8.2: type = value_of_initializer %int.make_type_unsigned.loc5 [template = constants.%.2]
// CHECK:STDOUT:   %.loc5_8.3: type = converted %int.make_type_unsigned.loc5, %.loc5_8.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc6_8.1: Core.IntLiteral = int_value 16 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_unsigned.loc6: init type = call constants.%UInt(%.loc6_8.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc6_8.2: type = value_of_initializer %int.make_type_unsigned.loc6 [template = constants.%.2]
// CHECK:STDOUT:   %.loc6_8.3: type = converted %int.make_type_unsigned.loc6, %.loc6_8.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_8.1: Core.IntLiteral = int_value 16 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc8: init type = call constants.%Int(%.loc8_8.1) [template = constants.%.7]
// CHECK:STDOUT:   %.loc8_8.2: type = value_of_initializer %int.make_type_signed.loc8 [template = constants.%.7]
// CHECK:STDOUT:   %.loc8_8.3: type = converted %int.make_type_signed.loc8, %.loc8_8.2 [template = constants.%.7]
// CHECK:STDOUT:   %.loc9_8.1: Core.IntLiteral = int_value 16 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc9: init type = call constants.%Int(%.loc9_8.1) [template = constants.%.7]
// CHECK:STDOUT:   %.loc9_8.2: type = value_of_initializer %int.make_type_signed.loc9 [template = constants.%.7]
// CHECK:STDOUT:   %.loc9_8.3: type = converted %int.make_type_signed.loc9, %.loc9_8.2 [template = constants.%.7]
// CHECK:STDOUT:   %.loc11_8.1: Core.IntLiteral = int_value 16 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_unsigned.loc11: init type = call constants.%UInt(%.loc11_8.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_8.2: type = value_of_initializer %int.make_type_unsigned.loc11 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_8.3: type = converted %int.make_type_unsigned.loc11, %.loc11_8.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_8.1: Core.IntLiteral = int_value 16 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_unsigned.loc12: init type = call constants.%UInt(%.loc12_8.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_8.2: type = value_of_initializer %int.make_type_unsigned.loc12 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_8.3: type = converted %int.make_type_unsigned.loc12, %.loc12_8.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc14_8.1: Core.IntLiteral = int_value 16 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc14: init type = call constants.%Int(%.loc14_8.1) [template = constants.%.7]
// CHECK:STDOUT:   %.loc14_8.2: type = value_of_initializer %int.make_type_signed.loc14 [template = constants.%.7]
// CHECK:STDOUT:   %.loc14_8.3: type = converted %int.make_type_signed.loc14, %.loc14_8.2 [template = constants.%.7]
// CHECK:STDOUT:   %.loc15_8.1: Core.IntLiteral = int_value 16 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc15: init type = call constants.%Int(%.loc15_8.1) [template = constants.%.7]
// CHECK:STDOUT:   %.loc15_8.2: type = value_of_initializer %int.make_type_signed.loc15 [template = constants.%.7]
// CHECK:STDOUT:   %.loc15_8.3: type = converted %int.make_type_signed.loc15, %.loc15_8.2 [template = constants.%.7]
// CHECK:STDOUT:   %.loc17_18.1: Core.IntLiteral = int_value 16 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc17: init type = call constants.%Int(%.loc17_18.1) [template = constants.%.7]
// CHECK:STDOUT:   %.loc17_18.2: type = value_of_initializer %int.make_type_signed.loc17 [template = constants.%.7]
// CHECK:STDOUT:   %.loc17_18.3: type = converted %int.make_type_signed.loc17, %.loc17_18.2 [template = constants.%.7]
// CHECK:STDOUT:   %.loc18_18.1: Core.IntLiteral = int_value 16 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc18: init type = call constants.%Int(%.loc18_18.1) [template = constants.%.7]
// CHECK:STDOUT:   %.loc18_18.2: type = value_of_initializer %int.make_type_signed.loc18 [template = constants.%.7]
// CHECK:STDOUT:   %.loc18_18.3: type = converted %int.make_type_signed.loc18, %.loc18_18.2 [template = constants.%.7]
// CHECK:STDOUT:   %.loc20_18.1: Core.IntLiteral = int_value 16 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_unsigned.loc20: init type = call constants.%UInt(%.loc20_18.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc20_18.2: type = value_of_initializer %int.make_type_unsigned.loc20 [template = constants.%.2]
// CHECK:STDOUT:   %.loc20_18.3: type = converted %int.make_type_unsigned.loc20, %.loc20_18.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc21_18.1: Core.IntLiteral = int_value 16 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_unsigned.loc21: init type = call constants.%UInt(%.loc21_18.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc21_18.2: type = value_of_initializer %int.make_type_unsigned.loc21 [template = constants.%.2]
// CHECK:STDOUT:   %.loc21_18.3: type = converted %int.make_type_unsigned.loc21, %.loc21_18.2 [template = constants.%.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint16(%a.param_patt: i32) -> %.2 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToInt16(%a.param_patt: i32) -> %.7 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NegateI32(%a.param_patt: i32) -> i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToUint16(%a.param_patt: %.14) -> %.2 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint32(%a.param_patt: i32) -> %.14 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToInt16(%a.param_patt: %.14) -> %.7 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteralToInt16(%a.param_patt: Core.IntLiteral) -> %.7 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToIntLiteral(%a.param_patt: i32) -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteralToUint16(%a.param_patt: Core.IntLiteral) -> %.2 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Int32ToUint16.ref.loc5: %Int32ToUint16.type = name_ref Int32ToUint16, imports.%import_ref.11 [template = constants.%Int32ToUint16]
// CHECK:STDOUT:   %.loc5_28: i32 = int_value 0 [template = constants.%.3]
// CHECK:STDOUT:   %int.convert_checked.loc5: init %.2 = call %Int32ToUint16.ref.loc5(%.loc5_28) [template = constants.%.4]
// CHECK:STDOUT:   %.loc5_30.1: %.2 = value_of_initializer %int.convert_checked.loc5 [template = constants.%.4]
// CHECK:STDOUT:   %.loc5_30.2: %.2 = converted %int.convert_checked.loc5, %.loc5_30.1 [template = constants.%.4]
// CHECK:STDOUT:   %a: %.2 = bind_name a, %.loc5_30.2
// CHECK:STDOUT:   %Int32ToUint16.ref.loc6: %Int32ToUint16.type = name_ref Int32ToUint16, imports.%import_ref.11 [template = constants.%Int32ToUint16]
// CHECK:STDOUT:   %.loc6_28: i32 = int_value 65535 [template = constants.%.5]
// CHECK:STDOUT:   %int.convert_checked.loc6: init %.2 = call %Int32ToUint16.ref.loc6(%.loc6_28) [template = constants.%.6]
// CHECK:STDOUT:   %.loc6_35.1: %.2 = value_of_initializer %int.convert_checked.loc6 [template = constants.%.6]
// CHECK:STDOUT:   %.loc6_35.2: %.2 = converted %int.convert_checked.loc6, %.loc6_35.1 [template = constants.%.6]
// CHECK:STDOUT:   %b: %.2 = bind_name b, %.loc6_35.2
// CHECK:STDOUT:   %Int32ToInt16.ref.loc8: %Int32ToInt16.type = name_ref Int32ToInt16, imports.%import_ref.10 [template = constants.%Int32ToInt16]
// CHECK:STDOUT:   %.loc8_27: i32 = int_value 32767 [template = constants.%.8]
// CHECK:STDOUT:   %int.convert_checked.loc8: init %.7 = call %Int32ToInt16.ref.loc8(%.loc8_27) [template = constants.%.9]
// CHECK:STDOUT:   %.loc8_34.1: %.7 = value_of_initializer %int.convert_checked.loc8 [template = constants.%.9]
// CHECK:STDOUT:   %.loc8_34.2: %.7 = converted %int.convert_checked.loc8, %.loc8_34.1 [template = constants.%.9]
// CHECK:STDOUT:   %c: %.7 = bind_name c, %.loc8_34.2
// CHECK:STDOUT:   %Int32ToInt16.ref.loc9: %Int32ToInt16.type = name_ref Int32ToInt16, imports.%import_ref.10 [template = constants.%Int32ToInt16]
// CHECK:STDOUT:   %NegateI32.ref.loc9: %NegateI32.type = name_ref NegateI32, imports.%import_ref.1 [template = constants.%NegateI32]
// CHECK:STDOUT:   %.loc9_37: i32 = int_value 32768 [template = constants.%.10]
// CHECK:STDOUT:   %int.snegate.loc9: init i32 = call %NegateI32.ref.loc9(%.loc9_37) [template = constants.%.11]
// CHECK:STDOUT:   %.loc9_36.1: i32 = value_of_initializer %int.snegate.loc9 [template = constants.%.11]
// CHECK:STDOUT:   %.loc9_36.2: i32 = converted %int.snegate.loc9, %.loc9_36.1 [template = constants.%.11]
// CHECK:STDOUT:   %int.convert_checked.loc9: init %.7 = call %Int32ToInt16.ref.loc9(%.loc9_36.2) [template = constants.%.12]
// CHECK:STDOUT:   %.loc9_45.1: %.7 = value_of_initializer %int.convert_checked.loc9 [template = constants.%.12]
// CHECK:STDOUT:   %.loc9_45.2: %.7 = converted %int.convert_checked.loc9, %.loc9_45.1 [template = constants.%.12]
// CHECK:STDOUT:   %d: %.7 = bind_name d, %.loc9_45.2
// CHECK:STDOUT:   %Uint32ToUint16.ref.loc11: %Uint32ToUint16.type = name_ref Uint32ToUint16, imports.%import_ref.13 [template = constants.%Uint32ToUint16]
// CHECK:STDOUT:   %Int32ToUint32.ref.loc11: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %.loc11_43: i32 = int_value 0 [template = constants.%.3]
// CHECK:STDOUT:   %int.convert_checked.loc11_42: init %.14 = call %Int32ToUint32.ref.loc11(%.loc11_43) [template = constants.%.15]
// CHECK:STDOUT:   %.loc11_42.1: %.14 = value_of_initializer %int.convert_checked.loc11_42 [template = constants.%.15]
// CHECK:STDOUT:   %.loc11_42.2: %.14 = converted %int.convert_checked.loc11_42, %.loc11_42.1 [template = constants.%.15]
// CHECK:STDOUT:   %int.convert_checked.loc11_28: init %.2 = call %Uint32ToUint16.ref.loc11(%.loc11_42.2) [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_46.1: %.2 = value_of_initializer %int.convert_checked.loc11_28 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_46.2: %.2 = converted %int.convert_checked.loc11_28, %.loc11_46.1 [template = constants.%.4]
// CHECK:STDOUT:   %e: %.2 = bind_name e, %.loc11_46.2
// CHECK:STDOUT:   %Uint32ToUint16.ref.loc12: %Uint32ToUint16.type = name_ref Uint32ToUint16, imports.%import_ref.13 [template = constants.%Uint32ToUint16]
// CHECK:STDOUT:   %Int32ToUint32.ref.loc12: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %.loc12_43: i32 = int_value 65535 [template = constants.%.5]
// CHECK:STDOUT:   %int.convert_checked.loc12_42: init %.14 = call %Int32ToUint32.ref.loc12(%.loc12_43) [template = constants.%.16]
// CHECK:STDOUT:   %.loc12_42.1: %.14 = value_of_initializer %int.convert_checked.loc12_42 [template = constants.%.16]
// CHECK:STDOUT:   %.loc12_42.2: %.14 = converted %int.convert_checked.loc12_42, %.loc12_42.1 [template = constants.%.16]
// CHECK:STDOUT:   %int.convert_checked.loc12_28: init %.2 = call %Uint32ToUint16.ref.loc12(%.loc12_42.2) [template = constants.%.6]
// CHECK:STDOUT:   %.loc12_51.1: %.2 = value_of_initializer %int.convert_checked.loc12_28 [template = constants.%.6]
// CHECK:STDOUT:   %.loc12_51.2: %.2 = converted %int.convert_checked.loc12_28, %.loc12_51.1 [template = constants.%.6]
// CHECK:STDOUT:   %f: %.2 = bind_name f, %.loc12_51.2
// CHECK:STDOUT:   %Uint32ToInt16.ref.loc14: %Uint32ToInt16.type = name_ref Uint32ToInt16, imports.%import_ref.12 [template = constants.%Uint32ToInt16]
// CHECK:STDOUT:   %Int32ToUint32.ref.loc14: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %.loc14_42: i32 = int_value 0 [template = constants.%.3]
// CHECK:STDOUT:   %int.convert_checked.loc14_41: init %.14 = call %Int32ToUint32.ref.loc14(%.loc14_42) [template = constants.%.15]
// CHECK:STDOUT:   %.loc14_41.1: %.14 = value_of_initializer %int.convert_checked.loc14_41 [template = constants.%.15]
// CHECK:STDOUT:   %.loc14_41.2: %.14 = converted %int.convert_checked.loc14_41, %.loc14_41.1 [template = constants.%.15]
// CHECK:STDOUT:   %int.convert_checked.loc14_27: init %.7 = call %Uint32ToInt16.ref.loc14(%.loc14_41.2) [template = constants.%.17]
// CHECK:STDOUT:   %.loc14_45.1: %.7 = value_of_initializer %int.convert_checked.loc14_27 [template = constants.%.17]
// CHECK:STDOUT:   %.loc14_45.2: %.7 = converted %int.convert_checked.loc14_27, %.loc14_45.1 [template = constants.%.17]
// CHECK:STDOUT:   %g: %.7 = bind_name g, %.loc14_45.2
// CHECK:STDOUT:   %Uint32ToInt16.ref.loc15: %Uint32ToInt16.type = name_ref Uint32ToInt16, imports.%import_ref.12 [template = constants.%Uint32ToInt16]
// CHECK:STDOUT:   %Int32ToUint32.ref.loc15: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %.loc15_42: i32 = int_value 32767 [template = constants.%.8]
// CHECK:STDOUT:   %int.convert_checked.loc15_41: init %.14 = call %Int32ToUint32.ref.loc15(%.loc15_42) [template = constants.%.18]
// CHECK:STDOUT:   %.loc15_41.1: %.14 = value_of_initializer %int.convert_checked.loc15_41 [template = constants.%.18]
// CHECK:STDOUT:   %.loc15_41.2: %.14 = converted %int.convert_checked.loc15_41, %.loc15_41.1 [template = constants.%.18]
// CHECK:STDOUT:   %int.convert_checked.loc15_27: init %.7 = call %Uint32ToInt16.ref.loc15(%.loc15_41.2) [template = constants.%.9]
// CHECK:STDOUT:   %.loc15_50.1: %.7 = value_of_initializer %int.convert_checked.loc15_27 [template = constants.%.9]
// CHECK:STDOUT:   %.loc15_50.2: %.7 = converted %int.convert_checked.loc15_27, %.loc15_50.1 [template = constants.%.9]
// CHECK:STDOUT:   %h: %.7 = bind_name h, %.loc15_50.2
// CHECK:STDOUT:   %IntLiteralToInt16.ref.loc17: %IntLiteralToInt16.type = name_ref IntLiteralToInt16, imports.%import_ref.14 [template = constants.%IntLiteralToInt16]
// CHECK:STDOUT:   %Int32ToIntLiteral.ref.loc17: %Int32ToIntLiteral.type = name_ref Int32ToIntLiteral, imports.%import_ref.20 [template = constants.%Int32ToIntLiteral]
// CHECK:STDOUT:   %NegateI32.ref.loc17: %NegateI32.type = name_ref NegateI32, imports.%import_ref.1 [template = constants.%NegateI32]
// CHECK:STDOUT:   %.loc17_70: i32 = int_value 32768 [template = constants.%.10]
// CHECK:STDOUT:   %int.snegate.loc17: init i32 = call %NegateI32.ref.loc17(%.loc17_70) [template = constants.%.11]
// CHECK:STDOUT:   %.loc17_69.1: i32 = value_of_initializer %int.snegate.loc17 [template = constants.%.11]
// CHECK:STDOUT:   %.loc17_69.2: i32 = converted %int.snegate.loc17, %.loc17_69.1 [template = constants.%.11]
// CHECK:STDOUT:   %int.convert_checked.loc17_59: init Core.IntLiteral = call %Int32ToIntLiteral.ref.loc17(%.loc17_69.2) [template = constants.%.19]
// CHECK:STDOUT:   %.loc17_59.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc17_59 [template = constants.%.19]
// CHECK:STDOUT:   %.loc17_59.2: Core.IntLiteral = converted %int.convert_checked.loc17_59, %.loc17_59.1 [template = constants.%.19]
// CHECK:STDOUT:   %int.convert_checked.loc17_41: init %.7 = call %IntLiteralToInt16.ref.loc17(%.loc17_59.2) [template = constants.%.12]
// CHECK:STDOUT:   %.loc17_79.1: %.7 = value_of_initializer %int.convert_checked.loc17_41 [template = constants.%.12]
// CHECK:STDOUT:   %.loc17_79.2: %.7 = converted %int.convert_checked.loc17_41, %.loc17_79.1 [template = constants.%.12]
// CHECK:STDOUT:   %lit_i16_min: %.7 = bind_name lit_i16_min, %.loc17_79.2
// CHECK:STDOUT:   %IntLiteralToInt16.ref.loc18: %IntLiteralToInt16.type = name_ref IntLiteralToInt16, imports.%import_ref.14 [template = constants.%IntLiteralToInt16]
// CHECK:STDOUT:   %Int32ToIntLiteral.ref.loc18: %Int32ToIntLiteral.type = name_ref Int32ToIntLiteral, imports.%import_ref.20 [template = constants.%Int32ToIntLiteral]
// CHECK:STDOUT:   %.loc18_60: i32 = int_value 32767 [template = constants.%.8]
// CHECK:STDOUT:   %int.convert_checked.loc18_59: init Core.IntLiteral = call %Int32ToIntLiteral.ref.loc18(%.loc18_60) [template = constants.%.20]
// CHECK:STDOUT:   %.loc18_59.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc18_59 [template = constants.%.20]
// CHECK:STDOUT:   %.loc18_59.2: Core.IntLiteral = converted %int.convert_checked.loc18_59, %.loc18_59.1 [template = constants.%.20]
// CHECK:STDOUT:   %int.convert_checked.loc18_41: init %.7 = call %IntLiteralToInt16.ref.loc18(%.loc18_59.2) [template = constants.%.9]
// CHECK:STDOUT:   %.loc18_68.1: %.7 = value_of_initializer %int.convert_checked.loc18_41 [template = constants.%.9]
// CHECK:STDOUT:   %.loc18_68.2: %.7 = converted %int.convert_checked.loc18_41, %.loc18_68.1 [template = constants.%.9]
// CHECK:STDOUT:   %lit_i16_max: %.7 = bind_name lit_i16_max, %.loc18_68.2
// CHECK:STDOUT:   %IntLiteralToUint16.ref.loc20: %IntLiteralToUint16.type = name_ref IntLiteralToUint16, imports.%import_ref.15 [template = constants.%IntLiteralToUint16]
// CHECK:STDOUT:   %Int32ToIntLiteral.ref.loc20: %Int32ToIntLiteral.type = name_ref Int32ToIntLiteral, imports.%import_ref.20 [template = constants.%Int32ToIntLiteral]
// CHECK:STDOUT:   %.loc20_61: i32 = int_value 0 [template = constants.%.3]
// CHECK:STDOUT:   %int.convert_checked.loc20_60: init Core.IntLiteral = call %Int32ToIntLiteral.ref.loc20(%.loc20_61) [template = constants.%.21]
// CHECK:STDOUT:   %.loc20_60.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc20_60 [template = constants.%.21]
// CHECK:STDOUT:   %.loc20_60.2: Core.IntLiteral = converted %int.convert_checked.loc20_60, %.loc20_60.1 [template = constants.%.21]
// CHECK:STDOUT:   %int.convert_checked.loc20_42: init %.2 = call %IntLiteralToUint16.ref.loc20(%.loc20_60.2) [template = constants.%.4]
// CHECK:STDOUT:   %.loc20_64.1: %.2 = value_of_initializer %int.convert_checked.loc20_42 [template = constants.%.4]
// CHECK:STDOUT:   %.loc20_64.2: %.2 = converted %int.convert_checked.loc20_42, %.loc20_64.1 [template = constants.%.4]
// CHECK:STDOUT:   %lit_u16_min: %.2 = bind_name lit_u16_min, %.loc20_64.2
// CHECK:STDOUT:   %IntLiteralToUint16.ref.loc21: %IntLiteralToUint16.type = name_ref IntLiteralToUint16, imports.%import_ref.15 [template = constants.%IntLiteralToUint16]
// CHECK:STDOUT:   %Int32ToIntLiteral.ref.loc21: %Int32ToIntLiteral.type = name_ref Int32ToIntLiteral, imports.%import_ref.20 [template = constants.%Int32ToIntLiteral]
// CHECK:STDOUT:   %.loc21_61: i32 = int_value 65535 [template = constants.%.5]
// CHECK:STDOUT:   %int.convert_checked.loc21_60: init Core.IntLiteral = call %Int32ToIntLiteral.ref.loc21(%.loc21_61) [template = constants.%.22]
// CHECK:STDOUT:   %.loc21_60.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc21_60 [template = constants.%.22]
// CHECK:STDOUT:   %.loc21_60.2: Core.IntLiteral = converted %int.convert_checked.loc21_60, %.loc21_60.1 [template = constants.%.22]
// CHECK:STDOUT:   %int.convert_checked.loc21_42: init %.2 = call %IntLiteralToUint16.ref.loc21(%.loc21_60.2) [template = constants.%.6]
// CHECK:STDOUT:   %.loc21_69.1: %.2 = value_of_initializer %int.convert_checked.loc21_42 [template = constants.%.6]
// CHECK:STDOUT:   %.loc21_69.2: %.2 = converted %int.convert_checked.loc21_42, %.loc21_69.1 [template = constants.%.6]
// CHECK:STDOUT:   %lit_u16_max: %.2 = bind_name lit_u16_max, %.loc21_69.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- zero_extend.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type unsigned, %.1 [template]
// CHECK:STDOUT:   %Uint32ToUint64.type: type = fn_type @Uint32ToUint64 [template]
// CHECK:STDOUT:   %Uint32ToUint64: %Uint32ToUint64.type = struct_value () [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %.4: type = int_type unsigned, %.3 [template]
// CHECK:STDOUT:   %Int32ToUint32.type: type = fn_type @Int32ToUint32 [template]
// CHECK:STDOUT:   %Int32ToUint32: %Int32ToUint32.type = struct_value () [template]
// CHECK:STDOUT:   %.5: i32 = int_value 0 [template]
// CHECK:STDOUT:   %.6: %.4 = int_value 0 [template]
// CHECK:STDOUT:   %.7: %.2 = int_value 0 [template]
// CHECK:STDOUT:   %AddU32.type: type = fn_type @AddU32 [template]
// CHECK:STDOUT:   %AddU32: %AddU32.type = struct_value () [template]
// CHECK:STDOUT:   %.8: i32 = int_value 2147483647 [template]
// CHECK:STDOUT:   %.9: %.4 = int_value 2147483647 [template]
// CHECK:STDOUT:   %.10: %.4 = int_value 4294967294 [template]
// CHECK:STDOUT:   %.11: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.12: %.4 = int_value 1 [template]
// CHECK:STDOUT:   %.13: %.4 = int_value 4294967295 [template]
// CHECK:STDOUT:   %.14: %.2 = int_value 4294967295 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.15: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %Uint32ToInt64.type: type = fn_type @Uint32ToInt64 [template]
// CHECK:STDOUT:   %Uint32ToInt64: %Uint32ToInt64.type = struct_value () [template]
// CHECK:STDOUT:   %.16: %.15 = int_value 0 [template]
// CHECK:STDOUT:   %.17: %.15 = int_value 4294967295 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//int_ops, inst+25, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//int_ops, inst+49, unloaded
// CHECK:STDOUT:   %import_ref.3: %AddU32.type = import_ref Main//int_ops, inst+89, loaded [template = constants.%AddU32]
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//int_ops, inst+96, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//int_ops, inst+113, unloaded
// CHECK:STDOUT:   %import_ref.6: %Int32ToUint32.type = import_ref Main//int_ops, inst+131, loaded [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//int_ops, inst+149, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, inst+168, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//int_ops, inst+187, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//int_ops, inst+218, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//int_ops, inst+237, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//int_ops, inst+256, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//int_ops, inst+275, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//int_ops, inst+294, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//int_ops, inst+313, unloaded
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//int_ops, inst+333, unloaded
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//int_ops, inst+352, unloaded
// CHECK:STDOUT:   %import_ref.18: %Uint32ToInt64.type = import_ref Main//int_ops, inst+371, loaded [template = constants.%Uint32ToInt64]
// CHECK:STDOUT:   %import_ref.19: %Uint32ToUint64.type = import_ref Main//int_ops, inst+390, loaded [template = constants.%Uint32ToUint64]
// CHECK:STDOUT:   %import_ref.20 = import_ref Main//int_ops, inst+408, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, inst+427, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .UInt = %import_ref.22
// CHECK:STDOUT:     .Int = %import_ref.23
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
// CHECK:STDOUT:   %.loc5_8.1: Core.IntLiteral = int_value 64 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_unsigned.loc5: init type = call constants.%UInt(%.loc5_8.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc5_8.2: type = value_of_initializer %int.make_type_unsigned.loc5 [template = constants.%.2]
// CHECK:STDOUT:   %.loc5_8.3: type = converted %int.make_type_unsigned.loc5, %.loc5_8.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc6_8.1: Core.IntLiteral = int_value 64 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_unsigned.loc6: init type = call constants.%UInt(%.loc6_8.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc6_8.2: type = value_of_initializer %int.make_type_unsigned.loc6 [template = constants.%.2]
// CHECK:STDOUT:   %.loc6_8.3: type = converted %int.make_type_unsigned.loc6, %.loc6_8.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_8.1: Core.IntLiteral = int_value 64 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc11: init type = call constants.%Int(%.loc11_8.1) [template = constants.%.15]
// CHECK:STDOUT:   %.loc11_8.2: type = value_of_initializer %int.make_type_signed.loc11 [template = constants.%.15]
// CHECK:STDOUT:   %.loc11_8.3: type = converted %int.make_type_signed.loc11, %.loc11_8.2 [template = constants.%.15]
// CHECK:STDOUT:   %.loc12_8.1: Core.IntLiteral = int_value 64 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%.loc12_8.1) [template = constants.%.15]
// CHECK:STDOUT:   %.loc12_8.2: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%.15]
// CHECK:STDOUT:   %.loc12_8.3: type = converted %int.make_type_signed.loc12, %.loc12_8.2 [template = constants.%.15]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToUint64(%a.param_patt: %.4) -> %.2 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint32(%a.param_patt: i32) -> %.4 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddU32(%a.param_patt: %.4, %b.param_patt: %.4) -> %.4 = "int.uadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToInt64(%a.param_patt: %.4) -> %.15 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Uint32ToUint64.ref.loc5: %Uint32ToUint64.type = name_ref Uint32ToUint64, imports.%import_ref.19 [template = constants.%Uint32ToUint64]
// CHECK:STDOUT:   %Int32ToUint32.ref.loc5: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %.loc5_43: i32 = int_value 0 [template = constants.%.5]
// CHECK:STDOUT:   %int.convert_checked.loc5_42: init %.4 = call %Int32ToUint32.ref.loc5(%.loc5_43) [template = constants.%.6]
// CHECK:STDOUT:   %.loc5_42.1: %.4 = value_of_initializer %int.convert_checked.loc5_42 [template = constants.%.6]
// CHECK:STDOUT:   %.loc5_42.2: %.4 = converted %int.convert_checked.loc5_42, %.loc5_42.1 [template = constants.%.6]
// CHECK:STDOUT:   %int.convert_checked.loc5_28: init %.2 = call %Uint32ToUint64.ref.loc5(%.loc5_42.2) [template = constants.%.7]
// CHECK:STDOUT:   %.loc5_46.1: %.2 = value_of_initializer %int.convert_checked.loc5_28 [template = constants.%.7]
// CHECK:STDOUT:   %.loc5_46.2: %.2 = converted %int.convert_checked.loc5_28, %.loc5_46.1 [template = constants.%.7]
// CHECK:STDOUT:   %a: %.2 = bind_name a, %.loc5_46.2
// CHECK:STDOUT:   %Uint32ToUint64.ref.loc6: %Uint32ToUint64.type = name_ref Uint32ToUint64, imports.%import_ref.19 [template = constants.%Uint32ToUint64]
// CHECK:STDOUT:   %AddU32.ref.loc7: %AddU32.type = name_ref AddU32, imports.%import_ref.3 [template = constants.%AddU32]
// CHECK:STDOUT:   %AddU32.ref.loc8: %AddU32.type = name_ref AddU32, imports.%import_ref.3 [template = constants.%AddU32]
// CHECK:STDOUT:   %Int32ToUint32.ref.loc8_12: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %.loc8_26: i32 = int_value 2147483647 [template = constants.%.8]
// CHECK:STDOUT:   %int.convert_checked.loc8_25: init %.4 = call %Int32ToUint32.ref.loc8_12(%.loc8_26) [template = constants.%.9]
// CHECK:STDOUT:   %Int32ToUint32.ref.loc8_40: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %.loc8_54: i32 = int_value 2147483647 [template = constants.%.8]
// CHECK:STDOUT:   %int.convert_checked.loc8_53: init %.4 = call %Int32ToUint32.ref.loc8_40(%.loc8_54) [template = constants.%.9]
// CHECK:STDOUT:   %.loc8_25.1: %.4 = value_of_initializer %int.convert_checked.loc8_25 [template = constants.%.9]
// CHECK:STDOUT:   %.loc8_25.2: %.4 = converted %int.convert_checked.loc8_25, %.loc8_25.1 [template = constants.%.9]
// CHECK:STDOUT:   %.loc8_53.1: %.4 = value_of_initializer %int.convert_checked.loc8_53 [template = constants.%.9]
// CHECK:STDOUT:   %.loc8_53.2: %.4 = converted %int.convert_checked.loc8_53, %.loc8_53.1 [template = constants.%.9]
// CHECK:STDOUT:   %int.uadd.loc8: init %.4 = call %AddU32.ref.loc8(%.loc8_25.2, %.loc8_53.2) [template = constants.%.10]
// CHECK:STDOUT:   %Int32ToUint32.ref.loc9: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %.loc9_19: i32 = int_value 1 [template = constants.%.11]
// CHECK:STDOUT:   %int.convert_checked.loc9: init %.4 = call %Int32ToUint32.ref.loc9(%.loc9_19) [template = constants.%.12]
// CHECK:STDOUT:   %.loc8_11.1: %.4 = value_of_initializer %int.uadd.loc8 [template = constants.%.10]
// CHECK:STDOUT:   %.loc8_11.2: %.4 = converted %int.uadd.loc8, %.loc8_11.1 [template = constants.%.10]
// CHECK:STDOUT:   %.loc9_18.1: %.4 = value_of_initializer %int.convert_checked.loc9 [template = constants.%.12]
// CHECK:STDOUT:   %.loc9_18.2: %.4 = converted %int.convert_checked.loc9, %.loc9_18.1 [template = constants.%.12]
// CHECK:STDOUT:   %int.uadd.loc7: init %.4 = call %AddU32.ref.loc7(%.loc8_11.2, %.loc9_18.2) [template = constants.%.13]
// CHECK:STDOUT:   %.loc7_9.1: %.4 = value_of_initializer %int.uadd.loc7 [template = constants.%.13]
// CHECK:STDOUT:   %.loc7_9.2: %.4 = converted %int.uadd.loc7, %.loc7_9.1 [template = constants.%.13]
// CHECK:STDOUT:   %int.convert_checked.loc6: init %.2 = call %Uint32ToUint64.ref.loc6(%.loc7_9.2) [template = constants.%.14]
// CHECK:STDOUT:   %.loc9_23.1: %.2 = value_of_initializer %int.convert_checked.loc6 [template = constants.%.14]
// CHECK:STDOUT:   %.loc9_23.2: %.2 = converted %int.convert_checked.loc6, %.loc9_23.1 [template = constants.%.14]
// CHECK:STDOUT:   %b: %.2 = bind_name b, %.loc9_23.2
// CHECK:STDOUT:   %Uint32ToInt64.ref.loc11: %Uint32ToInt64.type = name_ref Uint32ToInt64, imports.%import_ref.18 [template = constants.%Uint32ToInt64]
// CHECK:STDOUT:   %Int32ToUint32.ref.loc11: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %.loc11_42: i32 = int_value 0 [template = constants.%.5]
// CHECK:STDOUT:   %int.convert_checked.loc11_41: init %.4 = call %Int32ToUint32.ref.loc11(%.loc11_42) [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_41.1: %.4 = value_of_initializer %int.convert_checked.loc11_41 [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_41.2: %.4 = converted %int.convert_checked.loc11_41, %.loc11_41.1 [template = constants.%.6]
// CHECK:STDOUT:   %int.convert_checked.loc11_27: init %.15 = call %Uint32ToInt64.ref.loc11(%.loc11_41.2) [template = constants.%.16]
// CHECK:STDOUT:   %.loc11_45.1: %.15 = value_of_initializer %int.convert_checked.loc11_27 [template = constants.%.16]
// CHECK:STDOUT:   %.loc11_45.2: %.15 = converted %int.convert_checked.loc11_27, %.loc11_45.1 [template = constants.%.16]
// CHECK:STDOUT:   %c: %.15 = bind_name c, %.loc11_45.2
// CHECK:STDOUT:   %Uint32ToInt64.ref.loc12: %Uint32ToInt64.type = name_ref Uint32ToInt64, imports.%import_ref.18 [template = constants.%Uint32ToInt64]
// CHECK:STDOUT:   %AddU32.ref.loc13: %AddU32.type = name_ref AddU32, imports.%import_ref.3 [template = constants.%AddU32]
// CHECK:STDOUT:   %AddU32.ref.loc14: %AddU32.type = name_ref AddU32, imports.%import_ref.3 [template = constants.%AddU32]
// CHECK:STDOUT:   %Int32ToUint32.ref.loc14_12: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %.loc14_26: i32 = int_value 2147483647 [template = constants.%.8]
// CHECK:STDOUT:   %int.convert_checked.loc14_25: init %.4 = call %Int32ToUint32.ref.loc14_12(%.loc14_26) [template = constants.%.9]
// CHECK:STDOUT:   %Int32ToUint32.ref.loc14_40: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %.loc14_54: i32 = int_value 2147483647 [template = constants.%.8]
// CHECK:STDOUT:   %int.convert_checked.loc14_53: init %.4 = call %Int32ToUint32.ref.loc14_40(%.loc14_54) [template = constants.%.9]
// CHECK:STDOUT:   %.loc14_25.1: %.4 = value_of_initializer %int.convert_checked.loc14_25 [template = constants.%.9]
// CHECK:STDOUT:   %.loc14_25.2: %.4 = converted %int.convert_checked.loc14_25, %.loc14_25.1 [template = constants.%.9]
// CHECK:STDOUT:   %.loc14_53.1: %.4 = value_of_initializer %int.convert_checked.loc14_53 [template = constants.%.9]
// CHECK:STDOUT:   %.loc14_53.2: %.4 = converted %int.convert_checked.loc14_53, %.loc14_53.1 [template = constants.%.9]
// CHECK:STDOUT:   %int.uadd.loc14: init %.4 = call %AddU32.ref.loc14(%.loc14_25.2, %.loc14_53.2) [template = constants.%.10]
// CHECK:STDOUT:   %Int32ToUint32.ref.loc15: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %.loc15_19: i32 = int_value 1 [template = constants.%.11]
// CHECK:STDOUT:   %int.convert_checked.loc15: init %.4 = call %Int32ToUint32.ref.loc15(%.loc15_19) [template = constants.%.12]
// CHECK:STDOUT:   %.loc14_11.1: %.4 = value_of_initializer %int.uadd.loc14 [template = constants.%.10]
// CHECK:STDOUT:   %.loc14_11.2: %.4 = converted %int.uadd.loc14, %.loc14_11.1 [template = constants.%.10]
// CHECK:STDOUT:   %.loc15_18.1: %.4 = value_of_initializer %int.convert_checked.loc15 [template = constants.%.12]
// CHECK:STDOUT:   %.loc15_18.2: %.4 = converted %int.convert_checked.loc15, %.loc15_18.1 [template = constants.%.12]
// CHECK:STDOUT:   %int.uadd.loc13: init %.4 = call %AddU32.ref.loc13(%.loc14_11.2, %.loc15_18.2) [template = constants.%.13]
// CHECK:STDOUT:   %.loc13_9.1: %.4 = value_of_initializer %int.uadd.loc13 [template = constants.%.13]
// CHECK:STDOUT:   %.loc13_9.2: %.4 = converted %int.uadd.loc13, %.loc13_9.1 [template = constants.%.13]
// CHECK:STDOUT:   %int.convert_checked.loc12: init %.15 = call %Uint32ToInt64.ref.loc12(%.loc13_9.2) [template = constants.%.17]
// CHECK:STDOUT:   %.loc15_23.1: %.15 = value_of_initializer %int.convert_checked.loc12 [template = constants.%.17]
// CHECK:STDOUT:   %.loc15_23.2: %.15 = converted %int.convert_checked.loc12, %.loc15_23.1 [template = constants.%.17]
// CHECK:STDOUT:   %d: %.15 = bind_name d, %.loc15_23.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- sign_extend.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type unsigned, %.1 [template]
// CHECK:STDOUT:   %Int32ToUint64.type: type = fn_type @Int32ToUint64 [template]
// CHECK:STDOUT:   %Int32ToUint64: %Int32ToUint64.type = struct_value () [template]
// CHECK:STDOUT:   %.3: i32 = int_value 0 [template]
// CHECK:STDOUT:   %.4: %.2 = int_value 0 [template]
// CHECK:STDOUT:   %.5: i32 = int_value 2147483647 [template]
// CHECK:STDOUT:   %.6: %.2 = int_value 2147483647 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.7: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %Int32ToInt64.type: type = fn_type @Int32ToInt64 [template]
// CHECK:STDOUT:   %Int32ToInt64: %Int32ToInt64.type = struct_value () [template]
// CHECK:STDOUT:   %SubI32.type: type = fn_type @SubI32 [template]
// CHECK:STDOUT:   %SubI32: %SubI32.type = struct_value () [template]
// CHECK:STDOUT:   %NegateI32.type: type = fn_type @NegateI32 [template]
// CHECK:STDOUT:   %NegateI32: %NegateI32.type = struct_value () [template]
// CHECK:STDOUT:   %.8: i32 = int_value -2147483647 [template]
// CHECK:STDOUT:   %.9: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.10: i32 = int_value -2147483648 [template]
// CHECK:STDOUT:   %.11: %.7 = int_value -2147483648 [template]
// CHECK:STDOUT:   %.12: %.7 = int_value 2147483647 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %NegateI32.type = import_ref Main//int_ops, inst+25, loaded [template = constants.%NegateI32]
// CHECK:STDOUT:   %import_ref.2: %SubI32.type = import_ref Main//int_ops, inst+49, loaded [template = constants.%SubI32]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//int_ops, inst+89, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//int_ops, inst+96, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//int_ops, inst+113, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//int_ops, inst+131, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//int_ops, inst+149, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, inst+168, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//int_ops, inst+187, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//int_ops, inst+218, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//int_ops, inst+237, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//int_ops, inst+256, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//int_ops, inst+275, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//int_ops, inst+294, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//int_ops, inst+313, unloaded
// CHECK:STDOUT:   %import_ref.16: %Int32ToInt64.type = import_ref Main//int_ops, inst+333, loaded [template = constants.%Int32ToInt64]
// CHECK:STDOUT:   %import_ref.17: %Int32ToUint64.type = import_ref Main//int_ops, inst+352, loaded [template = constants.%Int32ToUint64]
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//int_ops, inst+371, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//int_ops, inst+390, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Main//int_ops, inst+408, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, inst+427, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .UInt = %import_ref.22
// CHECK:STDOUT:     .Int = %import_ref.23
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
// CHECK:STDOUT:   %.loc5_8.1: Core.IntLiteral = int_value 64 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_unsigned.loc5: init type = call constants.%UInt(%.loc5_8.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc5_8.2: type = value_of_initializer %int.make_type_unsigned.loc5 [template = constants.%.2]
// CHECK:STDOUT:   %.loc5_8.3: type = converted %int.make_type_unsigned.loc5, %.loc5_8.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc6_8.1: Core.IntLiteral = int_value 64 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_unsigned.loc6: init type = call constants.%UInt(%.loc6_8.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc6_8.2: type = value_of_initializer %int.make_type_unsigned.loc6 [template = constants.%.2]
// CHECK:STDOUT:   %.loc6_8.3: type = converted %int.make_type_unsigned.loc6, %.loc6_8.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_8.1: Core.IntLiteral = int_value 64 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc8: init type = call constants.%Int(%.loc8_8.1) [template = constants.%.7]
// CHECK:STDOUT:   %.loc8_8.2: type = value_of_initializer %int.make_type_signed.loc8 [template = constants.%.7]
// CHECK:STDOUT:   %.loc8_8.3: type = converted %int.make_type_signed.loc8, %.loc8_8.2 [template = constants.%.7]
// CHECK:STDOUT:   %.loc9_8.1: Core.IntLiteral = int_value 64 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc9: init type = call constants.%Int(%.loc9_8.1) [template = constants.%.7]
// CHECK:STDOUT:   %.loc9_8.2: type = value_of_initializer %int.make_type_signed.loc9 [template = constants.%.7]
// CHECK:STDOUT:   %.loc9_8.3: type = converted %int.make_type_signed.loc9, %.loc9_8.2 [template = constants.%.7]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint64(%a.param_patt: i32) -> %.2 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToInt64(%a.param_patt: i32) -> %.7 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubI32(%a.param_patt: i32, %b.param_patt: i32) -> i32 = "int.ssub";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NegateI32(%a.param_patt: i32) -> i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Int32ToUint64.ref.loc5: %Int32ToUint64.type = name_ref Int32ToUint64, imports.%import_ref.17 [template = constants.%Int32ToUint64]
// CHECK:STDOUT:   %.loc5_28: i32 = int_value 0 [template = constants.%.3]
// CHECK:STDOUT:   %int.convert_checked.loc5: init %.2 = call %Int32ToUint64.ref.loc5(%.loc5_28) [template = constants.%.4]
// CHECK:STDOUT:   %.loc5_30.1: %.2 = value_of_initializer %int.convert_checked.loc5 [template = constants.%.4]
// CHECK:STDOUT:   %.loc5_30.2: %.2 = converted %int.convert_checked.loc5, %.loc5_30.1 [template = constants.%.4]
// CHECK:STDOUT:   %a: %.2 = bind_name a, %.loc5_30.2
// CHECK:STDOUT:   %Int32ToUint64.ref.loc6: %Int32ToUint64.type = name_ref Int32ToUint64, imports.%import_ref.17 [template = constants.%Int32ToUint64]
// CHECK:STDOUT:   %.loc6_28: i32 = int_value 2147483647 [template = constants.%.5]
// CHECK:STDOUT:   %int.convert_checked.loc6: init %.2 = call %Int32ToUint64.ref.loc6(%.loc6_28) [template = constants.%.6]
// CHECK:STDOUT:   %.loc6_40.1: %.2 = value_of_initializer %int.convert_checked.loc6 [template = constants.%.6]
// CHECK:STDOUT:   %.loc6_40.2: %.2 = converted %int.convert_checked.loc6, %.loc6_40.1 [template = constants.%.6]
// CHECK:STDOUT:   %b: %.2 = bind_name b, %.loc6_40.2
// CHECK:STDOUT:   %Int32ToInt64.ref.loc8: %Int32ToInt64.type = name_ref Int32ToInt64, imports.%import_ref.16 [template = constants.%Int32ToInt64]
// CHECK:STDOUT:   %SubI32.ref: %SubI32.type = name_ref SubI32, imports.%import_ref.2 [template = constants.%SubI32]
// CHECK:STDOUT:   %NegateI32.ref: %NegateI32.type = name_ref NegateI32, imports.%import_ref.1 [template = constants.%NegateI32]
// CHECK:STDOUT:   %.loc8_44: i32 = int_value 2147483647 [template = constants.%.5]
// CHECK:STDOUT:   %int.snegate: init i32 = call %NegateI32.ref(%.loc8_44) [template = constants.%.8]
// CHECK:STDOUT:   %.loc8_58: i32 = int_value 1 [template = constants.%.9]
// CHECK:STDOUT:   %.loc8_43.1: i32 = value_of_initializer %int.snegate [template = constants.%.8]
// CHECK:STDOUT:   %.loc8_43.2: i32 = converted %int.snegate, %.loc8_43.1 [template = constants.%.8]
// CHECK:STDOUT:   %int.ssub: init i32 = call %SubI32.ref(%.loc8_43.2, %.loc8_58) [template = constants.%.10]
// CHECK:STDOUT:   %.loc8_33.1: i32 = value_of_initializer %int.ssub [template = constants.%.10]
// CHECK:STDOUT:   %.loc8_33.2: i32 = converted %int.ssub, %.loc8_33.1 [template = constants.%.10]
// CHECK:STDOUT:   %int.convert_checked.loc8: init %.7 = call %Int32ToInt64.ref.loc8(%.loc8_33.2) [template = constants.%.11]
// CHECK:STDOUT:   %.loc8_61.1: %.7 = value_of_initializer %int.convert_checked.loc8 [template = constants.%.11]
// CHECK:STDOUT:   %.loc8_61.2: %.7 = converted %int.convert_checked.loc8, %.loc8_61.1 [template = constants.%.11]
// CHECK:STDOUT:   %c: %.7 = bind_name c, %.loc8_61.2
// CHECK:STDOUT:   %Int32ToInt64.ref.loc9: %Int32ToInt64.type = name_ref Int32ToInt64, imports.%import_ref.16 [template = constants.%Int32ToInt64]
// CHECK:STDOUT:   %.loc9_27: i32 = int_value 2147483647 [template = constants.%.5]
// CHECK:STDOUT:   %int.convert_checked.loc9: init %.7 = call %Int32ToInt64.ref.loc9(%.loc9_27) [template = constants.%.12]
// CHECK:STDOUT:   %.loc9_39.1: %.7 = value_of_initializer %int.convert_checked.loc9 [template = constants.%.12]
// CHECK:STDOUT:   %.loc9_39.2: %.7 = converted %int.convert_checked.loc9, %.loc9_39.1 [template = constants.%.12]
// CHECK:STDOUT:   %d: %.7 = bind_name d, %.loc9_39.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_too_large_u32_for_i32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Uint32ToInt32.type: type = fn_type @Uint32ToInt32 [template]
// CHECK:STDOUT:   %Uint32ToInt32: %Uint32ToInt32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %.2: type = int_type unsigned, %.1 [template]
// CHECK:STDOUT:   %AddU32.type: type = fn_type @AddU32 [template]
// CHECK:STDOUT:   %AddU32: %AddU32.type = struct_value () [template]
// CHECK:STDOUT:   %Int32ToUint32.type: type = fn_type @Int32ToUint32 [template]
// CHECK:STDOUT:   %Int32ToUint32: %Int32ToUint32.type = struct_value () [template]
// CHECK:STDOUT:   %.3: i32 = int_value 2147483647 [template]
// CHECK:STDOUT:   %.4: %.2 = int_value 2147483647 [template]
// CHECK:STDOUT:   %.5: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.6: %.2 = int_value 1 [template]
// CHECK:STDOUT:   %.7: %.2 = int_value 2147483648 [template]
// CHECK:STDOUT:   %.8: i32 = int_value 2147483648 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//int_ops, inst+25, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//int_ops, inst+49, unloaded
// CHECK:STDOUT:   %import_ref.3: %AddU32.type = import_ref Main//int_ops, inst+89, loaded [template = constants.%AddU32]
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//int_ops, inst+96, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//int_ops, inst+113, unloaded
// CHECK:STDOUT:   %import_ref.6: %Int32ToUint32.type = import_ref Main//int_ops, inst+131, loaded [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %import_ref.7: %Uint32ToInt32.type = import_ref Main//int_ops, inst+149, loaded [template = constants.%Uint32ToInt32]
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, inst+168, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//int_ops, inst+187, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//int_ops, inst+218, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//int_ops, inst+237, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//int_ops, inst+256, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//int_ops, inst+275, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//int_ops, inst+294, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//int_ops, inst+313, unloaded
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//int_ops, inst+333, unloaded
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//int_ops, inst+352, unloaded
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//int_ops, inst+371, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//int_ops, inst+390, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Main//int_ops, inst+408, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, inst+427, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.22
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
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_19.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc5_19.2: type = converted %int.make_type_32, %.loc5_19.1 [template = i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToInt32(%a.param_patt: %.2) -> i32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddU32(%a.param_patt: %.2, %b.param_patt: %.2) -> %.2 = "int.uadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint32(%a.param_patt: i32) -> %.2 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Uint32ToInt32.ref: %Uint32ToInt32.type = name_ref Uint32ToInt32, imports.%import_ref.7 [template = constants.%Uint32ToInt32]
// CHECK:STDOUT:   %AddU32.ref: %AddU32.type = name_ref AddU32, imports.%import_ref.3 [template = constants.%AddU32]
// CHECK:STDOUT:   %Int32ToUint32.ref.loc11: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %.loc11_26: i32 = int_value 2147483647 [template = constants.%.3]
// CHECK:STDOUT:   %int.convert_checked.loc11: init %.2 = call %Int32ToUint32.ref.loc11(%.loc11_26) [template = constants.%.4]
// CHECK:STDOUT:   %Int32ToUint32.ref.loc12: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %.loc12_26: i32 = int_value 1 [template = constants.%.5]
// CHECK:STDOUT:   %int.convert_checked.loc12: init %.2 = call %Int32ToUint32.ref.loc12(%.loc12_26) [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_25.1: %.2 = value_of_initializer %int.convert_checked.loc11 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_25.2: %.2 = converted %int.convert_checked.loc11, %.loc11_25.1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc12_25.1: %.2 = value_of_initializer %int.convert_checked.loc12 [template = constants.%.6]
// CHECK:STDOUT:   %.loc12_25.2: %.2 = converted %int.convert_checked.loc12, %.loc12_25.1 [template = constants.%.6]
// CHECK:STDOUT:   %int.uadd: init %.2 = call %AddU32.ref(%.loc11_25.2, %.loc12_25.2) [template = constants.%.7]
// CHECK:STDOUT:   %.loc11_11.1: %.2 = value_of_initializer %int.uadd [template = constants.%.7]
// CHECK:STDOUT:   %.loc11_11.2: %.2 = converted %int.uadd, %.loc11_11.1 [template = constants.%.7]
// CHECK:STDOUT:   %int.convert_checked.loc10: init i32 = call %Uint32ToInt32.ref(%.loc11_11.2) [template = constants.%.8]
// CHECK:STDOUT:   %.loc12_30.1: i32 = value_of_initializer %int.convert_checked.loc10 [template = constants.%.8]
// CHECK:STDOUT:   %.loc12_30.2: i32 = converted %int.convert_checked.loc10, %.loc12_30.1 [template = constants.%.8]
// CHECK:STDOUT:   %max_plus_one: i32 = bind_name max_plus_one, %.loc12_30.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_too_large_i32_for_i16.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 16 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %Int32ToInt16.type: type = fn_type @Int32ToInt16 [template]
// CHECK:STDOUT:   %Int32ToInt16: %Int32ToInt16.type = struct_value () [template]
// CHECK:STDOUT:   %.3: i32 = int_value 32768 [template]
// CHECK:STDOUT:   %.4: %.2 = int_value 32768 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//int_ops, inst+25, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//int_ops, inst+49, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//int_ops, inst+89, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//int_ops, inst+96, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//int_ops, inst+113, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//int_ops, inst+131, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//int_ops, inst+149, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, inst+168, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//int_ops, inst+187, unloaded
// CHECK:STDOUT:   %import_ref.10: %Int32ToInt16.type = import_ref Main//int_ops, inst+218, loaded [template = constants.%Int32ToInt16]
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//int_ops, inst+237, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//int_ops, inst+256, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//int_ops, inst+275, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//int_ops, inst+294, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//int_ops, inst+313, unloaded
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//int_ops, inst+333, unloaded
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//int_ops, inst+352, unloaded
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//int_ops, inst+371, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//int_ops, inst+390, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Main//int_ops, inst+408, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, inst+427, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.22
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
// CHECK:STDOUT:   %.loc9_19.1: Core.IntLiteral = int_value 16 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc9_19.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc9_19.2: type = value_of_initializer %int.make_type_signed [template = constants.%.2]
// CHECK:STDOUT:   %.loc9_19.3: type = converted %int.make_type_signed, %.loc9_19.2 [template = constants.%.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToInt16(%a.param_patt: i32) -> %.2 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Int32ToInt16.ref: %Int32ToInt16.type = name_ref Int32ToInt16, imports.%import_ref.10 [template = constants.%Int32ToInt16]
// CHECK:STDOUT:   %.loc9_38: i32 = int_value 32768 [template = constants.%.3]
// CHECK:STDOUT:   %int.convert_checked: init %.2 = call %Int32ToInt16.ref(%.loc9_38) [template = constants.%.4]
// CHECK:STDOUT:   %.loc9_45.1: %.2 = value_of_initializer %int.convert_checked [template = constants.%.4]
// CHECK:STDOUT:   %.loc9_45.2: %.2 = converted %int.convert_checked, %.loc9_45.1 [template = constants.%.4]
// CHECK:STDOUT:   %max_plus_one: %.2 = bind_name max_plus_one, %.loc9_45.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_too_large_i32_for_u16.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 16 [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type unsigned, %.1 [template]
// CHECK:STDOUT:   %Int32ToUint16.type: type = fn_type @Int32ToUint16 [template]
// CHECK:STDOUT:   %Int32ToUint16: %Int32ToUint16.type = struct_value () [template]
// CHECK:STDOUT:   %.3: i32 = int_value 65536 [template]
// CHECK:STDOUT:   %.4: %.2 = int_value 65536 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//int_ops, inst+25, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//int_ops, inst+49, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//int_ops, inst+89, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//int_ops, inst+96, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//int_ops, inst+113, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//int_ops, inst+131, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//int_ops, inst+149, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, inst+168, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//int_ops, inst+187, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//int_ops, inst+218, unloaded
// CHECK:STDOUT:   %import_ref.11: %Int32ToUint16.type = import_ref Main//int_ops, inst+237, loaded [template = constants.%Int32ToUint16]
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//int_ops, inst+256, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//int_ops, inst+275, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//int_ops, inst+294, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//int_ops, inst+313, unloaded
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//int_ops, inst+333, unloaded
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//int_ops, inst+352, unloaded
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//int_ops, inst+371, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//int_ops, inst+390, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Main//int_ops, inst+408, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, inst+427, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .UInt = %import_ref.22
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
// CHECK:STDOUT:   %.loc9_19.1: Core.IntLiteral = int_value 16 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_unsigned: init type = call constants.%UInt(%.loc9_19.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc9_19.2: type = value_of_initializer %int.make_type_unsigned [template = constants.%.2]
// CHECK:STDOUT:   %.loc9_19.3: type = converted %int.make_type_unsigned, %.loc9_19.2 [template = constants.%.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint16(%a.param_patt: i32) -> %.2 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Int32ToUint16.ref: %Int32ToUint16.type = name_ref Int32ToUint16, imports.%import_ref.11 [template = constants.%Int32ToUint16]
// CHECK:STDOUT:   %.loc9_39: i32 = int_value 65536 [template = constants.%.3]
// CHECK:STDOUT:   %int.convert_checked: init %.2 = call %Int32ToUint16.ref(%.loc9_39) [template = constants.%.4]
// CHECK:STDOUT:   %.loc9_48.1: %.2 = value_of_initializer %int.convert_checked [template = constants.%.4]
// CHECK:STDOUT:   %.loc9_48.2: %.2 = converted %int.convert_checked, %.loc9_48.1 [template = constants.%.4]
// CHECK:STDOUT:   %max_plus_one: %.2 = bind_name max_plus_one, %.loc9_48.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_too_large_u32_for_i16.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 16 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %Uint32ToInt16.type: type = fn_type @Uint32ToInt16 [template]
// CHECK:STDOUT:   %Uint32ToInt16: %Uint32ToInt16.type = struct_value () [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %.4: type = int_type unsigned, %.3 [template]
// CHECK:STDOUT:   %Int32ToUint32.type: type = fn_type @Int32ToUint32 [template]
// CHECK:STDOUT:   %Int32ToUint32: %Int32ToUint32.type = struct_value () [template]
// CHECK:STDOUT:   %.5: i32 = int_value 32768 [template]
// CHECK:STDOUT:   %.6: %.4 = int_value 32768 [template]
// CHECK:STDOUT:   %.7: %.2 = int_value 32768 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//int_ops, inst+25, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//int_ops, inst+49, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//int_ops, inst+89, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//int_ops, inst+96, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//int_ops, inst+113, unloaded
// CHECK:STDOUT:   %import_ref.6: %Int32ToUint32.type = import_ref Main//int_ops, inst+131, loaded [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//int_ops, inst+149, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, inst+168, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//int_ops, inst+187, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//int_ops, inst+218, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//int_ops, inst+237, unloaded
// CHECK:STDOUT:   %import_ref.12: %Uint32ToInt16.type = import_ref Main//int_ops, inst+256, loaded [template = constants.%Uint32ToInt16]
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//int_ops, inst+275, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//int_ops, inst+294, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//int_ops, inst+313, unloaded
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//int_ops, inst+333, unloaded
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//int_ops, inst+352, unloaded
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//int_ops, inst+371, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//int_ops, inst+390, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Main//int_ops, inst+408, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, inst+427, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.22
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
// CHECK:STDOUT:   %.loc9_19.1: Core.IntLiteral = int_value 16 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc9_19.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc9_19.2: type = value_of_initializer %int.make_type_signed [template = constants.%.2]
// CHECK:STDOUT:   %.loc9_19.3: type = converted %int.make_type_signed, %.loc9_19.2 [template = constants.%.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToInt16(%a.param_patt: %.4) -> %.2 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint32(%a.param_patt: i32) -> %.4 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Uint32ToInt16.ref: %Uint32ToInt16.type = name_ref Uint32ToInt16, imports.%import_ref.12 [template = constants.%Uint32ToInt16]
// CHECK:STDOUT:   %Int32ToUint32.ref: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %.loc9_53: i32 = int_value 32768 [template = constants.%.5]
// CHECK:STDOUT:   %int.convert_checked.loc9_52: init %.4 = call %Int32ToUint32.ref(%.loc9_53) [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_52.1: %.4 = value_of_initializer %int.convert_checked.loc9_52 [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_52.2: %.4 = converted %int.convert_checked.loc9_52, %.loc9_52.1 [template = constants.%.6]
// CHECK:STDOUT:   %int.convert_checked.loc9_38: init %.2 = call %Uint32ToInt16.ref(%.loc9_52.2) [template = constants.%.7]
// CHECK:STDOUT:   %.loc9_61.1: %.2 = value_of_initializer %int.convert_checked.loc9_38 [template = constants.%.7]
// CHECK:STDOUT:   %.loc9_61.2: %.2 = converted %int.convert_checked.loc9_38, %.loc9_61.1 [template = constants.%.7]
// CHECK:STDOUT:   %max_plus_one: %.2 = bind_name max_plus_one, %.loc9_61.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_too_large_u32_for_u16.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 16 [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type unsigned, %.1 [template]
// CHECK:STDOUT:   %Uint32ToUint16.type: type = fn_type @Uint32ToUint16 [template]
// CHECK:STDOUT:   %Uint32ToUint16: %Uint32ToUint16.type = struct_value () [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %.4: type = int_type unsigned, %.3 [template]
// CHECK:STDOUT:   %Int32ToUint32.type: type = fn_type @Int32ToUint32 [template]
// CHECK:STDOUT:   %Int32ToUint32: %Int32ToUint32.type = struct_value () [template]
// CHECK:STDOUT:   %.5: i32 = int_value 65536 [template]
// CHECK:STDOUT:   %.6: %.4 = int_value 65536 [template]
// CHECK:STDOUT:   %.7: %.2 = int_value 65536 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//int_ops, inst+25, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//int_ops, inst+49, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//int_ops, inst+89, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//int_ops, inst+96, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//int_ops, inst+113, unloaded
// CHECK:STDOUT:   %import_ref.6: %Int32ToUint32.type = import_ref Main//int_ops, inst+131, loaded [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//int_ops, inst+149, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, inst+168, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//int_ops, inst+187, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//int_ops, inst+218, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//int_ops, inst+237, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//int_ops, inst+256, unloaded
// CHECK:STDOUT:   %import_ref.13: %Uint32ToUint16.type = import_ref Main//int_ops, inst+275, loaded [template = constants.%Uint32ToUint16]
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//int_ops, inst+294, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//int_ops, inst+313, unloaded
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//int_ops, inst+333, unloaded
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//int_ops, inst+352, unloaded
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//int_ops, inst+371, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//int_ops, inst+390, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Main//int_ops, inst+408, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, inst+427, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .UInt = %import_ref.22
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
// CHECK:STDOUT:   %.loc9_19.1: Core.IntLiteral = int_value 16 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_unsigned: init type = call constants.%UInt(%.loc9_19.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc9_19.2: type = value_of_initializer %int.make_type_unsigned [template = constants.%.2]
// CHECK:STDOUT:   %.loc9_19.3: type = converted %int.make_type_unsigned, %.loc9_19.2 [template = constants.%.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Uint32ToUint16(%a.param_patt: %.4) -> %.2 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint32(%a.param_patt: i32) -> %.4 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Uint32ToUint16.ref: %Uint32ToUint16.type = name_ref Uint32ToUint16, imports.%import_ref.13 [template = constants.%Uint32ToUint16]
// CHECK:STDOUT:   %Int32ToUint32.ref: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %.loc9_54: i32 = int_value 65536 [template = constants.%.5]
// CHECK:STDOUT:   %int.convert_checked.loc9_53: init %.4 = call %Int32ToUint32.ref(%.loc9_54) [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_53.1: %.4 = value_of_initializer %int.convert_checked.loc9_53 [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_53.2: %.4 = converted %int.convert_checked.loc9_53, %.loc9_53.1 [template = constants.%.6]
// CHECK:STDOUT:   %int.convert_checked.loc9_39: init %.2 = call %Uint32ToUint16.ref(%.loc9_53.2) [template = constants.%.7]
// CHECK:STDOUT:   %.loc9_64.1: %.2 = value_of_initializer %int.convert_checked.loc9_39 [template = constants.%.7]
// CHECK:STDOUT:   %.loc9_64.2: %.2 = converted %int.convert_checked.loc9_39, %.loc9_64.1 [template = constants.%.7]
// CHECK:STDOUT:   %max_plus_one: %.2 = bind_name max_plus_one, %.loc9_64.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_negative_i32_to_u16.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 16 [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type unsigned, %.1 [template]
// CHECK:STDOUT:   %Int32ToUint16.type: type = fn_type @Int32ToUint16 [template]
// CHECK:STDOUT:   %Int32ToUint16: %Int32ToUint16.type = struct_value () [template]
// CHECK:STDOUT:   %SubI32.type: type = fn_type @SubI32 [template]
// CHECK:STDOUT:   %SubI32: %SubI32.type = struct_value () [template]
// CHECK:STDOUT:   %.3: i32 = int_value 0 [template]
// CHECK:STDOUT:   %.4: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_value -1 [template]
// CHECK:STDOUT:   %.6: %.2 = int_value 18446744073709551615 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//int_ops, inst+25, unloaded
// CHECK:STDOUT:   %import_ref.2: %SubI32.type = import_ref Main//int_ops, inst+49, loaded [template = constants.%SubI32]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//int_ops, inst+89, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//int_ops, inst+96, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//int_ops, inst+113, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//int_ops, inst+131, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//int_ops, inst+149, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, inst+168, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//int_ops, inst+187, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//int_ops, inst+218, unloaded
// CHECK:STDOUT:   %import_ref.11: %Int32ToUint16.type = import_ref Main//int_ops, inst+237, loaded [template = constants.%Int32ToUint16]
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//int_ops, inst+256, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//int_ops, inst+275, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//int_ops, inst+294, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//int_ops, inst+313, unloaded
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//int_ops, inst+333, unloaded
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//int_ops, inst+352, unloaded
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//int_ops, inst+371, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//int_ops, inst+390, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Main//int_ops, inst+408, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, inst+427, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .UInt = %import_ref.22
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
// CHECK:STDOUT:   %.loc9_23.1: Core.IntLiteral = int_value 16 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_unsigned: init type = call constants.%UInt(%.loc9_23.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc9_23.2: type = value_of_initializer %int.make_type_unsigned [template = constants.%.2]
// CHECK:STDOUT:   %.loc9_23.3: type = converted %int.make_type_unsigned, %.loc9_23.2 [template = constants.%.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint16(%a.param_patt: i32) -> %.2 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubI32(%a.param_patt: i32, %b.param_patt: i32) -> i32 = "int.ssub";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Int32ToUint16.ref: %Int32ToUint16.type = name_ref Int32ToUint16, imports.%import_ref.11 [template = constants.%Int32ToUint16]
// CHECK:STDOUT:   %SubI32.ref: %SubI32.type = name_ref SubI32, imports.%import_ref.2 [template = constants.%SubI32]
// CHECK:STDOUT:   %.loc9_50: i32 = int_value 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc9_53: i32 = int_value 1 [template = constants.%.4]
// CHECK:STDOUT:   %int.ssub: init i32 = call %SubI32.ref(%.loc9_50, %.loc9_53) [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_49.1: i32 = value_of_initializer %int.ssub [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_49.2: i32 = converted %int.ssub, %.loc9_49.1 [template = constants.%.5]
// CHECK:STDOUT:   %int.convert_checked: init %.2 = call %Int32ToUint16.ref(%.loc9_49.2) [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_56.1: %.2 = value_of_initializer %int.convert_checked [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_56.2: %.2 = converted %int.convert_checked, %.loc9_56.1 [template = constants.%.6]
// CHECK:STDOUT:   %minus_one_to_u16: %.2 = bind_name minus_one_to_u16, %.loc9_56.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_negative_i32_to_u32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type unsigned, %.1 [template]
// CHECK:STDOUT:   %Int32ToUint32.type: type = fn_type @Int32ToUint32 [template]
// CHECK:STDOUT:   %Int32ToUint32: %Int32ToUint32.type = struct_value () [template]
// CHECK:STDOUT:   %SubI32.type: type = fn_type @SubI32 [template]
// CHECK:STDOUT:   %SubI32: %SubI32.type = struct_value () [template]
// CHECK:STDOUT:   %.3: i32 = int_value 0 [template]
// CHECK:STDOUT:   %.4: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_value -1 [template]
// CHECK:STDOUT:   %.6: %.2 = int_value 18446744073709551615 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//int_ops, inst+25, unloaded
// CHECK:STDOUT:   %import_ref.2: %SubI32.type = import_ref Main//int_ops, inst+49, loaded [template = constants.%SubI32]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//int_ops, inst+89, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//int_ops, inst+96, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//int_ops, inst+113, unloaded
// CHECK:STDOUT:   %import_ref.6: %Int32ToUint32.type = import_ref Main//int_ops, inst+131, loaded [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//int_ops, inst+149, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, inst+168, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//int_ops, inst+187, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//int_ops, inst+218, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//int_ops, inst+237, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//int_ops, inst+256, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//int_ops, inst+275, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//int_ops, inst+294, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//int_ops, inst+313, unloaded
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//int_ops, inst+333, unloaded
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//int_ops, inst+352, unloaded
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//int_ops, inst+371, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//int_ops, inst+390, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Main//int_ops, inst+408, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, inst+427, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .UInt = %import_ref.22
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
// CHECK:STDOUT:   %.loc9_23.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_unsigned: init type = call constants.%UInt(%.loc9_23.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc9_23.2: type = value_of_initializer %int.make_type_unsigned [template = constants.%.2]
// CHECK:STDOUT:   %.loc9_23.3: type = converted %int.make_type_unsigned, %.loc9_23.2 [template = constants.%.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint32(%a.param_patt: i32) -> %.2 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubI32(%a.param_patt: i32, %b.param_patt: i32) -> i32 = "int.ssub";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Int32ToUint32.ref: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%import_ref.6 [template = constants.%Int32ToUint32]
// CHECK:STDOUT:   %SubI32.ref: %SubI32.type = name_ref SubI32, imports.%import_ref.2 [template = constants.%SubI32]
// CHECK:STDOUT:   %.loc9_50: i32 = int_value 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc9_53: i32 = int_value 1 [template = constants.%.4]
// CHECK:STDOUT:   %int.ssub: init i32 = call %SubI32.ref(%.loc9_50, %.loc9_53) [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_49.1: i32 = value_of_initializer %int.ssub [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_49.2: i32 = converted %int.ssub, %.loc9_49.1 [template = constants.%.5]
// CHECK:STDOUT:   %int.convert_checked: init %.2 = call %Int32ToUint32.ref(%.loc9_49.2) [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_56.1: %.2 = value_of_initializer %int.convert_checked [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_56.2: %.2 = converted %int.convert_checked, %.loc9_56.1 [template = constants.%.6]
// CHECK:STDOUT:   %minus_one_to_u32: %.2 = bind_name minus_one_to_u32, %.loc9_56.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_negative_i32_to_u64.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type unsigned, %.1 [template]
// CHECK:STDOUT:   %Int32ToUint64.type: type = fn_type @Int32ToUint64 [template]
// CHECK:STDOUT:   %Int32ToUint64: %Int32ToUint64.type = struct_value () [template]
// CHECK:STDOUT:   %SubI32.type: type = fn_type @SubI32 [template]
// CHECK:STDOUT:   %SubI32: %SubI32.type = struct_value () [template]
// CHECK:STDOUT:   %.3: i32 = int_value 0 [template]
// CHECK:STDOUT:   %.4: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_value -1 [template]
// CHECK:STDOUT:   %.6: %.2 = int_value 18446744073709551615 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//int_ops, inst+25, unloaded
// CHECK:STDOUT:   %import_ref.2: %SubI32.type = import_ref Main//int_ops, inst+49, loaded [template = constants.%SubI32]
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//int_ops, inst+89, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//int_ops, inst+96, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//int_ops, inst+113, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//int_ops, inst+131, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//int_ops, inst+149, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, inst+168, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//int_ops, inst+187, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//int_ops, inst+218, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//int_ops, inst+237, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//int_ops, inst+256, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//int_ops, inst+275, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//int_ops, inst+294, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//int_ops, inst+313, unloaded
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//int_ops, inst+333, unloaded
// CHECK:STDOUT:   %import_ref.17: %Int32ToUint64.type = import_ref Main//int_ops, inst+352, loaded [template = constants.%Int32ToUint64]
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//int_ops, inst+371, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//int_ops, inst+390, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Main//int_ops, inst+408, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, inst+427, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .UInt = %import_ref.22
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
// CHECK:STDOUT:   %.loc9_23.1: Core.IntLiteral = int_value 64 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_unsigned: init type = call constants.%UInt(%.loc9_23.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc9_23.2: type = value_of_initializer %int.make_type_unsigned [template = constants.%.2]
// CHECK:STDOUT:   %.loc9_23.3: type = converted %int.make_type_unsigned, %.loc9_23.2 [template = constants.%.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToUint64(%a.param_patt: i32) -> %.2 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubI32(%a.param_patt: i32, %b.param_patt: i32) -> i32 = "int.ssub";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Int32ToUint64.ref: %Int32ToUint64.type = name_ref Int32ToUint64, imports.%import_ref.17 [template = constants.%Int32ToUint64]
// CHECK:STDOUT:   %SubI32.ref: %SubI32.type = name_ref SubI32, imports.%import_ref.2 [template = constants.%SubI32]
// CHECK:STDOUT:   %.loc9_50: i32 = int_value 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc9_53: i32 = int_value 1 [template = constants.%.4]
// CHECK:STDOUT:   %int.ssub: init i32 = call %SubI32.ref(%.loc9_50, %.loc9_53) [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_49.1: i32 = value_of_initializer %int.ssub [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_49.2: i32 = converted %int.ssub, %.loc9_49.1 [template = constants.%.5]
// CHECK:STDOUT:   %int.convert_checked: init %.2 = call %Int32ToUint64.ref(%.loc9_49.2) [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_56.1: %.2 = value_of_initializer %int.convert_checked [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_56.2: %.2 = converted %int.convert_checked, %.loc9_56.1 [template = constants.%.6]
// CHECK:STDOUT:   %minus_one_to_u64: %.2 = bind_name minus_one_to_u64, %.loc9_56.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_too_small_i32_for_i16.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 16 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %Int32ToInt16.type: type = fn_type @Int32ToInt16 [template]
// CHECK:STDOUT:   %Int32ToInt16: %Int32ToInt16.type = struct_value () [template]
// CHECK:STDOUT:   %NegateI32.type: type = fn_type @NegateI32 [template]
// CHECK:STDOUT:   %NegateI32: %NegateI32.type = struct_value () [template]
// CHECK:STDOUT:   %.3: i32 = int_value 32769 [template]
// CHECK:STDOUT:   %.4: i32 = int_value -32769 [template]
// CHECK:STDOUT:   %.5: %.2 = int_value -32769 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %NegateI32.type = import_ref Main//int_ops, inst+25, loaded [template = constants.%NegateI32]
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//int_ops, inst+49, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//int_ops, inst+89, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//int_ops, inst+96, unloaded
// CHECK:STDOUT:   %import_ref.5 = import_ref Main//int_ops, inst+113, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//int_ops, inst+131, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//int_ops, inst+149, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, inst+168, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//int_ops, inst+187, unloaded
// CHECK:STDOUT:   %import_ref.10: %Int32ToInt16.type = import_ref Main//int_ops, inst+218, loaded [template = constants.%Int32ToInt16]
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//int_ops, inst+237, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//int_ops, inst+256, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//int_ops, inst+275, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//int_ops, inst+294, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//int_ops, inst+313, unloaded
// CHECK:STDOUT:   %import_ref.16 = import_ref Main//int_ops, inst+333, unloaded
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//int_ops, inst+352, unloaded
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//int_ops, inst+371, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//int_ops, inst+390, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Main//int_ops, inst+408, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, inst+427, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.22
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
// CHECK:STDOUT:   %.loc9_20.1: Core.IntLiteral = int_value 16 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc9_20.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc9_20.2: type = value_of_initializer %int.make_type_signed [template = constants.%.2]
// CHECK:STDOUT:   %.loc9_20.3: type = converted %int.make_type_signed, %.loc9_20.2 [template = constants.%.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToInt16(%a.param_patt: i32) -> %.2 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NegateI32(%a.param_patt: i32) -> i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Int32ToInt16.ref: %Int32ToInt16.type = name_ref Int32ToInt16, imports.%import_ref.10 [template = constants.%Int32ToInt16]
// CHECK:STDOUT:   %NegateI32.ref: %NegateI32.type = name_ref NegateI32, imports.%import_ref.1 [template = constants.%NegateI32]
// CHECK:STDOUT:   %.loc9_49: i32 = int_value 32769 [template = constants.%.3]
// CHECK:STDOUT:   %int.snegate: init i32 = call %NegateI32.ref(%.loc9_49) [template = constants.%.4]
// CHECK:STDOUT:   %.loc9_48.1: i32 = value_of_initializer %int.snegate [template = constants.%.4]
// CHECK:STDOUT:   %.loc9_48.2: i32 = converted %int.snegate, %.loc9_48.1 [template = constants.%.4]
// CHECK:STDOUT:   %int.convert_checked: init %.2 = call %Int32ToInt16.ref(%.loc9_48.2) [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_57.1: %.2 = value_of_initializer %int.convert_checked [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_57.2: %.2 = converted %int.convert_checked, %.loc9_57.1 [template = constants.%.5]
// CHECK:STDOUT:   %min_minus_one: %.2 = bind_name min_minus_one, %.loc9_57.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_not_constant.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: i32 = int_value 0 [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 16 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = int_type signed, %.2 [template]
// CHECK:STDOUT:   %Int32ToInt16.type: type = fn_type @Int32ToInt16 [template]
// CHECK:STDOUT:   %Int32ToInt16: %Int32ToInt16.type = struct_value () [template]
// CHECK:STDOUT:   %Int32ToInt32.type: type = fn_type @Int32ToInt32 [template]
// CHECK:STDOUT:   %Int32ToInt32: %Int32ToInt32.type = struct_value () [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %.5: type = int_type signed, %.4 [template]
// CHECK:STDOUT:   %Int32ToInt64.type: type = fn_type @Int32ToInt64 [template]
// CHECK:STDOUT:   %Int32ToInt64: %Int32ToInt64.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//int_ops, inst+25, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//int_ops, inst+49, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//int_ops, inst+89, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//int_ops, inst+96, unloaded
// CHECK:STDOUT:   %import_ref.5: %Int32ToInt32.type = import_ref Main//int_ops, inst+113, loaded [template = constants.%Int32ToInt32]
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//int_ops, inst+131, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//int_ops, inst+149, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//int_ops, inst+168, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//int_ops, inst+187, unloaded
// CHECK:STDOUT:   %import_ref.10: %Int32ToInt16.type = import_ref Main//int_ops, inst+218, loaded [template = constants.%Int32ToInt16]
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//int_ops, inst+237, unloaded
// CHECK:STDOUT:   %import_ref.12 = import_ref Main//int_ops, inst+256, unloaded
// CHECK:STDOUT:   %import_ref.13 = import_ref Main//int_ops, inst+275, unloaded
// CHECK:STDOUT:   %import_ref.14 = import_ref Main//int_ops, inst+294, unloaded
// CHECK:STDOUT:   %import_ref.15 = import_ref Main//int_ops, inst+313, unloaded
// CHECK:STDOUT:   %import_ref.16: %Int32ToInt64.type = import_ref Main//int_ops, inst+333, loaded [template = constants.%Int32ToInt64]
// CHECK:STDOUT:   %import_ref.17 = import_ref Main//int_ops, inst+352, unloaded
// CHECK:STDOUT:   %import_ref.18 = import_ref Main//int_ops, inst+371, unloaded
// CHECK:STDOUT:   %import_ref.19 = import_ref Main//int_ops, inst+390, unloaded
// CHECK:STDOUT:   %import_ref.20 = import_ref Main//int_ops, inst+408, unloaded
// CHECK:STDOUT:   %import_ref.21 = import_ref Main//int_ops, inst+427, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.22
// CHECK:STDOUT:     .Int = %import_ref.23
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
// CHECK:STDOUT:   %int.make_type_32.loc5: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_19.1: type = value_of_initializer %int.make_type_32.loc5 [template = i32]
// CHECK:STDOUT:   %.loc5_19.2: type = converted %int.make_type_32.loc5, %.loc5_19.1 [template = i32]
// CHECK:STDOUT:   %.loc15_34.1: Core.IntLiteral = int_value 16 [template = constants.%.2]
// CHECK:STDOUT:   %int.make_type_signed.loc15: init type = call constants.%Int(%.loc15_34.1) [template = constants.%.3]
// CHECK:STDOUT:   %.loc15_34.2: type = value_of_initializer %int.make_type_signed.loc15 [template = constants.%.3]
// CHECK:STDOUT:   %.loc15_34.3: type = converted %int.make_type_signed.loc15, %.loc15_34.2 [template = constants.%.3]
// CHECK:STDOUT:   %int.make_type_32.loc25: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc25_32.1: type = value_of_initializer %int.make_type_32.loc25 [template = i32]
// CHECK:STDOUT:   %.loc25_32.2: type = converted %int.make_type_32.loc25, %.loc25_32.1 [template = i32]
// CHECK:STDOUT:   %.loc34_33.1: Core.IntLiteral = int_value 64 [template = constants.%.4]
// CHECK:STDOUT:   %int.make_type_signed.loc34: init type = call constants.%Int(%.loc34_33.1) [template = constants.%.5]
// CHECK:STDOUT:   %.loc34_33.2: type = value_of_initializer %int.make_type_signed.loc34 [template = constants.%.5]
// CHECK:STDOUT:   %.loc34_33.3: type = converted %int.make_type_signed.loc34, %.loc34_33.2 [template = constants.%.5]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToInt16(%a.param_patt: i32) -> %.3 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToInt32(%a.param_patt: i32) -> i32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32ToInt64(%a.param_patt: i32) -> %.5 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc5: i32 = int_value 0 [template = constants.%.1]
// CHECK:STDOUT:   %not_constant: i32 = bind_name not_constant, %.loc5
// CHECK:STDOUT:   %Int32ToInt16.ref: %Int32ToInt16.type = name_ref Int32ToInt16, imports.%import_ref.10 [template = constants.%Int32ToInt16]
// CHECK:STDOUT:   %not_constant.ref.loc15: i32 = name_ref not_constant, %not_constant
// CHECK:STDOUT:   %int.convert_checked.loc15: init %.3 = call %Int32ToInt16.ref(%not_constant.ref.loc15)
// CHECK:STDOUT:   %.loc15_66.1: %.3 = value_of_initializer %int.convert_checked.loc15
// CHECK:STDOUT:   %.loc15_66.2: %.3 = converted %int.convert_checked.loc15, %.loc15_66.1
// CHECK:STDOUT:   %convert_not_constant_narrow: %.3 = bind_name convert_not_constant_narrow, %.loc15_66.2
// CHECK:STDOUT:   %Int32ToInt32.ref: %Int32ToInt32.type = name_ref Int32ToInt32, imports.%import_ref.5 [template = constants.%Int32ToInt32]
// CHECK:STDOUT:   %not_constant.ref.loc25: i32 = name_ref not_constant, %not_constant
// CHECK:STDOUT:   %int.convert_checked.loc25: init i32 = call %Int32ToInt32.ref(%not_constant.ref.loc25)
// CHECK:STDOUT:   %.loc25_64.1: i32 = value_of_initializer %int.convert_checked.loc25
// CHECK:STDOUT:   %.loc25_64.2: i32 = converted %int.convert_checked.loc25, %.loc25_64.1
// CHECK:STDOUT:   %convert_not_constant_same: i32 = bind_name convert_not_constant_same, %.loc25_64.2
// CHECK:STDOUT:   %Int32ToInt64.ref: %Int32ToInt64.type = name_ref Int32ToInt64, imports.%import_ref.16 [template = constants.%Int32ToInt64]
// CHECK:STDOUT:   %not_constant.ref.loc34: i32 = name_ref not_constant, %not_constant
// CHECK:STDOUT:   %int.convert_checked.loc34: init %.5 = call %Int32ToInt64.ref(%not_constant.ref.loc34)
// CHECK:STDOUT:   %.loc34_65.1: %.5 = value_of_initializer %int.convert_checked.loc34
// CHECK:STDOUT:   %.loc34_65.2: %.5 = converted %int.convert_checked.loc34, %.loc34_65.1
// CHECK:STDOUT:   %convert_not_constant_widen: %.5 = bind_name convert_not_constant_widen, %.loc34_65.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
