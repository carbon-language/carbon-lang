// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/uint.carbon
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

// --- runtime_call.carbon

library "[[@TEST_NAME]]";
import library "int_ops";

//@dump-sem-ir-begin
let SizePreserving: u32 = Int32ToUint32(1);
let Narrowing: i16 = Int32ToInt16(1);
let Widening: i64 = Int32ToInt64(1);
//@dump-sem-ir-end

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

// CHECK:STDOUT: --- runtime_call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.4a9: type = pattern_type %u32 [concrete]
// CHECK:STDOUT:   %Int32ToUint32.type: type = fn_type @Int32ToUint32 [concrete]
// CHECK:STDOUT:   %Int32ToUint32: %Int32ToUint32.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cf3: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.6da: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.8a0: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.1d6: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.8a0 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ed2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.ba4, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.5a7: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.7c1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.5a7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.cf3 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ed2) [concrete]
// CHECK:STDOUT:   %.2c2: type = fn_type_with_self_type %ImplicitAs.Convert.type.6da, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.7c1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.7c1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.47b: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_1.c1d: %u32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.88f: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %Int32ToInt16.type: type = fn_type @Int32ToInt16 [concrete]
// CHECK:STDOUT:   %Int32ToInt16: %Int32ToInt16.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.c22: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.a10: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %Int32ToInt64.type: type = fn_type @Int32ToInt64 [concrete]
// CHECK:STDOUT:   %Int32ToInt64: %Int32ToInt64.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.a95: %i64 = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Int32ToUint32: %Int32ToUint32.type = import_ref Main//int_ops, Int32ToUint32, loaded [concrete = constants.%Int32ToUint32]
// CHECK:STDOUT:   %Main.Int32ToInt16: %Int32ToInt16.type = import_ref Main//int_ops, Int32ToInt16, loaded [concrete = constants.%Int32ToInt16]
// CHECK:STDOUT:   %Main.Int32ToInt64: %Int32ToInt64.type = import_ref Main//int_ops, Int32ToInt64, loaded [concrete = constants.%Int32ToInt64]
// CHECK:STDOUT:   %Core.import_ref.7bb: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.8a0) = import_ref Core//prelude/parts/int, loc23_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.1d6)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.ba4 = impl_witness_table (%Core.import_ref.7bb), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %SizePreserving.patt: %pattern_type.4a9 = binding_pattern SizePreserving [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_21: type = splice_block %u32 [concrete = constants.%u32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %u32: type = class_type @UInt, @UInt(constants.%int_32) [concrete = constants.%u32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6_42.1: %u32 = value_of_initializer @__global_init.%Int32ToUint32.call [concrete = constants.%int_1.c1d]
// CHECK:STDOUT:   %.loc6_42.2: %u32 = converted @__global_init.%Int32ToUint32.call, %.loc6_42.1 [concrete = constants.%int_1.c1d]
// CHECK:STDOUT:   %SizePreserving: %u32 = bind_name SizePreserving, %.loc6_42.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %Narrowing.patt: %pattern_type.88f = binding_pattern Narrowing [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_16: type = splice_block %i16 [concrete = constants.%i16] {
// CHECK:STDOUT:     %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:     %i16: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_36.1: %i16 = value_of_initializer @__global_init.%Int32ToInt16.call [concrete = constants.%int_1.c22]
// CHECK:STDOUT:   %.loc7_36.2: %i16 = converted @__global_init.%Int32ToInt16.call, %.loc7_36.1 [concrete = constants.%int_1.c22]
// CHECK:STDOUT:   %Narrowing: %i16 = bind_name Narrowing, %.loc7_36.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %Widening.patt: %pattern_type.a10 = binding_pattern Widening [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_15: type = splice_block %i64 [concrete = constants.%i64] {
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %i64: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_35.1: %i64 = value_of_initializer @__global_init.%Int32ToInt64.call [concrete = constants.%int_1.a95]
// CHECK:STDOUT:   %.loc8_35.2: %i64 = converted @__global_init.%Int32ToInt64.call, %.loc8_35.1 [concrete = constants.%int_1.a95]
// CHECK:STDOUT:   %Widening: %i64 = bind_name Widening, %.loc8_35.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Int32ToUint32.ref: %Int32ToUint32.type = name_ref Int32ToUint32, imports.%Main.Int32ToUint32 [concrete = constants.%Int32ToUint32]
// CHECK:STDOUT:   %int_1.loc6: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc6: %.2c2 = impl_witness_access constants.%ImplicitAs.impl_witness.ed2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.7c1]
// CHECK:STDOUT:   %bound_method.loc6_41.1: <bound method> = bound_method %int_1.loc6, %impl.elem0.loc6 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_41.2: <bound method> = bound_method %int_1.loc6, %specific_fn.loc6 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6: init %i32 = call %bound_method.loc6_41.2(%int_1.loc6) [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %.loc6_41.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6 [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %.loc6_41.2: %i32 = converted %int_1.loc6, %.loc6_41.1 [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %Int32ToUint32.call: init %u32 = call %Int32ToUint32.ref(%.loc6_41.2) [concrete = constants.%int_1.c1d]
// CHECK:STDOUT:   %Int32ToInt16.ref: %Int32ToInt16.type = name_ref Int32ToInt16, imports.%Main.Int32ToInt16 [concrete = constants.%Int32ToInt16]
// CHECK:STDOUT:   %int_1.loc7: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc7: %.2c2 = impl_witness_access constants.%ImplicitAs.impl_witness.ed2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.7c1]
// CHECK:STDOUT:   %bound_method.loc7_35.1: <bound method> = bound_method %int_1.loc7, %impl.elem0.loc7 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_35.2: <bound method> = bound_method %int_1.loc7, %specific_fn.loc7 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7: init %i32 = call %bound_method.loc7_35.2(%int_1.loc7) [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %.loc7_35.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %.loc7_35.2: %i32 = converted %int_1.loc7, %.loc7_35.1 [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %Int32ToInt16.call: init %i16 = call %Int32ToInt16.ref(%.loc7_35.2) [concrete = constants.%int_1.c22]
// CHECK:STDOUT:   %Int32ToInt64.ref: %Int32ToInt64.type = name_ref Int32ToInt64, imports.%Main.Int32ToInt64 [concrete = constants.%Int32ToInt64]
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8: %.2c2 = impl_witness_access constants.%ImplicitAs.impl_witness.ed2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.7c1]
// CHECK:STDOUT:   %bound_method.loc8_34.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_34.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %i32 = call %bound_method.loc8_34.2(%int_1.loc8) [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %.loc8_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %.loc8_34.2: %i32 = converted %int_1.loc8, %.loc8_34.1 [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %Int32ToInt64.call: init %i64 = call %Int32ToInt64.ref(%.loc8_34.2) [concrete = constants.%int_1.a95]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
