// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

fn RuntimeCall(a: i32, b: i32) -> i32 {
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
// CHECK:STDERR: fail_comp_time_only_shift.carbon:[[@LINE+6]]:31: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: let bad4: Core.IntLiteral() = LeftShiftOfLit(12, an_i32);
// CHECK:STDERR:                               ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_comp_time_only_shift.carbon:[[@LINE-39]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR: fn LeftShiftOfLit(a: Core.IntLiteral(), b: i32) -> Core.IntLiteral() = "int.left_shift";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
let bad4: Core.IntLiteral() = LeftShiftOfLit(12, an_i32);

// CHECK:STDOUT: --- i32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %N.2: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.2: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Expect.type: type = generic_class_type @Expect [template]
// CHECK:STDOUT:   %Expect.generic: %Expect.type = struct_value () [template]
// CHECK:STDOUT:   %Expect.1: type = class_type @Expect, @Expect(%N.2) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [template]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [template]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %Expect.1 [symbolic]
// CHECK:STDOUT:   %Expect.val.1: %Expect.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShift.type.1: type = fn_type @LeftShift.1 [template]
// CHECK:STDOUT:   %LeftShift: %LeftShift.type.1 = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.19: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_0.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %Expect.2: type = class_type @Expect, @Expect(%int_0.2) [template]
// CHECK:STDOUT:   %Test.specific_fn.1: <specific function> = specific_function %Test, @Test(%int_0.2) [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int_30.1: Core.IntLiteral = int_value 30 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_30.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_30.2: %i32 = int_value 30 [template]
// CHECK:STDOUT:   %int_1073741824.1: %i32 = int_value 1073741824 [template]
// CHECK:STDOUT:   %Expect.3: type = class_type @Expect, @Expect(%int_1073741824.1) [template]
// CHECK:STDOUT:   %Test.specific_fn.2: <specific function> = specific_function %Test, @Test(%int_1073741824.1) [template]
// CHECK:STDOUT:   %int_1073741824.2: Core.IntLiteral = int_value 1073741824 [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_1073741824.2, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_5.1: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.5: <bound method> = bound_method %int_5.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.5: <specific function> = specific_function %Convert.bound.5, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_5.2: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %Convert.bound.6: <bound method> = bound_method %int_2.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.6: <specific function> = specific_function %Convert.bound.6, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %int_20.1: %i32 = int_value 20 [template]
// CHECK:STDOUT:   %Expect.4: type = class_type @Expect, @Expect(%int_20.1) [template]
// CHECK:STDOUT:   %Test.specific_fn.3: <specific function> = specific_function %Test, @Test(%int_20.1) [template]
// CHECK:STDOUT:   %int_20.2: Core.IntLiteral = int_value 20 [template]
// CHECK:STDOUT:   %Convert.bound.7: <bound method> = bound_method %int_20.2, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.7: <specific function> = specific_function %Convert.bound.7, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %Op.type.13: type = fn_type @Op.13 [template]
// CHECK:STDOUT:   %Op.type.14: type = fn_type @Op.14 [template]
// CHECK:STDOUT:   %Op.14: %Op.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.20: <witness> = interface_witness (%Op.14) [template]
// CHECK:STDOUT:   %Op.bound.1: <bound method> = bound_method %int_1.1, %Op.14 [template]
// CHECK:STDOUT:   %int_-1.1: Core.IntLiteral = int_value -1 [template]
// CHECK:STDOUT:   %Convert.bound.8: <bound method> = bound_method %int_-1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.8: <specific function> = specific_function %Convert.bound.8, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_-1.2: %i32 = int_value -1 [template]
// CHECK:STDOUT:   %int_-2.1: %i32 = int_value -2 [template]
// CHECK:STDOUT:   %Expect.5: type = class_type @Expect, @Expect(%int_-2.1) [template]
// CHECK:STDOUT:   %Test.specific_fn.4: <specific function> = specific_function %Test, @Test(%int_-2.1) [template]
// CHECK:STDOUT:   %Op.bound.2: <bound method> = bound_method %int_2.1, %Op.14 [template]
// CHECK:STDOUT:   %int_-2.2: Core.IntLiteral = int_value -2 [template]
// CHECK:STDOUT:   %Convert.bound.9: <bound method> = bound_method %int_-2.2, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.9: <specific function> = specific_function %Convert.bound.9, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_-4.1: %i32 = int_value -4 [template]
// CHECK:STDOUT:   %Expect.6: type = class_type @Expect, @Expect(%int_-4.1) [template]
// CHECK:STDOUT:   %Test.specific_fn.5: <specific function> = specific_function %Test, @Test(%int_-4.1) [template]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %Op.bound.3: <bound method> = bound_method %int_4, %Op.14 [template]
// CHECK:STDOUT:   %int_-4.2: Core.IntLiteral = int_value -4 [template]
// CHECK:STDOUT:   %Convert.bound.10: <bound method> = bound_method %int_-4.2, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.10: <specific function> = specific_function %Convert.bound.10, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %Op.bound.4: <bound method> = bound_method %int_3, %Op.14 [template]
// CHECK:STDOUT:   %int_-3.1: Core.IntLiteral = int_value -3 [template]
// CHECK:STDOUT:   %Convert.bound.11: <bound method> = bound_method %int_-3.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.11: <specific function> = specific_function %Convert.bound.11, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_-3.2: %i32 = int_value -3 [template]
// CHECK:STDOUT:   %int_-6.1: %i32 = int_value -6 [template]
// CHECK:STDOUT:   %Expect.7: type = class_type @Expect, @Expect(%int_-6.1) [template]
// CHECK:STDOUT:   %Test.specific_fn.6: <specific function> = specific_function %Test, @Test(%int_-6.1) [template]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %Op.bound.5: <bound method> = bound_method %int_6, %Op.14 [template]
// CHECK:STDOUT:   %int_-6.2: Core.IntLiteral = int_value -6 [template]
// CHECK:STDOUT:   %Convert.bound.12: <bound method> = bound_method %int_-6.2, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.12: <specific function> = specific_function %Convert.bound.12, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %RuntimeCall.type: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %RuntimeCall: %RuntimeCall.type = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.2: %Expect.2 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.3: %Expect.3 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.4: %Expect.4 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.5: %Expect.5 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.6: %Expect.6 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.7: %Expect.7 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.5
// CHECK:STDOUT:     .Negate = %import_ref.229
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Expect = %Expect.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:     .LeftShift = %LeftShift.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Expect.decl: %Expect.type = class_decl @Expect [template = constants.%Expect.generic] {
// CHECK:STDOUT:     %N.patt.loc4_14.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_14.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc4_14.1, runtime_param<invalid> [symbolic = %N.patt.loc4_14.2 (constants.%N.patt.2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc4: type = splice_block %i32 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_14.1: %i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc4_14.2 (constants.%N.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [template = constants.%Test] {
// CHECK:STDOUT:     %N.patt.loc5_9.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc5_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc5_9.1, runtime_param<invalid> [symbolic = %N.patt.loc5_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %return.patt: @Test.%Expect.loc5_29.2 (%Expect.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Test.%Expect.loc5_29.2 (%Expect.1) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Expect.ref: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:     %N.ref: %i32 = name_ref N, %N.loc5_9.1 [symbolic = %N.loc5_9.2 (constants.%N.2)]
// CHECK:STDOUT:     %Expect.loc5_29.1: type = class_type @Expect, @Expect(constants.%N.2) [symbolic = %Expect.loc5_29.2 (constants.%Expect.1)]
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc5_13: type = splice_block %i32 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc5_9.1: %i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc5_9.2 (constants.%N.2)]
// CHECK:STDOUT:     %return.param: ref @Test.%Expect.loc5_29.2 (%Expect.1) = out_param runtime_param0
// CHECK:STDOUT:     %return: ref @Test.%Expect.loc5_29.2 (%Expect.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %LeftShift.decl: %LeftShift.type.1 = fn_decl @LeftShift.1 [template = constants.%LeftShift] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc7_33: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc7_33: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc7_17: type = splice_block %i32.loc7_17 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc7_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc7_17: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc7_25: type = splice_block %i32.loc7_25 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc7_25: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc7_25: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc20_35: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc20_35: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc20_19: type = splice_block %i32.loc20_19 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc20_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc20_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc20_27: type = splice_block %i32.loc20_27 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc20_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc20_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Expect(%N.loc4_14.1: %i32) {
// CHECK:STDOUT:   %N.loc4_14.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc4_14.2 (constants.%N.2)]
// CHECK:STDOUT:   %N.patt.loc4_14.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_14.2 (constants.%N.patt.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Expect.1
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%N.loc5_9.1: %i32) {
// CHECK:STDOUT:   %N.loc5_9.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc5_9.2 (constants.%N.2)]
// CHECK:STDOUT:   %N.patt.loc5_9.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc5_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:   %Expect.loc5_29.2: type = class_type @Expect, @Expect(%N.loc5_9.2) [symbolic = %Expect.loc5_29.2 (constants.%Expect.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Test.%Expect.loc5_29.2 (%Expect.1) [symbolic = %require_complete (constants.%require_complete.2)]
// CHECK:STDOUT:   %Expect.val: @Test.%Expect.loc5_29.2 (%Expect.1) = struct_value () [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: %i32) -> %return.param_patt: @Test.%Expect.loc5_29.2 (%Expect.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc5_41.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_41.2: init @Test.%Expect.loc5_29.2 (%Expect.1) = class_init (), %return [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:     %.loc5_42: init @Test.%Expect.loc5_29.2 (%Expect.1) = converted %.loc5_41.1, %.loc5_41.2 [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:     return %.loc5_42 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LeftShift.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.left_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Test.ref.loc10: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc10: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_0.loc10_18: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %int_0.loc10_21: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc10_18: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc10_18: <bound method> = bound_method %int_0.loc10_18, %impl.elem0.loc10_18 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_18: <specific function> = specific_function %Convert.bound.loc10_18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc10_18: init %i32 = call %Convert.specific_fn.loc10_18(%int_0.loc10_18) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_18.1: %i32 = value_of_initializer %int.convert_checked.loc10_18 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_18.2: %i32 = converted %int_0.loc10_18, %.loc10_18.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %impl.elem0.loc10_21: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc10_21: <bound method> = bound_method %int_0.loc10_21, %impl.elem0.loc10_21 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_21: <specific function> = specific_function %Convert.bound.loc10_21, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc10_21: init %i32 = call %Convert.specific_fn.loc10_21(%int_0.loc10_21) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_21.1: %i32 = value_of_initializer %int.convert_checked.loc10_21 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_21.2: %i32 = converted %int_0.loc10_21, %.loc10_21.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %int.left_shift.loc10: init %i32 = call %LeftShift.ref.loc10(%.loc10_18.2, %.loc10_21.2) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_23.1: %i32 = value_of_initializer %int.left_shift.loc10 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_23.2: %i32 = converted %int.left_shift.loc10, %.loc10_23.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Test.specific_fn.loc10: <specific function> = specific_function %Test.ref.loc10, @Test(constants.%int_0.2) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc10_23.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc10: init %Expect.2 = call %Test.specific_fn.loc10() to %.loc10_23.3
// CHECK:STDOUT:   %Expect.ref.loc10: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc10_35: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc10_36: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc10_36: <bound method> = bound_method %int_0.loc10_35, %impl.elem0.loc10_36 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_36: <specific function> = specific_function %Convert.bound.loc10_36, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc10_36: init %i32 = call %Convert.specific_fn.loc10_36(%int_0.loc10_35) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_36.1: %i32 = value_of_initializer %int.convert_checked.loc10_36 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_36.2: %i32 = converted %int_0.loc10_35, %.loc10_36.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Expect.loc10: type = class_type @Expect, @Expect(constants.%int_0.2) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc10_23.4: ref %Expect.2 = temporary %.loc10_23.3, %Test.call.loc10
// CHECK:STDOUT:   %Test.ref.loc11: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc11: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_0.loc11_18: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc11_18: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_18: <bound method> = bound_method %int_0.loc11_18, %impl.elem0.loc11_18 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_18: <specific function> = specific_function %Convert.bound.loc11_18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_18: init %i32 = call %Convert.specific_fn.loc11_18(%int_0.loc11_18) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_18.1: %i32 = value_of_initializer %int.convert_checked.loc11_18 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_18.2: %i32 = converted %int_0.loc11_18, %.loc11_18.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %impl.elem0.loc11_21: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_21: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_21 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_21: <specific function> = specific_function %Convert.bound.loc11_21, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc11_21: init %i32 = call %Convert.specific_fn.loc11_21(%int_1.loc11) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc11_21.1: %i32 = value_of_initializer %int.convert_checked.loc11_21 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc11_21.2: %i32 = converted %int_1.loc11, %.loc11_21.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.left_shift.loc11: init %i32 = call %LeftShift.ref.loc11(%.loc11_18.2, %.loc11_21.2) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_23.1: %i32 = value_of_initializer %int.left_shift.loc11 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_23.2: %i32 = converted %int.left_shift.loc11, %.loc11_23.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Test.specific_fn.loc11: <specific function> = specific_function %Test.ref.loc11, @Test(constants.%int_0.2) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc11_23.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc11: init %Expect.2 = call %Test.specific_fn.loc11() to %.loc11_23.3
// CHECK:STDOUT:   %Expect.ref.loc11: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc11_35: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc11_36: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_36: <bound method> = bound_method %int_0.loc11_35, %impl.elem0.loc11_36 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_36: <specific function> = specific_function %Convert.bound.loc11_36, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_36: init %i32 = call %Convert.specific_fn.loc11_36(%int_0.loc11_35) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_36.1: %i32 = value_of_initializer %int.convert_checked.loc11_36 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_36.2: %i32 = converted %int_0.loc11_35, %.loc11_36.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Expect.loc11: type = class_type @Expect, @Expect(constants.%int_0.2) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc11_23.4: ref %Expect.2 = temporary %.loc11_23.3, %Test.call.loc11
// CHECK:STDOUT:   %Test.ref.loc12: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc12: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_0.loc12_18: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %int_30.loc12: Core.IntLiteral = int_value 30 [template = constants.%int_30.1]
// CHECK:STDOUT:   %impl.elem0.loc12_18: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_18: <bound method> = bound_method %int_0.loc12_18, %impl.elem0.loc12_18 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_18: <specific function> = specific_function %Convert.bound.loc12_18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc12_18: init %i32 = call %Convert.specific_fn.loc12_18(%int_0.loc12_18) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc12_18.1: %i32 = value_of_initializer %int.convert_checked.loc12_18 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc12_18.2: %i32 = converted %int_0.loc12_18, %.loc12_18.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %impl.elem0.loc12_21: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_21: <bound method> = bound_method %int_30.loc12, %impl.elem0.loc12_21 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_21: <specific function> = specific_function %Convert.bound.loc12_21, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc12_21: init %i32 = call %Convert.specific_fn.loc12_21(%int_30.loc12) [template = constants.%int_30.2]
// CHECK:STDOUT:   %.loc12_21.1: %i32 = value_of_initializer %int.convert_checked.loc12_21 [template = constants.%int_30.2]
// CHECK:STDOUT:   %.loc12_21.2: %i32 = converted %int_30.loc12, %.loc12_21.1 [template = constants.%int_30.2]
// CHECK:STDOUT:   %int.left_shift.loc12: init %i32 = call %LeftShift.ref.loc12(%.loc12_18.2, %.loc12_21.2) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc12_24.1: %i32 = value_of_initializer %int.left_shift.loc12 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc12_24.2: %i32 = converted %int.left_shift.loc12, %.loc12_24.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Test.specific_fn.loc12: <specific function> = specific_function %Test.ref.loc12, @Test(constants.%int_0.2) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc12_24.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc12: init %Expect.2 = call %Test.specific_fn.loc12() to %.loc12_24.3
// CHECK:STDOUT:   %Expect.ref.loc12: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc12_36: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc12_37: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_37: <bound method> = bound_method %int_0.loc12_36, %impl.elem0.loc12_37 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_37: <specific function> = specific_function %Convert.bound.loc12_37, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc12_37: init %i32 = call %Convert.specific_fn.loc12_37(%int_0.loc12_36) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc12_37.1: %i32 = value_of_initializer %int.convert_checked.loc12_37 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc12_37.2: %i32 = converted %int_0.loc12_36, %.loc12_37.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Expect.loc12: type = class_type @Expect, @Expect(constants.%int_0.2) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc12_24.4: ref %Expect.2 = temporary %.loc12_24.3, %Test.call.loc12
// CHECK:STDOUT:   %Test.ref.loc13: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc13: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_30.loc13: Core.IntLiteral = int_value 30 [template = constants.%int_30.1]
// CHECK:STDOUT:   %impl.elem0.loc13_18: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc13_18: <bound method> = bound_method %int_1.loc13, %impl.elem0.loc13_18 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_18: <specific function> = specific_function %Convert.bound.loc13_18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc13_18: init %i32 = call %Convert.specific_fn.loc13_18(%int_1.loc13) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_18.1: %i32 = value_of_initializer %int.convert_checked.loc13_18 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_18.2: %i32 = converted %int_1.loc13, %.loc13_18.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc13_21: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc13_21: <bound method> = bound_method %int_30.loc13, %impl.elem0.loc13_21 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_21: <specific function> = specific_function %Convert.bound.loc13_21, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc13_21: init %i32 = call %Convert.specific_fn.loc13_21(%int_30.loc13) [template = constants.%int_30.2]
// CHECK:STDOUT:   %.loc13_21.1: %i32 = value_of_initializer %int.convert_checked.loc13_21 [template = constants.%int_30.2]
// CHECK:STDOUT:   %.loc13_21.2: %i32 = converted %int_30.loc13, %.loc13_21.1 [template = constants.%int_30.2]
// CHECK:STDOUT:   %int.left_shift.loc13: init %i32 = call %LeftShift.ref.loc13(%.loc13_18.2, %.loc13_21.2) [template = constants.%int_1073741824.1]
// CHECK:STDOUT:   %.loc13_24.1: %i32 = value_of_initializer %int.left_shift.loc13 [template = constants.%int_1073741824.1]
// CHECK:STDOUT:   %.loc13_24.2: %i32 = converted %int.left_shift.loc13, %.loc13_24.1 [template = constants.%int_1073741824.1]
// CHECK:STDOUT:   %Test.specific_fn.loc13: <specific function> = specific_function %Test.ref.loc13, @Test(constants.%int_1073741824.1) [template = constants.%Test.specific_fn.2]
// CHECK:STDOUT:   %.loc13_24.3: ref %Expect.3 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc13: init %Expect.3 = call %Test.specific_fn.loc13() to %.loc13_24.3
// CHECK:STDOUT:   %Expect.ref.loc13: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_1073741824: Core.IntLiteral = int_value 1073741824 [template = constants.%int_1073741824.2]
// CHECK:STDOUT:   %impl.elem0.loc13_47: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc13_47: <bound method> = bound_method %int_1073741824, %impl.elem0.loc13_47 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_47: <specific function> = specific_function %Convert.bound.loc13_47, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc13_47: init %i32 = call %Convert.specific_fn.loc13_47(%int_1073741824) [template = constants.%int_1073741824.1]
// CHECK:STDOUT:   %.loc13_47.1: %i32 = value_of_initializer %int.convert_checked.loc13_47 [template = constants.%int_1073741824.1]
// CHECK:STDOUT:   %.loc13_47.2: %i32 = converted %int_1073741824, %.loc13_47.1 [template = constants.%int_1073741824.1]
// CHECK:STDOUT:   %Expect.loc13: type = class_type @Expect, @Expect(constants.%int_1073741824.1) [template = constants.%Expect.3]
// CHECK:STDOUT:   %.loc13_24.4: ref %Expect.3 = temporary %.loc13_24.3, %Test.call.loc13
// CHECK:STDOUT:   %Test.ref.loc14: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc14: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:   %int_2.loc14: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %impl.elem0.loc14_18: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc14_18: <bound method> = bound_method %int_5, %impl.elem0.loc14_18 [template = constants.%Convert.bound.5]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_18: <specific function> = specific_function %Convert.bound.loc14_18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %int.convert_checked.loc14_18: init %i32 = call %Convert.specific_fn.loc14_18(%int_5) [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc14_18.1: %i32 = value_of_initializer %int.convert_checked.loc14_18 [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc14_18.2: %i32 = converted %int_5, %.loc14_18.1 [template = constants.%int_5.2]
// CHECK:STDOUT:   %impl.elem0.loc14_21: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc14_21: <bound method> = bound_method %int_2.loc14, %impl.elem0.loc14_21 [template = constants.%Convert.bound.6]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_21: <specific function> = specific_function %Convert.bound.loc14_21, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.6]
// CHECK:STDOUT:   %int.convert_checked.loc14_21: init %i32 = call %Convert.specific_fn.loc14_21(%int_2.loc14) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc14_21.1: %i32 = value_of_initializer %int.convert_checked.loc14_21 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc14_21.2: %i32 = converted %int_2.loc14, %.loc14_21.1 [template = constants.%int_2.2]
// CHECK:STDOUT:   %int.left_shift.loc14: init %i32 = call %LeftShift.ref.loc14(%.loc14_18.2, %.loc14_21.2) [template = constants.%int_20.1]
// CHECK:STDOUT:   %.loc14_23.1: %i32 = value_of_initializer %int.left_shift.loc14 [template = constants.%int_20.1]
// CHECK:STDOUT:   %.loc14_23.2: %i32 = converted %int.left_shift.loc14, %.loc14_23.1 [template = constants.%int_20.1]
// CHECK:STDOUT:   %Test.specific_fn.loc14: <specific function> = specific_function %Test.ref.loc14, @Test(constants.%int_20.1) [template = constants.%Test.specific_fn.3]
// CHECK:STDOUT:   %.loc14_23.3: ref %Expect.4 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc14: init %Expect.4 = call %Test.specific_fn.loc14() to %.loc14_23.3
// CHECK:STDOUT:   %Expect.ref.loc14: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [template = constants.%int_20.2]
// CHECK:STDOUT:   %impl.elem0.loc14_37: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc14_37: <bound method> = bound_method %int_20, %impl.elem0.loc14_37 [template = constants.%Convert.bound.7]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_37: <specific function> = specific_function %Convert.bound.loc14_37, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.7]
// CHECK:STDOUT:   %int.convert_checked.loc14_37: init %i32 = call %Convert.specific_fn.loc14_37(%int_20) [template = constants.%int_20.1]
// CHECK:STDOUT:   %.loc14_37.1: %i32 = value_of_initializer %int.convert_checked.loc14_37 [template = constants.%int_20.1]
// CHECK:STDOUT:   %.loc14_37.2: %i32 = converted %int_20, %.loc14_37.1 [template = constants.%int_20.1]
// CHECK:STDOUT:   %Expect.loc14: type = class_type @Expect, @Expect(constants.%int_20.1) [template = constants.%Expect.4]
// CHECK:STDOUT:   %.loc14_23.4: ref %Expect.4 = temporary %.loc14_23.3, %Test.call.loc14
// CHECK:STDOUT:   %Test.ref.loc15: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc15: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc15_19: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc15_18.1: %Op.type.13 = interface_witness_access constants.%interface.20, element0 [template = constants.%Op.14]
// CHECK:STDOUT:   %Op.bound.loc15_18: <bound method> = bound_method %int_1.loc15_19, %impl.elem0.loc15_18.1 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc15_18: init Core.IntLiteral = call %Op.bound.loc15_18(%int_1.loc15_19) [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int_1.loc15_22: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc15_18.2: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc15_18: <bound method> = bound_method %int.snegate.loc15_18, %impl.elem0.loc15_18.2 [template = constants.%Convert.bound.8]
// CHECK:STDOUT:   %Convert.specific_fn.loc15_18: <specific function> = specific_function %Convert.bound.loc15_18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.8]
// CHECK:STDOUT:   %.loc15_18.1: Core.IntLiteral = value_of_initializer %int.snegate.loc15_18 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc15_18.2: Core.IntLiteral = converted %int.snegate.loc15_18, %.loc15_18.1 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int.convert_checked.loc15_18: init %i32 = call %Convert.specific_fn.loc15_18(%.loc15_18.2) [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc15_18.3: %i32 = value_of_initializer %int.convert_checked.loc15_18 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc15_18.4: %i32 = converted %int.snegate.loc15_18, %.loc15_18.3 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %impl.elem0.loc15_22: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc15_22: <bound method> = bound_method %int_1.loc15_22, %impl.elem0.loc15_22 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc15_22: <specific function> = specific_function %Convert.bound.loc15_22, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc15_22: init %i32 = call %Convert.specific_fn.loc15_22(%int_1.loc15_22) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc15_22.1: %i32 = value_of_initializer %int.convert_checked.loc15_22 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc15_22.2: %i32 = converted %int_1.loc15_22, %.loc15_22.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.left_shift.loc15: init %i32 = call %LeftShift.ref.loc15(%.loc15_18.4, %.loc15_22.2) [template = constants.%int_-2.1]
// CHECK:STDOUT:   %.loc15_24.1: %i32 = value_of_initializer %int.left_shift.loc15 [template = constants.%int_-2.1]
// CHECK:STDOUT:   %.loc15_24.2: %i32 = converted %int.left_shift.loc15, %.loc15_24.1 [template = constants.%int_-2.1]
// CHECK:STDOUT:   %Test.specific_fn.loc15: <specific function> = specific_function %Test.ref.loc15, @Test(constants.%int_-2.1) [template = constants.%Test.specific_fn.4]
// CHECK:STDOUT:   %.loc15_24.3: ref %Expect.5 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc15: init %Expect.5 = call %Test.specific_fn.loc15() to %.loc15_24.3
// CHECK:STDOUT:   %Expect.ref.loc15: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_2.loc15: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %impl.elem0.loc15_36: %Op.type.13 = interface_witness_access constants.%interface.20, element0 [template = constants.%Op.14]
// CHECK:STDOUT:   %Op.bound.loc15_36: <bound method> = bound_method %int_2.loc15, %impl.elem0.loc15_36 [template = constants.%Op.bound.2]
// CHECK:STDOUT:   %int.snegate.loc15_36: init Core.IntLiteral = call %Op.bound.loc15_36(%int_2.loc15) [template = constants.%int_-2.2]
// CHECK:STDOUT:   %impl.elem0.loc15_38: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc15_38: <bound method> = bound_method %int.snegate.loc15_36, %impl.elem0.loc15_38 [template = constants.%Convert.bound.9]
// CHECK:STDOUT:   %Convert.specific_fn.loc15_38: <specific function> = specific_function %Convert.bound.loc15_38, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.9]
// CHECK:STDOUT:   %.loc15_36.1: Core.IntLiteral = value_of_initializer %int.snegate.loc15_36 [template = constants.%int_-2.2]
// CHECK:STDOUT:   %.loc15_36.2: Core.IntLiteral = converted %int.snegate.loc15_36, %.loc15_36.1 [template = constants.%int_-2.2]
// CHECK:STDOUT:   %int.convert_checked.loc15_38: init %i32 = call %Convert.specific_fn.loc15_38(%.loc15_36.2) [template = constants.%int_-2.1]
// CHECK:STDOUT:   %.loc15_38.1: %i32 = value_of_initializer %int.convert_checked.loc15_38 [template = constants.%int_-2.1]
// CHECK:STDOUT:   %.loc15_38.2: %i32 = converted %int.snegate.loc15_36, %.loc15_38.1 [template = constants.%int_-2.1]
// CHECK:STDOUT:   %Expect.loc15: type = class_type @Expect, @Expect(constants.%int_-2.1) [template = constants.%Expect.5]
// CHECK:STDOUT:   %.loc15_24.4: ref %Expect.5 = temporary %.loc15_24.3, %Test.call.loc15
// CHECK:STDOUT:   %Test.ref.loc16: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc16: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_2.loc16: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %impl.elem0.loc16_18.1: %Op.type.13 = interface_witness_access constants.%interface.20, element0 [template = constants.%Op.14]
// CHECK:STDOUT:   %Op.bound.loc16_18: <bound method> = bound_method %int_2.loc16, %impl.elem0.loc16_18.1 [template = constants.%Op.bound.2]
// CHECK:STDOUT:   %int.snegate.loc16_18: init Core.IntLiteral = call %Op.bound.loc16_18(%int_2.loc16) [template = constants.%int_-2.2]
// CHECK:STDOUT:   %int_1.loc16: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc16_18.2: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc16_18: <bound method> = bound_method %int.snegate.loc16_18, %impl.elem0.loc16_18.2 [template = constants.%Convert.bound.9]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_18: <specific function> = specific_function %Convert.bound.loc16_18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.9]
// CHECK:STDOUT:   %.loc16_18.1: Core.IntLiteral = value_of_initializer %int.snegate.loc16_18 [template = constants.%int_-2.2]
// CHECK:STDOUT:   %.loc16_18.2: Core.IntLiteral = converted %int.snegate.loc16_18, %.loc16_18.1 [template = constants.%int_-2.2]
// CHECK:STDOUT:   %int.convert_checked.loc16_18: init %i32 = call %Convert.specific_fn.loc16_18(%.loc16_18.2) [template = constants.%int_-2.1]
// CHECK:STDOUT:   %.loc16_18.3: %i32 = value_of_initializer %int.convert_checked.loc16_18 [template = constants.%int_-2.1]
// CHECK:STDOUT:   %.loc16_18.4: %i32 = converted %int.snegate.loc16_18, %.loc16_18.3 [template = constants.%int_-2.1]
// CHECK:STDOUT:   %impl.elem0.loc16_22: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc16_22: <bound method> = bound_method %int_1.loc16, %impl.elem0.loc16_22 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_22: <specific function> = specific_function %Convert.bound.loc16_22, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc16_22: init %i32 = call %Convert.specific_fn.loc16_22(%int_1.loc16) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc16_22.1: %i32 = value_of_initializer %int.convert_checked.loc16_22 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc16_22.2: %i32 = converted %int_1.loc16, %.loc16_22.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.left_shift.loc16: init %i32 = call %LeftShift.ref.loc16(%.loc16_18.4, %.loc16_22.2) [template = constants.%int_-4.1]
// CHECK:STDOUT:   %.loc16_24.1: %i32 = value_of_initializer %int.left_shift.loc16 [template = constants.%int_-4.1]
// CHECK:STDOUT:   %.loc16_24.2: %i32 = converted %int.left_shift.loc16, %.loc16_24.1 [template = constants.%int_-4.1]
// CHECK:STDOUT:   %Test.specific_fn.loc16: <specific function> = specific_function %Test.ref.loc16, @Test(constants.%int_-4.1) [template = constants.%Test.specific_fn.5]
// CHECK:STDOUT:   %.loc16_24.3: ref %Expect.6 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc16: init %Expect.6 = call %Test.specific_fn.loc16() to %.loc16_24.3
// CHECK:STDOUT:   %Expect.ref.loc16: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template = constants.%int_4]
// CHECK:STDOUT:   %impl.elem0.loc16_36: %Op.type.13 = interface_witness_access constants.%interface.20, element0 [template = constants.%Op.14]
// CHECK:STDOUT:   %Op.bound.loc16_36: <bound method> = bound_method %int_4, %impl.elem0.loc16_36 [template = constants.%Op.bound.3]
// CHECK:STDOUT:   %int.snegate.loc16_36: init Core.IntLiteral = call %Op.bound.loc16_36(%int_4) [template = constants.%int_-4.2]
// CHECK:STDOUT:   %impl.elem0.loc16_38: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc16_38: <bound method> = bound_method %int.snegate.loc16_36, %impl.elem0.loc16_38 [template = constants.%Convert.bound.10]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_38: <specific function> = specific_function %Convert.bound.loc16_38, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.10]
// CHECK:STDOUT:   %.loc16_36.1: Core.IntLiteral = value_of_initializer %int.snegate.loc16_36 [template = constants.%int_-4.2]
// CHECK:STDOUT:   %.loc16_36.2: Core.IntLiteral = converted %int.snegate.loc16_36, %.loc16_36.1 [template = constants.%int_-4.2]
// CHECK:STDOUT:   %int.convert_checked.loc16_38: init %i32 = call %Convert.specific_fn.loc16_38(%.loc16_36.2) [template = constants.%int_-4.1]
// CHECK:STDOUT:   %.loc16_38.1: %i32 = value_of_initializer %int.convert_checked.loc16_38 [template = constants.%int_-4.1]
// CHECK:STDOUT:   %.loc16_38.2: %i32 = converted %int.snegate.loc16_36, %.loc16_38.1 [template = constants.%int_-4.1]
// CHECK:STDOUT:   %Expect.loc16: type = class_type @Expect, @Expect(constants.%int_-4.1) [template = constants.%Expect.6]
// CHECK:STDOUT:   %.loc16_24.4: ref %Expect.6 = temporary %.loc16_24.3, %Test.call.loc16
// CHECK:STDOUT:   %Test.ref.loc17: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc17: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %impl.elem0.loc17_18.1: %Op.type.13 = interface_witness_access constants.%interface.20, element0 [template = constants.%Op.14]
// CHECK:STDOUT:   %Op.bound.loc17_18: <bound method> = bound_method %int_3, %impl.elem0.loc17_18.1 [template = constants.%Op.bound.4]
// CHECK:STDOUT:   %int.snegate.loc17_18: init Core.IntLiteral = call %Op.bound.loc17_18(%int_3) [template = constants.%int_-3.1]
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc17_18.2: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc17_18: <bound method> = bound_method %int.snegate.loc17_18, %impl.elem0.loc17_18.2 [template = constants.%Convert.bound.11]
// CHECK:STDOUT:   %Convert.specific_fn.loc17_18: <specific function> = specific_function %Convert.bound.loc17_18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.11]
// CHECK:STDOUT:   %.loc17_18.1: Core.IntLiteral = value_of_initializer %int.snegate.loc17_18 [template = constants.%int_-3.1]
// CHECK:STDOUT:   %.loc17_18.2: Core.IntLiteral = converted %int.snegate.loc17_18, %.loc17_18.1 [template = constants.%int_-3.1]
// CHECK:STDOUT:   %int.convert_checked.loc17_18: init %i32 = call %Convert.specific_fn.loc17_18(%.loc17_18.2) [template = constants.%int_-3.2]
// CHECK:STDOUT:   %.loc17_18.3: %i32 = value_of_initializer %int.convert_checked.loc17_18 [template = constants.%int_-3.2]
// CHECK:STDOUT:   %.loc17_18.4: %i32 = converted %int.snegate.loc17_18, %.loc17_18.3 [template = constants.%int_-3.2]
// CHECK:STDOUT:   %impl.elem0.loc17_22: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc17_22: <bound method> = bound_method %int_1.loc17, %impl.elem0.loc17_22 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc17_22: <specific function> = specific_function %Convert.bound.loc17_22, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc17_22: init %i32 = call %Convert.specific_fn.loc17_22(%int_1.loc17) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc17_22.1: %i32 = value_of_initializer %int.convert_checked.loc17_22 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc17_22.2: %i32 = converted %int_1.loc17, %.loc17_22.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.left_shift.loc17: init %i32 = call %LeftShift.ref.loc17(%.loc17_18.4, %.loc17_22.2) [template = constants.%int_-6.1]
// CHECK:STDOUT:   %.loc17_24.1: %i32 = value_of_initializer %int.left_shift.loc17 [template = constants.%int_-6.1]
// CHECK:STDOUT:   %.loc17_24.2: %i32 = converted %int.left_shift.loc17, %.loc17_24.1 [template = constants.%int_-6.1]
// CHECK:STDOUT:   %Test.specific_fn.loc17: <specific function> = specific_function %Test.ref.loc17, @Test(constants.%int_-6.1) [template = constants.%Test.specific_fn.6]
// CHECK:STDOUT:   %.loc17_24.3: ref %Expect.7 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc17: init %Expect.7 = call %Test.specific_fn.loc17() to %.loc17_24.3
// CHECK:STDOUT:   %Expect.ref.loc17: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [template = constants.%int_6]
// CHECK:STDOUT:   %impl.elem0.loc17_36: %Op.type.13 = interface_witness_access constants.%interface.20, element0 [template = constants.%Op.14]
// CHECK:STDOUT:   %Op.bound.loc17_36: <bound method> = bound_method %int_6, %impl.elem0.loc17_36 [template = constants.%Op.bound.5]
// CHECK:STDOUT:   %int.snegate.loc17_36: init Core.IntLiteral = call %Op.bound.loc17_36(%int_6) [template = constants.%int_-6.2]
// CHECK:STDOUT:   %impl.elem0.loc17_38: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc17_38: <bound method> = bound_method %int.snegate.loc17_36, %impl.elem0.loc17_38 [template = constants.%Convert.bound.12]
// CHECK:STDOUT:   %Convert.specific_fn.loc17_38: <specific function> = specific_function %Convert.bound.loc17_38, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.12]
// CHECK:STDOUT:   %.loc17_36.1: Core.IntLiteral = value_of_initializer %int.snegate.loc17_36 [template = constants.%int_-6.2]
// CHECK:STDOUT:   %.loc17_36.2: Core.IntLiteral = converted %int.snegate.loc17_36, %.loc17_36.1 [template = constants.%int_-6.2]
// CHECK:STDOUT:   %int.convert_checked.loc17_38: init %i32 = call %Convert.specific_fn.loc17_38(%.loc17_36.2) [template = constants.%int_-6.1]
// CHECK:STDOUT:   %.loc17_38.1: %i32 = value_of_initializer %int.convert_checked.loc17_38 [template = constants.%int_-6.1]
// CHECK:STDOUT:   %.loc17_38.2: %i32 = converted %int.snegate.loc17_36, %.loc17_38.1 [template = constants.%int_-6.1]
// CHECK:STDOUT:   %Expect.loc17: type = class_type @Expect, @Expect(constants.%int_-6.1) [template = constants.%Expect.7]
// CHECK:STDOUT:   %.loc17_24.4: ref %Expect.7 = temporary %.loc17_24.3, %Test.call.loc17
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %LeftShift.ref: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.left_shift: init %i32 = call %LeftShift.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc21_25.1: %i32 = value_of_initializer %int.left_shift
// CHECK:STDOUT:   %.loc21_25.2: %i32 = converted %int.left_shift, %.loc21_25.1
// CHECK:STDOUT:   return %.loc21_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%N.2) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%N.2
// CHECK:STDOUT:   %N.patt.loc4_14.2 => constants.%N.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(@Test.%N.loc5_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%N.2) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%N.2
// CHECK:STDOUT:   %N.patt.loc5_9.2 => constants.%N.2
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_0.2) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_0.2
// CHECK:STDOUT:   %N.patt.loc5_9.2 => constants.%int_0.2
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.2
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_0.2) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_0.2
// CHECK:STDOUT:   %N.patt.loc4_14.2 => constants.%int_0.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_1073741824.1) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_1073741824.1
// CHECK:STDOUT:   %N.patt.loc5_9.2 => constants.%int_1073741824.1
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.2
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_1073741824.1) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_1073741824.1
// CHECK:STDOUT:   %N.patt.loc4_14.2 => constants.%int_1073741824.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_20.1) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_20.1
// CHECK:STDOUT:   %N.patt.loc5_9.2 => constants.%int_20.1
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.2
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_20.1) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_20.1
// CHECK:STDOUT:   %N.patt.loc4_14.2 => constants.%int_20.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_-2.1) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_-2.1
// CHECK:STDOUT:   %N.patt.loc5_9.2 => constants.%int_-2.1
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.2
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-2.1) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_-2.1
// CHECK:STDOUT:   %N.patt.loc4_14.2 => constants.%int_-2.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_-4.1) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_-4.1
// CHECK:STDOUT:   %N.patt.loc5_9.2 => constants.%int_-4.1
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.2
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-4.1) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_-4.1
// CHECK:STDOUT:   %N.patt.loc4_14.2 => constants.%int_-4.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_-6.1) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_-6.1
// CHECK:STDOUT:   %N.patt.loc5_9.2 => constants.%int_-6.1
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.2
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-6.1) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_-6.1
// CHECK:STDOUT:   %N.patt.loc4_14.2 => constants.%int_-6.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- u32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [template]
// CHECK:STDOUT:   %N.2: %u32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.2: %u32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Expect.type: type = generic_class_type @Expect [template]
// CHECK:STDOUT:   %Expect.generic: %Expect.type = struct_value () [template]
// CHECK:STDOUT:   %Expect.1: type = class_type @Expect, @Expect(%N.2) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [template]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [template]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %Expect.1 [symbolic]
// CHECK:STDOUT:   %Expect.val.1: %Expect.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %LeftShift.type.1: type = fn_type @LeftShift.1 [template]
// CHECK:STDOUT:   %LeftShift: %LeftShift.type.1 = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%u32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.19: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_0.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.2: %u32 = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.16: type = fn_type @Convert.7, @impl.43(%int_32) [template]
// CHECK:STDOUT:   %Convert.16: %Convert.type.16 = struct_value () [template]
// CHECK:STDOUT:   %interface.38: <witness> = interface_witness (%Convert.16) [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_0.1, %Convert.16 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.7(%int_32) [template]
// CHECK:STDOUT:   %int_0.3: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %Expect.2: type = class_type @Expect, @Expect(%int_0.2) [template]
// CHECK:STDOUT:   %Test.specific_fn.1: <specific function> = specific_function %Test, @Test(%int_0.2) [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_1.1, %Convert.16 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.7(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int_30.1: Core.IntLiteral = int_value 30 [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_30.1, %Convert.16 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.7(%int_32) [template]
// CHECK:STDOUT:   %int_30.2: %i32 = int_value 30 [template]
// CHECK:STDOUT:   %Convert.bound.5: <bound method> = bound_method %int_1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.5: <specific function> = specific_function %Convert.bound.5, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.3: %u32 = int_value 1 [template]
// CHECK:STDOUT:   %int_1073741824.1: %u32 = int_value 1073741824 [template]
// CHECK:STDOUT:   %Expect.3: type = class_type @Expect, @Expect(%int_1073741824.1) [template]
// CHECK:STDOUT:   %Test.specific_fn.2: <specific function> = specific_function %Test, @Test(%int_1073741824.1) [template]
// CHECK:STDOUT:   %int_1073741824.2: Core.IntLiteral = int_value 1073741824 [template]
// CHECK:STDOUT:   %Convert.bound.6: <bound method> = bound_method %int_1073741824.2, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.6: <specific function> = specific_function %Convert.bound.6, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_5.1: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.7: <bound method> = bound_method %int_5.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.7: <specific function> = specific_function %Convert.bound.7, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_5.2: %u32 = int_value 5 [template]
// CHECK:STDOUT:   %Convert.bound.8: <bound method> = bound_method %int_2.1, %Convert.16 [template]
// CHECK:STDOUT:   %Convert.specific_fn.8: <specific function> = specific_function %Convert.bound.8, @Convert.7(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %int_20.1: %u32 = int_value 20 [template]
// CHECK:STDOUT:   %Expect.4: type = class_type @Expect, @Expect(%int_20.1) [template]
// CHECK:STDOUT:   %Test.specific_fn.3: <specific function> = specific_function %Test, @Test(%int_20.1) [template]
// CHECK:STDOUT:   %int_20.2: Core.IntLiteral = int_value 20 [template]
// CHECK:STDOUT:   %Convert.bound.9: <bound method> = bound_method %int_20.2, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.9: <specific function> = specific_function %Convert.bound.9, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_4294967295.1: Core.IntLiteral = int_value 4294967295 [template]
// CHECK:STDOUT:   %Convert.bound.10: <bound method> = bound_method %int_4294967295.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.10: <specific function> = specific_function %Convert.bound.10, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_4294967295.2: %u32 = int_value 4294967295 [template]
// CHECK:STDOUT:   %int_4294967294.1: %u32 = int_value 4294967294 [template]
// CHECK:STDOUT:   %Expect.5: type = class_type @Expect, @Expect(%int_4294967294.1) [template]
// CHECK:STDOUT:   %Test.specific_fn.4: <specific function> = specific_function %Test, @Test(%int_4294967294.1) [template]
// CHECK:STDOUT:   %int_4294967294.2: Core.IntLiteral = int_value 4294967294 [template]
// CHECK:STDOUT:   %Convert.bound.11: <bound method> = bound_method %int_4294967294.2, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.11: <specific function> = specific_function %Convert.bound.11, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_4294967292.1: %u32 = int_value 4294967292 [template]
// CHECK:STDOUT:   %Expect.6: type = class_type @Expect, @Expect(%int_4294967292.1) [template]
// CHECK:STDOUT:   %Test.specific_fn.5: <specific function> = specific_function %Test, @Test(%int_4294967292.1) [template]
// CHECK:STDOUT:   %int_4294967292.2: Core.IntLiteral = int_value 4294967292 [template]
// CHECK:STDOUT:   %Convert.bound.12: <bound method> = bound_method %int_4294967292.2, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.12: <specific function> = specific_function %Convert.bound.12, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2882400001.1: Core.IntLiteral = int_value 2882400001 [template]
// CHECK:STDOUT:   %int_8.1: Core.IntLiteral = int_value 8 [template]
// CHECK:STDOUT:   %Convert.bound.13: <bound method> = bound_method %int_2882400001.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.13: <specific function> = specific_function %Convert.bound.13, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2882400001.2: %u32 = int_value 2882400001 [template]
// CHECK:STDOUT:   %Convert.bound.14: <bound method> = bound_method %int_8.1, %Convert.16 [template]
// CHECK:STDOUT:   %Convert.specific_fn.14: <specific function> = specific_function %Convert.bound.14, @Convert.7(%int_32) [template]
// CHECK:STDOUT:   %int_8.2: %i32 = int_value 8 [template]
// CHECK:STDOUT:   %int_3454992640.1: %u32 = int_value 3454992640 [template]
// CHECK:STDOUT:   %Expect.7: type = class_type @Expect, @Expect(%int_3454992640.1) [template]
// CHECK:STDOUT:   %Test.specific_fn.6: <specific function> = specific_function %Test, @Test(%int_3454992640.1) [template]
// CHECK:STDOUT:   %int_3454992640.2: Core.IntLiteral = int_value 3454992640 [template]
// CHECK:STDOUT:   %Convert.bound.15: <bound method> = bound_method %int_3454992640.2, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.15: <specific function> = specific_function %Convert.bound.15, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %RuntimeCall.type: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %RuntimeCall: %RuntimeCall.type = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.2: %Expect.2 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.3: %Expect.3 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.4: %Expect.4 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.5: %Expect.5 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.6: %Expect.6 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.7: %Expect.7 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .UInt = %import_ref.1
// CHECK:STDOUT:     .Int = %import_ref.5
// CHECK:STDOUT:     .ImplicitAs = %import_ref.9
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Expect = %Expect.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:     .LeftShift = %LeftShift.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Expect.decl: %Expect.type = class_decl @Expect [template = constants.%Expect.generic] {
// CHECK:STDOUT:     %N.patt.loc4_14.1: %u32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_14.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %N.param_patt: %u32 = value_param_pattern %N.patt.loc4_14.1, runtime_param<invalid> [symbolic = %N.patt.loc4_14.2 (constants.%N.patt.2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: %u32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc4: type = splice_block %u32 [template = constants.%u32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %u32: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_14.1: %u32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc4_14.2 (constants.%N.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [template = constants.%Test] {
// CHECK:STDOUT:     %N.patt.loc5_9.1: %u32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc5_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %N.param_patt: %u32 = value_param_pattern %N.patt.loc5_9.1, runtime_param<invalid> [symbolic = %N.patt.loc5_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %return.patt: @Test.%Expect.loc5_29.2 (%Expect.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Test.%Expect.loc5_29.2 (%Expect.1) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Expect.ref: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:     %N.ref: %u32 = name_ref N, %N.loc5_9.1 [symbolic = %N.loc5_9.2 (constants.%N.2)]
// CHECK:STDOUT:     %Expect.loc5_29.1: type = class_type @Expect, @Expect(constants.%N.2) [symbolic = %Expect.loc5_29.2 (constants.%Expect.1)]
// CHECK:STDOUT:     %N.param: %u32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc5_13: type = splice_block %u32 [template = constants.%u32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %u32: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc5_9.1: %u32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc5_9.2 (constants.%N.2)]
// CHECK:STDOUT:     %return.param: ref @Test.%Expect.loc5_29.2 (%Expect.1) = out_param runtime_param0
// CHECK:STDOUT:     %return: ref @Test.%Expect.loc5_29.2 (%Expect.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %LeftShift.decl: %LeftShift.type.1 = fn_decl @LeftShift.1 [template = constants.%LeftShift] {
// CHECK:STDOUT:     %a.patt: %u32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %u32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %u32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %u32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc7_33: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %u32.loc7_33: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     %a.param: %u32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc7_17: type = splice_block %u32.loc7_17 [template = constants.%u32] {
// CHECK:STDOUT:       %int_32.loc7_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %u32.loc7_17: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %u32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc7_25: type = splice_block %i32 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc7_25: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %u32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %u32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: %u32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %u32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %u32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %u32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc20_35: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %u32.loc20_35: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     %a.param: %u32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc20_19: type = splice_block %u32.loc20_19 [template = constants.%u32] {
// CHECK:STDOUT:       %int_32.loc20_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %u32.loc20_19: type = class_type @UInt, @UInt(constants.%int_32) [template = constants.%u32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %u32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc20_27: type = splice_block %i32 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc20_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %u32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %u32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Expect(%N.loc4_14.1: %u32) {
// CHECK:STDOUT:   %N.loc4_14.2: %u32 = bind_symbolic_name N, 0 [symbolic = %N.loc4_14.2 (constants.%N.2)]
// CHECK:STDOUT:   %N.patt.loc4_14.2: %u32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_14.2 (constants.%N.patt.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Expect.1
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%N.loc5_9.1: %u32) {
// CHECK:STDOUT:   %N.loc5_9.2: %u32 = bind_symbolic_name N, 0 [symbolic = %N.loc5_9.2 (constants.%N.2)]
// CHECK:STDOUT:   %N.patt.loc5_9.2: %u32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc5_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:   %Expect.loc5_29.2: type = class_type @Expect, @Expect(%N.loc5_9.2) [symbolic = %Expect.loc5_29.2 (constants.%Expect.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Test.%Expect.loc5_29.2 (%Expect.1) [symbolic = %require_complete (constants.%require_complete.2)]
// CHECK:STDOUT:   %Expect.val: @Test.%Expect.loc5_29.2 (%Expect.1) = struct_value () [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: %u32) -> %return.param_patt: @Test.%Expect.loc5_29.2 (%Expect.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc5_41.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_41.2: init @Test.%Expect.loc5_29.2 (%Expect.1) = class_init (), %return [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:     %.loc5_42: init @Test.%Expect.loc5_29.2 (%Expect.1) = converted %.loc5_41.1, %.loc5_41.2 [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:     return %.loc5_42 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LeftShift.1(%a.param_patt: %u32, %b.param_patt: %i32) -> %u32 = "int.left_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Test.ref.loc10: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc10: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_0.loc10_18: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %int_0.loc10_21: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc10_18: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc10_18: <bound method> = bound_method %int_0.loc10_18, %impl.elem0.loc10_18 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_18: <specific function> = specific_function %Convert.bound.loc10_18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc10_18: init %u32 = call %Convert.specific_fn.loc10_18(%int_0.loc10_18) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_18.1: %u32 = value_of_initializer %int.convert_checked.loc10_18 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_18.2: %u32 = converted %int_0.loc10_18, %.loc10_18.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %impl.elem0.loc10_21: %Convert.type.11 = interface_witness_access constants.%interface.38, element0 [template = constants.%Convert.16]
// CHECK:STDOUT:   %Convert.bound.loc10_21: <bound method> = bound_method %int_0.loc10_21, %impl.elem0.loc10_21 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_21: <specific function> = specific_function %Convert.bound.loc10_21, @Convert.7(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc10_21: init %i32 = call %Convert.specific_fn.loc10_21(%int_0.loc10_21) [template = constants.%int_0.3]
// CHECK:STDOUT:   %.loc10_21.1: %i32 = value_of_initializer %int.convert_checked.loc10_21 [template = constants.%int_0.3]
// CHECK:STDOUT:   %.loc10_21.2: %i32 = converted %int_0.loc10_21, %.loc10_21.1 [template = constants.%int_0.3]
// CHECK:STDOUT:   %int.left_shift.loc10: init %u32 = call %LeftShift.ref.loc10(%.loc10_18.2, %.loc10_21.2) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_23.1: %u32 = value_of_initializer %int.left_shift.loc10 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_23.2: %u32 = converted %int.left_shift.loc10, %.loc10_23.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Test.specific_fn.loc10: <specific function> = specific_function %Test.ref.loc10, @Test(constants.%int_0.2) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc10_23.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc10: init %Expect.2 = call %Test.specific_fn.loc10() to %.loc10_23.3
// CHECK:STDOUT:   %Expect.ref.loc10: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc10_35: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc10_36: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc10_36: <bound method> = bound_method %int_0.loc10_35, %impl.elem0.loc10_36 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_36: <specific function> = specific_function %Convert.bound.loc10_36, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc10_36: init %u32 = call %Convert.specific_fn.loc10_36(%int_0.loc10_35) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_36.1: %u32 = value_of_initializer %int.convert_checked.loc10_36 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc10_36.2: %u32 = converted %int_0.loc10_35, %.loc10_36.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Expect.loc10: type = class_type @Expect, @Expect(constants.%int_0.2) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc10_23.4: ref %Expect.2 = temporary %.loc10_23.3, %Test.call.loc10
// CHECK:STDOUT:   %Test.ref.loc11: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc11: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_0.loc11_18: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc11_18: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_18: <bound method> = bound_method %int_0.loc11_18, %impl.elem0.loc11_18 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_18: <specific function> = specific_function %Convert.bound.loc11_18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_18: init %u32 = call %Convert.specific_fn.loc11_18(%int_0.loc11_18) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_18.1: %u32 = value_of_initializer %int.convert_checked.loc11_18 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_18.2: %u32 = converted %int_0.loc11_18, %.loc11_18.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %impl.elem0.loc11_21: %Convert.type.11 = interface_witness_access constants.%interface.38, element0 [template = constants.%Convert.16]
// CHECK:STDOUT:   %Convert.bound.loc11_21: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_21 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_21: <specific function> = specific_function %Convert.bound.loc11_21, @Convert.7(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc11_21: init %i32 = call %Convert.specific_fn.loc11_21(%int_1.loc11) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc11_21.1: %i32 = value_of_initializer %int.convert_checked.loc11_21 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc11_21.2: %i32 = converted %int_1.loc11, %.loc11_21.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.left_shift.loc11: init %u32 = call %LeftShift.ref.loc11(%.loc11_18.2, %.loc11_21.2) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_23.1: %u32 = value_of_initializer %int.left_shift.loc11 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_23.2: %u32 = converted %int.left_shift.loc11, %.loc11_23.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Test.specific_fn.loc11: <specific function> = specific_function %Test.ref.loc11, @Test(constants.%int_0.2) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc11_23.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc11: init %Expect.2 = call %Test.specific_fn.loc11() to %.loc11_23.3
// CHECK:STDOUT:   %Expect.ref.loc11: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc11_35: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc11_36: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_36: <bound method> = bound_method %int_0.loc11_35, %impl.elem0.loc11_36 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_36: <specific function> = specific_function %Convert.bound.loc11_36, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_36: init %u32 = call %Convert.specific_fn.loc11_36(%int_0.loc11_35) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_36.1: %u32 = value_of_initializer %int.convert_checked.loc11_36 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_36.2: %u32 = converted %int_0.loc11_35, %.loc11_36.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Expect.loc11: type = class_type @Expect, @Expect(constants.%int_0.2) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc11_23.4: ref %Expect.2 = temporary %.loc11_23.3, %Test.call.loc11
// CHECK:STDOUT:   %Test.ref.loc12: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc12: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_0.loc12_18: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %int_30.loc12: Core.IntLiteral = int_value 30 [template = constants.%int_30.1]
// CHECK:STDOUT:   %impl.elem0.loc12_18: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_18: <bound method> = bound_method %int_0.loc12_18, %impl.elem0.loc12_18 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_18: <specific function> = specific_function %Convert.bound.loc12_18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc12_18: init %u32 = call %Convert.specific_fn.loc12_18(%int_0.loc12_18) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc12_18.1: %u32 = value_of_initializer %int.convert_checked.loc12_18 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc12_18.2: %u32 = converted %int_0.loc12_18, %.loc12_18.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %impl.elem0.loc12_21: %Convert.type.11 = interface_witness_access constants.%interface.38, element0 [template = constants.%Convert.16]
// CHECK:STDOUT:   %Convert.bound.loc12_21: <bound method> = bound_method %int_30.loc12, %impl.elem0.loc12_21 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_21: <specific function> = specific_function %Convert.bound.loc12_21, @Convert.7(constants.%int_32) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc12_21: init %i32 = call %Convert.specific_fn.loc12_21(%int_30.loc12) [template = constants.%int_30.2]
// CHECK:STDOUT:   %.loc12_21.1: %i32 = value_of_initializer %int.convert_checked.loc12_21 [template = constants.%int_30.2]
// CHECK:STDOUT:   %.loc12_21.2: %i32 = converted %int_30.loc12, %.loc12_21.1 [template = constants.%int_30.2]
// CHECK:STDOUT:   %int.left_shift.loc12: init %u32 = call %LeftShift.ref.loc12(%.loc12_18.2, %.loc12_21.2) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc12_24.1: %u32 = value_of_initializer %int.left_shift.loc12 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc12_24.2: %u32 = converted %int.left_shift.loc12, %.loc12_24.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Test.specific_fn.loc12: <specific function> = specific_function %Test.ref.loc12, @Test(constants.%int_0.2) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc12_24.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc12: init %Expect.2 = call %Test.specific_fn.loc12() to %.loc12_24.3
// CHECK:STDOUT:   %Expect.ref.loc12: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc12_36: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc12_37: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_37: <bound method> = bound_method %int_0.loc12_36, %impl.elem0.loc12_37 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_37: <specific function> = specific_function %Convert.bound.loc12_37, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc12_37: init %u32 = call %Convert.specific_fn.loc12_37(%int_0.loc12_36) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc12_37.1: %u32 = value_of_initializer %int.convert_checked.loc12_37 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc12_37.2: %u32 = converted %int_0.loc12_36, %.loc12_37.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %Expect.loc12: type = class_type @Expect, @Expect(constants.%int_0.2) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc12_24.4: ref %Expect.2 = temporary %.loc12_24.3, %Test.call.loc12
// CHECK:STDOUT:   %Test.ref.loc13: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc13: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_30.loc13: Core.IntLiteral = int_value 30 [template = constants.%int_30.1]
// CHECK:STDOUT:   %impl.elem0.loc13_18: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc13_18: <bound method> = bound_method %int_1.loc13, %impl.elem0.loc13_18 [template = constants.%Convert.bound.5]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_18: <specific function> = specific_function %Convert.bound.loc13_18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %int.convert_checked.loc13_18: init %u32 = call %Convert.specific_fn.loc13_18(%int_1.loc13) [template = constants.%int_1.3]
// CHECK:STDOUT:   %.loc13_18.1: %u32 = value_of_initializer %int.convert_checked.loc13_18 [template = constants.%int_1.3]
// CHECK:STDOUT:   %.loc13_18.2: %u32 = converted %int_1.loc13, %.loc13_18.1 [template = constants.%int_1.3]
// CHECK:STDOUT:   %impl.elem0.loc13_21: %Convert.type.11 = interface_witness_access constants.%interface.38, element0 [template = constants.%Convert.16]
// CHECK:STDOUT:   %Convert.bound.loc13_21: <bound method> = bound_method %int_30.loc13, %impl.elem0.loc13_21 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_21: <specific function> = specific_function %Convert.bound.loc13_21, @Convert.7(constants.%int_32) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc13_21: init %i32 = call %Convert.specific_fn.loc13_21(%int_30.loc13) [template = constants.%int_30.2]
// CHECK:STDOUT:   %.loc13_21.1: %i32 = value_of_initializer %int.convert_checked.loc13_21 [template = constants.%int_30.2]
// CHECK:STDOUT:   %.loc13_21.2: %i32 = converted %int_30.loc13, %.loc13_21.1 [template = constants.%int_30.2]
// CHECK:STDOUT:   %int.left_shift.loc13: init %u32 = call %LeftShift.ref.loc13(%.loc13_18.2, %.loc13_21.2) [template = constants.%int_1073741824.1]
// CHECK:STDOUT:   %.loc13_24.1: %u32 = value_of_initializer %int.left_shift.loc13 [template = constants.%int_1073741824.1]
// CHECK:STDOUT:   %.loc13_24.2: %u32 = converted %int.left_shift.loc13, %.loc13_24.1 [template = constants.%int_1073741824.1]
// CHECK:STDOUT:   %Test.specific_fn.loc13: <specific function> = specific_function %Test.ref.loc13, @Test(constants.%int_1073741824.1) [template = constants.%Test.specific_fn.2]
// CHECK:STDOUT:   %.loc13_24.3: ref %Expect.3 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc13: init %Expect.3 = call %Test.specific_fn.loc13() to %.loc13_24.3
// CHECK:STDOUT:   %Expect.ref.loc13: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_1073741824: Core.IntLiteral = int_value 1073741824 [template = constants.%int_1073741824.2]
// CHECK:STDOUT:   %impl.elem0.loc13_47: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc13_47: <bound method> = bound_method %int_1073741824, %impl.elem0.loc13_47 [template = constants.%Convert.bound.6]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_47: <specific function> = specific_function %Convert.bound.loc13_47, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.6]
// CHECK:STDOUT:   %int.convert_checked.loc13_47: init %u32 = call %Convert.specific_fn.loc13_47(%int_1073741824) [template = constants.%int_1073741824.1]
// CHECK:STDOUT:   %.loc13_47.1: %u32 = value_of_initializer %int.convert_checked.loc13_47 [template = constants.%int_1073741824.1]
// CHECK:STDOUT:   %.loc13_47.2: %u32 = converted %int_1073741824, %.loc13_47.1 [template = constants.%int_1073741824.1]
// CHECK:STDOUT:   %Expect.loc13: type = class_type @Expect, @Expect(constants.%int_1073741824.1) [template = constants.%Expect.3]
// CHECK:STDOUT:   %.loc13_24.4: ref %Expect.3 = temporary %.loc13_24.3, %Test.call.loc13
// CHECK:STDOUT:   %Test.ref.loc14: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc14: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %impl.elem0.loc14_18: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc14_18: <bound method> = bound_method %int_5, %impl.elem0.loc14_18 [template = constants.%Convert.bound.7]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_18: <specific function> = specific_function %Convert.bound.loc14_18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.7]
// CHECK:STDOUT:   %int.convert_checked.loc14_18: init %u32 = call %Convert.specific_fn.loc14_18(%int_5) [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc14_18.1: %u32 = value_of_initializer %int.convert_checked.loc14_18 [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc14_18.2: %u32 = converted %int_5, %.loc14_18.1 [template = constants.%int_5.2]
// CHECK:STDOUT:   %impl.elem0.loc14_21: %Convert.type.11 = interface_witness_access constants.%interface.38, element0 [template = constants.%Convert.16]
// CHECK:STDOUT:   %Convert.bound.loc14_21: <bound method> = bound_method %int_2, %impl.elem0.loc14_21 [template = constants.%Convert.bound.8]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_21: <specific function> = specific_function %Convert.bound.loc14_21, @Convert.7(constants.%int_32) [template = constants.%Convert.specific_fn.8]
// CHECK:STDOUT:   %int.convert_checked.loc14_21: init %i32 = call %Convert.specific_fn.loc14_21(%int_2) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc14_21.1: %i32 = value_of_initializer %int.convert_checked.loc14_21 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc14_21.2: %i32 = converted %int_2, %.loc14_21.1 [template = constants.%int_2.2]
// CHECK:STDOUT:   %int.left_shift.loc14: init %u32 = call %LeftShift.ref.loc14(%.loc14_18.2, %.loc14_21.2) [template = constants.%int_20.1]
// CHECK:STDOUT:   %.loc14_23.1: %u32 = value_of_initializer %int.left_shift.loc14 [template = constants.%int_20.1]
// CHECK:STDOUT:   %.loc14_23.2: %u32 = converted %int.left_shift.loc14, %.loc14_23.1 [template = constants.%int_20.1]
// CHECK:STDOUT:   %Test.specific_fn.loc14: <specific function> = specific_function %Test.ref.loc14, @Test(constants.%int_20.1) [template = constants.%Test.specific_fn.3]
// CHECK:STDOUT:   %.loc14_23.3: ref %Expect.4 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc14: init %Expect.4 = call %Test.specific_fn.loc14() to %.loc14_23.3
// CHECK:STDOUT:   %Expect.ref.loc14: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [template = constants.%int_20.2]
// CHECK:STDOUT:   %impl.elem0.loc14_37: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc14_37: <bound method> = bound_method %int_20, %impl.elem0.loc14_37 [template = constants.%Convert.bound.9]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_37: <specific function> = specific_function %Convert.bound.loc14_37, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.9]
// CHECK:STDOUT:   %int.convert_checked.loc14_37: init %u32 = call %Convert.specific_fn.loc14_37(%int_20) [template = constants.%int_20.1]
// CHECK:STDOUT:   %.loc14_37.1: %u32 = value_of_initializer %int.convert_checked.loc14_37 [template = constants.%int_20.1]
// CHECK:STDOUT:   %.loc14_37.2: %u32 = converted %int_20, %.loc14_37.1 [template = constants.%int_20.1]
// CHECK:STDOUT:   %Expect.loc14: type = class_type @Expect, @Expect(constants.%int_20.1) [template = constants.%Expect.4]
// CHECK:STDOUT:   %.loc14_23.4: ref %Expect.4 = temporary %.loc14_23.3, %Test.call.loc14
// CHECK:STDOUT:   %Test.ref.loc15: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc15: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_4294967295: Core.IntLiteral = int_value 4294967295 [template = constants.%int_4294967295.1]
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc15_18: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc15_18: <bound method> = bound_method %int_4294967295, %impl.elem0.loc15_18 [template = constants.%Convert.bound.10]
// CHECK:STDOUT:   %Convert.specific_fn.loc15_18: <specific function> = specific_function %Convert.bound.loc15_18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.10]
// CHECK:STDOUT:   %int.convert_checked.loc15_18: init %u32 = call %Convert.specific_fn.loc15_18(%int_4294967295) [template = constants.%int_4294967295.2]
// CHECK:STDOUT:   %.loc15_18.1: %u32 = value_of_initializer %int.convert_checked.loc15_18 [template = constants.%int_4294967295.2]
// CHECK:STDOUT:   %.loc15_18.2: %u32 = converted %int_4294967295, %.loc15_18.1 [template = constants.%int_4294967295.2]
// CHECK:STDOUT:   %impl.elem0.loc15_31: %Convert.type.11 = interface_witness_access constants.%interface.38, element0 [template = constants.%Convert.16]
// CHECK:STDOUT:   %Convert.bound.loc15_31: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15_31 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc15_31: <specific function> = specific_function %Convert.bound.loc15_31, @Convert.7(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc15_31: init %i32 = call %Convert.specific_fn.loc15_31(%int_1.loc15) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc15_31.1: %i32 = value_of_initializer %int.convert_checked.loc15_31 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc15_31.2: %i32 = converted %int_1.loc15, %.loc15_31.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.left_shift.loc15: init %u32 = call %LeftShift.ref.loc15(%.loc15_18.2, %.loc15_31.2) [template = constants.%int_4294967294.1]
// CHECK:STDOUT:   %.loc15_33.1: %u32 = value_of_initializer %int.left_shift.loc15 [template = constants.%int_4294967294.1]
// CHECK:STDOUT:   %.loc15_33.2: %u32 = converted %int.left_shift.loc15, %.loc15_33.1 [template = constants.%int_4294967294.1]
// CHECK:STDOUT:   %Test.specific_fn.loc15: <specific function> = specific_function %Test.ref.loc15, @Test(constants.%int_4294967294.1) [template = constants.%Test.specific_fn.4]
// CHECK:STDOUT:   %.loc15_33.3: ref %Expect.5 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc15: init %Expect.5 = call %Test.specific_fn.loc15() to %.loc15_33.3
// CHECK:STDOUT:   %Expect.ref.loc15: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_4294967294.loc15: Core.IntLiteral = int_value 4294967294 [template = constants.%int_4294967294.2]
// CHECK:STDOUT:   %impl.elem0.loc15_56: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc15_56: <bound method> = bound_method %int_4294967294.loc15, %impl.elem0.loc15_56 [template = constants.%Convert.bound.11]
// CHECK:STDOUT:   %Convert.specific_fn.loc15_56: <specific function> = specific_function %Convert.bound.loc15_56, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.11]
// CHECK:STDOUT:   %int.convert_checked.loc15_56: init %u32 = call %Convert.specific_fn.loc15_56(%int_4294967294.loc15) [template = constants.%int_4294967294.1]
// CHECK:STDOUT:   %.loc15_56.1: %u32 = value_of_initializer %int.convert_checked.loc15_56 [template = constants.%int_4294967294.1]
// CHECK:STDOUT:   %.loc15_56.2: %u32 = converted %int_4294967294.loc15, %.loc15_56.1 [template = constants.%int_4294967294.1]
// CHECK:STDOUT:   %Expect.loc15: type = class_type @Expect, @Expect(constants.%int_4294967294.1) [template = constants.%Expect.5]
// CHECK:STDOUT:   %.loc15_33.4: ref %Expect.5 = temporary %.loc15_33.3, %Test.call.loc15
// CHECK:STDOUT:   %Test.ref.loc16: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc16: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_4294967294.loc16: Core.IntLiteral = int_value 4294967294 [template = constants.%int_4294967294.2]
// CHECK:STDOUT:   %int_1.loc16: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc16_18: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc16_18: <bound method> = bound_method %int_4294967294.loc16, %impl.elem0.loc16_18 [template = constants.%Convert.bound.11]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_18: <specific function> = specific_function %Convert.bound.loc16_18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.11]
// CHECK:STDOUT:   %int.convert_checked.loc16_18: init %u32 = call %Convert.specific_fn.loc16_18(%int_4294967294.loc16) [template = constants.%int_4294967294.1]
// CHECK:STDOUT:   %.loc16_18.1: %u32 = value_of_initializer %int.convert_checked.loc16_18 [template = constants.%int_4294967294.1]
// CHECK:STDOUT:   %.loc16_18.2: %u32 = converted %int_4294967294.loc16, %.loc16_18.1 [template = constants.%int_4294967294.1]
// CHECK:STDOUT:   %impl.elem0.loc16_31: %Convert.type.11 = interface_witness_access constants.%interface.38, element0 [template = constants.%Convert.16]
// CHECK:STDOUT:   %Convert.bound.loc16_31: <bound method> = bound_method %int_1.loc16, %impl.elem0.loc16_31 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_31: <specific function> = specific_function %Convert.bound.loc16_31, @Convert.7(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc16_31: init %i32 = call %Convert.specific_fn.loc16_31(%int_1.loc16) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc16_31.1: %i32 = value_of_initializer %int.convert_checked.loc16_31 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc16_31.2: %i32 = converted %int_1.loc16, %.loc16_31.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.left_shift.loc16: init %u32 = call %LeftShift.ref.loc16(%.loc16_18.2, %.loc16_31.2) [template = constants.%int_4294967292.1]
// CHECK:STDOUT:   %.loc16_33.1: %u32 = value_of_initializer %int.left_shift.loc16 [template = constants.%int_4294967292.1]
// CHECK:STDOUT:   %.loc16_33.2: %u32 = converted %int.left_shift.loc16, %.loc16_33.1 [template = constants.%int_4294967292.1]
// CHECK:STDOUT:   %Test.specific_fn.loc16: <specific function> = specific_function %Test.ref.loc16, @Test(constants.%int_4294967292.1) [template = constants.%Test.specific_fn.5]
// CHECK:STDOUT:   %.loc16_33.3: ref %Expect.6 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc16: init %Expect.6 = call %Test.specific_fn.loc16() to %.loc16_33.3
// CHECK:STDOUT:   %Expect.ref.loc16: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_4294967292: Core.IntLiteral = int_value 4294967292 [template = constants.%int_4294967292.2]
// CHECK:STDOUT:   %impl.elem0.loc16_56: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc16_56: <bound method> = bound_method %int_4294967292, %impl.elem0.loc16_56 [template = constants.%Convert.bound.12]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_56: <specific function> = specific_function %Convert.bound.loc16_56, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.12]
// CHECK:STDOUT:   %int.convert_checked.loc16_56: init %u32 = call %Convert.specific_fn.loc16_56(%int_4294967292) [template = constants.%int_4294967292.1]
// CHECK:STDOUT:   %.loc16_56.1: %u32 = value_of_initializer %int.convert_checked.loc16_56 [template = constants.%int_4294967292.1]
// CHECK:STDOUT:   %.loc16_56.2: %u32 = converted %int_4294967292, %.loc16_56.1 [template = constants.%int_4294967292.1]
// CHECK:STDOUT:   %Expect.loc16: type = class_type @Expect, @Expect(constants.%int_4294967292.1) [template = constants.%Expect.6]
// CHECK:STDOUT:   %.loc16_33.4: ref %Expect.6 = temporary %.loc16_33.3, %Test.call.loc16
// CHECK:STDOUT:   %Test.ref.loc17: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc17: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_2882400001: Core.IntLiteral = int_value 2882400001 [template = constants.%int_2882400001.1]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [template = constants.%int_8.1]
// CHECK:STDOUT:   %impl.elem0.loc17_18: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc17_18: <bound method> = bound_method %int_2882400001, %impl.elem0.loc17_18 [template = constants.%Convert.bound.13]
// CHECK:STDOUT:   %Convert.specific_fn.loc17_18: <specific function> = specific_function %Convert.bound.loc17_18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.13]
// CHECK:STDOUT:   %int.convert_checked.loc17_18: init %u32 = call %Convert.specific_fn.loc17_18(%int_2882400001) [template = constants.%int_2882400001.2]
// CHECK:STDOUT:   %.loc17_18.1: %u32 = value_of_initializer %int.convert_checked.loc17_18 [template = constants.%int_2882400001.2]
// CHECK:STDOUT:   %.loc17_18.2: %u32 = converted %int_2882400001, %.loc17_18.1 [template = constants.%int_2882400001.2]
// CHECK:STDOUT:   %impl.elem0.loc17_31: %Convert.type.11 = interface_witness_access constants.%interface.38, element0 [template = constants.%Convert.16]
// CHECK:STDOUT:   %Convert.bound.loc17_31: <bound method> = bound_method %int_8, %impl.elem0.loc17_31 [template = constants.%Convert.bound.14]
// CHECK:STDOUT:   %Convert.specific_fn.loc17_31: <specific function> = specific_function %Convert.bound.loc17_31, @Convert.7(constants.%int_32) [template = constants.%Convert.specific_fn.14]
// CHECK:STDOUT:   %int.convert_checked.loc17_31: init %i32 = call %Convert.specific_fn.loc17_31(%int_8) [template = constants.%int_8.2]
// CHECK:STDOUT:   %.loc17_31.1: %i32 = value_of_initializer %int.convert_checked.loc17_31 [template = constants.%int_8.2]
// CHECK:STDOUT:   %.loc17_31.2: %i32 = converted %int_8, %.loc17_31.1 [template = constants.%int_8.2]
// CHECK:STDOUT:   %int.left_shift.loc17: init %u32 = call %LeftShift.ref.loc17(%.loc17_18.2, %.loc17_31.2) [template = constants.%int_3454992640.1]
// CHECK:STDOUT:   %.loc17_33.1: %u32 = value_of_initializer %int.left_shift.loc17 [template = constants.%int_3454992640.1]
// CHECK:STDOUT:   %.loc17_33.2: %u32 = converted %int.left_shift.loc17, %.loc17_33.1 [template = constants.%int_3454992640.1]
// CHECK:STDOUT:   %Test.specific_fn.loc17: <specific function> = specific_function %Test.ref.loc17, @Test(constants.%int_3454992640.1) [template = constants.%Test.specific_fn.6]
// CHECK:STDOUT:   %.loc17_33.3: ref %Expect.7 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc17: init %Expect.7 = call %Test.specific_fn.loc17() to %.loc17_33.3
// CHECK:STDOUT:   %Expect.ref.loc17: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_3454992640: Core.IntLiteral = int_value 3454992640 [template = constants.%int_3454992640.2]
// CHECK:STDOUT:   %impl.elem0.loc17_56: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc17_56: <bound method> = bound_method %int_3454992640, %impl.elem0.loc17_56 [template = constants.%Convert.bound.15]
// CHECK:STDOUT:   %Convert.specific_fn.loc17_56: <specific function> = specific_function %Convert.bound.loc17_56, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.15]
// CHECK:STDOUT:   %int.convert_checked.loc17_56: init %u32 = call %Convert.specific_fn.loc17_56(%int_3454992640) [template = constants.%int_3454992640.1]
// CHECK:STDOUT:   %.loc17_56.1: %u32 = value_of_initializer %int.convert_checked.loc17_56 [template = constants.%int_3454992640.1]
// CHECK:STDOUT:   %.loc17_56.2: %u32 = converted %int_3454992640, %.loc17_56.1 [template = constants.%int_3454992640.1]
// CHECK:STDOUT:   %Expect.loc17: type = class_type @Expect, @Expect(constants.%int_3454992640.1) [template = constants.%Expect.7]
// CHECK:STDOUT:   %.loc17_33.4: ref %Expect.7 = temporary %.loc17_33.3, %Test.call.loc17
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %u32, %b.param_patt: %i32) -> %u32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %LeftShift.ref: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %a.ref: %u32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.left_shift: init %u32 = call %LeftShift.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc21_25.1: %u32 = value_of_initializer %int.left_shift
// CHECK:STDOUT:   %.loc21_25.2: %u32 = converted %int.left_shift, %.loc21_25.1
// CHECK:STDOUT:   return %.loc21_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%N.2) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%N.2
// CHECK:STDOUT:   %N.patt.loc4_14.2 => constants.%N.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(@Test.%N.loc5_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%N.2) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%N.2
// CHECK:STDOUT:   %N.patt.loc5_9.2 => constants.%N.2
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_0.2) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_0.2
// CHECK:STDOUT:   %N.patt.loc5_9.2 => constants.%int_0.2
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.2
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_0.2) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_0.2
// CHECK:STDOUT:   %N.patt.loc4_14.2 => constants.%int_0.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_1073741824.1) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_1073741824.1
// CHECK:STDOUT:   %N.patt.loc5_9.2 => constants.%int_1073741824.1
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.2
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_1073741824.1) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_1073741824.1
// CHECK:STDOUT:   %N.patt.loc4_14.2 => constants.%int_1073741824.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_20.1) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_20.1
// CHECK:STDOUT:   %N.patt.loc5_9.2 => constants.%int_20.1
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.2
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_20.1) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_20.1
// CHECK:STDOUT:   %N.patt.loc4_14.2 => constants.%int_20.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_4294967294.1) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_4294967294.1
// CHECK:STDOUT:   %N.patt.loc5_9.2 => constants.%int_4294967294.1
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.2
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_4294967294.1) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_4294967294.1
// CHECK:STDOUT:   %N.patt.loc4_14.2 => constants.%int_4294967294.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_4294967292.1) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_4294967292.1
// CHECK:STDOUT:   %N.patt.loc5_9.2 => constants.%int_4294967292.1
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.2
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_4294967292.1) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_4294967292.1
// CHECK:STDOUT:   %N.patt.loc4_14.2 => constants.%int_4294967292.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_3454992640.1) {
// CHECK:STDOUT:   %N.loc5_9.2 => constants.%int_3454992640.1
// CHECK:STDOUT:   %N.patt.loc5_9.2 => constants.%int_3454992640.1
// CHECK:STDOUT:   %Expect.loc5_29.2 => constants.%Expect.7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.2
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_3454992640.1) {
// CHECK:STDOUT:   %N.loc4_14.2 => constants.%int_3454992640.1
// CHECK:STDOUT:   %N.patt.loc4_14.2 => constants.%int_3454992640.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %LeftShift.type: type = fn_type @LeftShift [template]
// CHECK:STDOUT:   %LeftShift: %LeftShift.type = struct_value () [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Expect.type: type = generic_class_type @Expect [template]
// CHECK:STDOUT:   %Expect.generic: %Expect.type = struct_value () [template]
// CHECK:STDOUT:   %Expect.1: type = class_type @Expect, @Expect(%N) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [template]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [template]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Expect.1 [symbolic]
// CHECK:STDOUT:   %Expect.val.1: %Expect.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Expect.2: type = class_type @Expect, @Expect(%int_0) [template]
// CHECK:STDOUT:   %Test.specific_fn.1: <specific function> = specific_function %Test, @Test(%int_0) [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_30: Core.IntLiteral = int_value 30 [template]
// CHECK:STDOUT:   %int_1000000000: Core.IntLiteral = int_value 1000000000 [template]
// CHECK:STDOUT:   %Expect.3: type = class_type @Expect, @Expect(%int_1) [template]
// CHECK:STDOUT:   %Test.specific_fn.2: <specific function> = specific_function %Test, @Test(%int_1) [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Expect.4: type = class_type @Expect, @Expect(%int_2) [template]
// CHECK:STDOUT:   %Test.specific_fn.3: <specific function> = specific_function %Test, @Test(%int_2) [template]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %Expect.5: type = class_type @Expect, @Expect(%int_4) [template]
// CHECK:STDOUT:   %Test.specific_fn.4: <specific function> = specific_function %Test, @Test(%int_4) [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %Expect.6: type = class_type @Expect, @Expect(%int_12) [template]
// CHECK:STDOUT:   %Test.specific_fn.5: <specific function> = specific_function %Test, @Test(%int_12) [template]
// CHECK:STDOUT:   %int_1073741824: Core.IntLiteral = int_value 1073741824 [template]
// CHECK:STDOUT:   %Expect.7: type = class_type @Expect, @Expect(%int_1073741824) [template]
// CHECK:STDOUT:   %Test.specific_fn.6: <specific function> = specific_function %Test, @Test(%int_1073741824) [template]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [template]
// CHECK:STDOUT:   %Expect.8: type = class_type @Expect, @Expect(%int_20) [template]
// CHECK:STDOUT:   %Test.specific_fn.7: <specific function> = specific_function %Test, @Test(%int_20) [template]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%Op.2) [template]
// CHECK:STDOUT:   %Op.bound.1: <bound method> = bound_method %int_1, %Op.2 [template]
// CHECK:STDOUT:   %int_-1: Core.IntLiteral = int_value -1 [template]
// CHECK:STDOUT:   %Expect.9: type = class_type @Expect, @Expect(%int_-1) [template]
// CHECK:STDOUT:   %Test.specific_fn.8: <specific function> = specific_function %Test, @Test(%int_-1) [template]
// CHECK:STDOUT:   %int_-2: Core.IntLiteral = int_value -2 [template]
// CHECK:STDOUT:   %Expect.10: type = class_type @Expect, @Expect(%int_-2) [template]
// CHECK:STDOUT:   %Test.specific_fn.9: <specific function> = specific_function %Test, @Test(%int_-2) [template]
// CHECK:STDOUT:   %Op.bound.2: <bound method> = bound_method %int_2, %Op.2 [template]
// CHECK:STDOUT:   %int_-4: Core.IntLiteral = int_value -4 [template]
// CHECK:STDOUT:   %Expect.11: type = class_type @Expect, @Expect(%int_-4) [template]
// CHECK:STDOUT:   %Test.specific_fn.10: <specific function> = specific_function %Test, @Test(%int_-4) [template]
// CHECK:STDOUT:   %Op.bound.3: <bound method> = bound_method %int_4, %Op.2 [template]
// CHECK:STDOUT:   %Op.bound.4: <bound method> = bound_method %int_3, %Op.2 [template]
// CHECK:STDOUT:   %int_-3: Core.IntLiteral = int_value -3 [template]
// CHECK:STDOUT:   %int_-6: Core.IntLiteral = int_value -6 [template]
// CHECK:STDOUT:   %Expect.12: type = class_type @Expect, @Expect(%int_-6) [template]
// CHECK:STDOUT:   %Test.specific_fn.11: <specific function> = specific_function %Test, @Test(%int_-6) [template]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %Op.bound.5: <bound method> = bound_method %int_6, %Op.2 [template]
// CHECK:STDOUT:   %int_4294967295: Core.IntLiteral = int_value 4294967295 [template]
// CHECK:STDOUT:   %int_8589934590: Core.IntLiteral = int_value 8589934590 [template]
// CHECK:STDOUT:   %Expect.13: type = class_type @Expect, @Expect(%int_8589934590) [template]
// CHECK:STDOUT:   %Test.specific_fn.12: <specific function> = specific_function %Test, @Test(%int_8589934590) [template]
// CHECK:STDOUT:   %int_4294967294: Core.IntLiteral = int_value 4294967294 [template]
// CHECK:STDOUT:   %int_8589934588: Core.IntLiteral = int_value 8589934588 [template]
// CHECK:STDOUT:   %Expect.14: type = class_type @Expect, @Expect(%int_8589934588) [template]
// CHECK:STDOUT:   %Test.specific_fn.13: <specific function> = specific_function %Test, @Test(%int_8589934588) [template]
// CHECK:STDOUT:   %int_2882400001: Core.IntLiteral = int_value 2882400001 [template]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [template]
// CHECK:STDOUT:   %int_737894400256: Core.IntLiteral = int_value 737894400256 [template]
// CHECK:STDOUT:   %Expect.15: type = class_type @Expect, @Expect(%int_737894400256) [template]
// CHECK:STDOUT:   %Test.specific_fn.14: <specific function> = specific_function %Test, @Test(%int_737894400256) [template]
// CHECK:STDOUT:   %int_9223372036854775807: Core.IntLiteral = int_value 9223372036854775807 [template]
// CHECK:STDOUT:   %int_18446744073709551614: Core.IntLiteral = int_value 18446744073709551614 [template]
// CHECK:STDOUT:   %Expect.16: type = class_type @Expect, @Expect(%int_18446744073709551614) [template]
// CHECK:STDOUT:   %Test.specific_fn.15: <specific function> = specific_function %Test, @Test(%int_18446744073709551614) [template]
// CHECK:STDOUT:   %int_18446744073709551615: Core.IntLiteral = int_value 18446744073709551615 [template]
// CHECK:STDOUT:   %int_36893488147419103230: Core.IntLiteral = int_value 36893488147419103230 [template]
// CHECK:STDOUT:   %Expect.17: type = class_type @Expect, @Expect(%int_36893488147419103230) [template]
// CHECK:STDOUT:   %Test.specific_fn.16: <specific function> = specific_function %Test, @Test(%int_36893488147419103230) [template]
// CHECK:STDOUT:   %Expect.val.2: %Expect.2 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.3: %Expect.3 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.4: %Expect.4 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.5: %Expect.5 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.6: %Expect.6 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.7: %Expect.7 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.8: %Expect.8 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.9: %Expect.9 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.10: %Expect.10 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.11: %Expect.11 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.12: %Expect.12 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.13: %Expect.13 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.14: %Expect.14 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.15: %Expect.15 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.16: %Expect.16 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.17: %Expect.17 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IntLiteral = %import_ref.1
// CHECK:STDOUT:     .Negate = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .LeftShift = %LeftShift.decl
// CHECK:STDOUT:     .Expect = %Expect.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %LeftShift.decl: %LeftShift.type = fn_decl @LeftShift [template = constants.%LeftShift] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: Core.IntLiteral = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: Core.IntLiteral = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc4_61: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_65: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_77: init type = call %IntLiteral.ref.loc4_65() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_77.1: type = value_of_initializer %int_literal.make_type.loc4_77 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_77.2: type = converted %int_literal.make_type.loc4_77, %.loc4_77.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc4_33.3: type = splice_block %.loc4_33.2 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc4_17: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc4_21: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc4_33: init type = call %IntLiteral.ref.loc4_21() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_33.1: type = value_of_initializer %int_literal.make_type.loc4_33 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_33.2: type = converted %int_literal.make_type.loc4_33, %.loc4_33.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: Core.IntLiteral = value_param runtime_param1
// CHECK:STDOUT:     %.loc4_55.3: type = splice_block %.loc4_55.2 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc4_39: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc4_43: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc4_55: init type = call %IntLiteral.ref.loc4_43() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_55.1: type = value_of_initializer %int_literal.make_type.loc4_55 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_55.2: type = converted %int_literal.make_type.loc4_55, %.loc4_55.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: Core.IntLiteral = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param2
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Expect.decl: %Expect.type = class_decl @Expect [template = constants.%Expect.generic] {
// CHECK:STDOUT:     %N.patt.loc6_14.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_14.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt.loc6_14.1, runtime_param<invalid> [symbolic = %N.patt.loc6_14.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc6_34.3: type = splice_block %.loc6_34.2 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_34.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_34.2: type = converted %int_literal.make_type, %.loc6_34.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_14.1: Core.IntLiteral = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc6_14.2 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [template = constants.%Test] {
// CHECK:STDOUT:     %N.patt.loc7_9.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc7_9.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt.loc7_9.1, runtime_param<invalid> [symbolic = %N.patt.loc7_9.2 (constants.%N.patt)]
// CHECK:STDOUT:     %return.patt: @Test.%Expect.loc7_43.2 (%Expect.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Test.%Expect.loc7_43.2 (%Expect.1) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Expect.ref: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:     %N.ref: Core.IntLiteral = name_ref N, %N.loc7_9.1 [symbolic = %N.loc7_9.2 (constants.%N)]
// CHECK:STDOUT:     %Expect.loc7_43.1: type = class_type @Expect, @Expect(constants.%N) [symbolic = %Expect.loc7_43.2 (constants.%Expect.1)]
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc7_29.3: type = splice_block %.loc7_29.2 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc7_29.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc7_29.2: type = converted %int_literal.make_type, %.loc7_29.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc7_9.1: Core.IntLiteral = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc7_9.2 (constants.%N)]
// CHECK:STDOUT:     %return.param: ref @Test.%Expect.loc7_43.2 (%Expect.1) = out_param runtime_param0
// CHECK:STDOUT:     %return: ref @Test.%Expect.loc7_43.2 (%Expect.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Expect(%N.loc6_14.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc6_14.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc6_14.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc6_14.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_14.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Expect.1
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LeftShift(%a.param_patt: Core.IntLiteral, %b.param_patt: Core.IntLiteral) -> Core.IntLiteral = "int.left_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%N.loc7_9.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc7_9.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc7_9.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc7_9.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc7_9.2 (constants.%N.patt)]
// CHECK:STDOUT:   %Expect.loc7_43.2: type = class_type @Expect, @Expect(%N.loc7_9.2) [symbolic = %Expect.loc7_43.2 (constants.%Expect.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Test.%Expect.loc7_43.2 (%Expect.1) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %Expect.val: @Test.%Expect.loc7_43.2 (%Expect.1) = struct_value () [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: Core.IntLiteral) -> %return.param_patt: @Test.%Expect.loc7_43.2 (%Expect.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc7_55.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc7_55.2: init @Test.%Expect.loc7_43.2 (%Expect.1) = class_init (), %return [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:     %.loc7_56: init @Test.%Expect.loc7_43.2 (%Expect.1) = converted %.loc7_55.1, %.loc7_55.2 [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:     return %.loc7_56 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Test.ref.loc11: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc11: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_0.loc11_18: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int_0.loc11_21: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int.left_shift.loc11: init Core.IntLiteral = call %LeftShift.ref.loc11(%int_0.loc11_18, %int_0.loc11_21) [template = constants.%int_0]
// CHECK:STDOUT:   %.loc11_23.1: Core.IntLiteral = value_of_initializer %int.left_shift.loc11 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc11_23.2: Core.IntLiteral = converted %int.left_shift.loc11, %.loc11_23.1 [template = constants.%int_0]
// CHECK:STDOUT:   %Test.specific_fn.loc11: <specific function> = specific_function %Test.ref.loc11, @Test(constants.%int_0) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc11_23.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc11: init %Expect.2 = call %Test.specific_fn.loc11() to %.loc11_23.3
// CHECK:STDOUT:   %Expect.ref.loc11: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc11_35: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %Expect.loc11: type = class_type @Expect, @Expect(constants.%int_0) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc11_23.4: ref %Expect.2 = temporary %.loc11_23.3, %Test.call.loc11
// CHECK:STDOUT:   %Test.ref.loc12: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc12: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_0.loc12_18: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int_0.loc12_21: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int.left_shift.loc12: init Core.IntLiteral = call %LeftShift.ref.loc12(%int_0.loc12_18, %int_0.loc12_21) [template = constants.%int_0]
// CHECK:STDOUT:   %.loc12_23.1: Core.IntLiteral = value_of_initializer %int.left_shift.loc12 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc12_23.2: Core.IntLiteral = converted %int.left_shift.loc12, %.loc12_23.1 [template = constants.%int_0]
// CHECK:STDOUT:   %Test.specific_fn.loc12: <specific function> = specific_function %Test.ref.loc12, @Test(constants.%int_0) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc12_23.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc12: init %Expect.2 = call %Test.specific_fn.loc12() to %.loc12_23.3
// CHECK:STDOUT:   %Expect.ref.loc12: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc12_35: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %Expect.loc12: type = class_type @Expect, @Expect(constants.%int_0) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc12_23.4: ref %Expect.2 = temporary %.loc12_23.3, %Test.call.loc12
// CHECK:STDOUT:   %Test.ref.loc13: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc13: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_0.loc13_18: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int.left_shift.loc13: init Core.IntLiteral = call %LeftShift.ref.loc13(%int_0.loc13_18, %int_1.loc13) [template = constants.%int_0]
// CHECK:STDOUT:   %.loc13_23.1: Core.IntLiteral = value_of_initializer %int.left_shift.loc13 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc13_23.2: Core.IntLiteral = converted %int.left_shift.loc13, %.loc13_23.1 [template = constants.%int_0]
// CHECK:STDOUT:   %Test.specific_fn.loc13: <specific function> = specific_function %Test.ref.loc13, @Test(constants.%int_0) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc13_23.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc13: init %Expect.2 = call %Test.specific_fn.loc13() to %.loc13_23.3
// CHECK:STDOUT:   %Expect.ref.loc13: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc13_35: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %Expect.loc13: type = class_type @Expect, @Expect(constants.%int_0) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc13_23.4: ref %Expect.2 = temporary %.loc13_23.3, %Test.call.loc13
// CHECK:STDOUT:   %Test.ref.loc14: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc14: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_0.loc14_18: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int_30.loc14: Core.IntLiteral = int_value 30 [template = constants.%int_30]
// CHECK:STDOUT:   %int.left_shift.loc14: init Core.IntLiteral = call %LeftShift.ref.loc14(%int_0.loc14_18, %int_30.loc14) [template = constants.%int_0]
// CHECK:STDOUT:   %.loc14_24.1: Core.IntLiteral = value_of_initializer %int.left_shift.loc14 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc14_24.2: Core.IntLiteral = converted %int.left_shift.loc14, %.loc14_24.1 [template = constants.%int_0]
// CHECK:STDOUT:   %Test.specific_fn.loc14: <specific function> = specific_function %Test.ref.loc14, @Test(constants.%int_0) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc14_24.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc14: init %Expect.2 = call %Test.specific_fn.loc14() to %.loc14_24.3
// CHECK:STDOUT:   %Expect.ref.loc14: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc14_36: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %Expect.loc14: type = class_type @Expect, @Expect(constants.%int_0) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc14_24.4: ref %Expect.2 = temporary %.loc14_24.3, %Test.call.loc14
// CHECK:STDOUT:   %Test.ref.loc15: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc15: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_0.loc15_18: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int_1000000000: Core.IntLiteral = int_value 1000000000 [template = constants.%int_1000000000]
// CHECK:STDOUT:   %int.left_shift.loc15: init Core.IntLiteral = call %LeftShift.ref.loc15(%int_0.loc15_18, %int_1000000000) [template = constants.%int_0]
// CHECK:STDOUT:   %.loc15_35.1: Core.IntLiteral = value_of_initializer %int.left_shift.loc15 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc15_35.2: Core.IntLiteral = converted %int.left_shift.loc15, %.loc15_35.1 [template = constants.%int_0]
// CHECK:STDOUT:   %Test.specific_fn.loc15: <specific function> = specific_function %Test.ref.loc15, @Test(constants.%int_0) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc15_35.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc15: init %Expect.2 = call %Test.specific_fn.loc15() to %.loc15_35.3
// CHECK:STDOUT:   %Expect.ref.loc15: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc15_47: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %Expect.loc15: type = class_type @Expect, @Expect(constants.%int_0) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc15_35.4: ref %Expect.2 = temporary %.loc15_35.3, %Test.call.loc15
// CHECK:STDOUT:   %Test.ref.loc18: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc18: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc18_18: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_0.loc18: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int.left_shift.loc18: init Core.IntLiteral = call %LeftShift.ref.loc18(%int_1.loc18_18, %int_0.loc18) [template = constants.%int_1]
// CHECK:STDOUT:   %.loc18_23.1: Core.IntLiteral = value_of_initializer %int.left_shift.loc18 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc18_23.2: Core.IntLiteral = converted %int.left_shift.loc18, %.loc18_23.1 [template = constants.%int_1]
// CHECK:STDOUT:   %Test.specific_fn.loc18: <specific function> = specific_function %Test.ref.loc18, @Test(constants.%int_1) [template = constants.%Test.specific_fn.2]
// CHECK:STDOUT:   %.loc18_23.3: ref %Expect.3 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc18: init %Expect.3 = call %Test.specific_fn.loc18() to %.loc18_23.3
// CHECK:STDOUT:   %Expect.ref.loc18: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_1.loc18_35: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %Expect.loc18: type = class_type @Expect, @Expect(constants.%int_1) [template = constants.%Expect.3]
// CHECK:STDOUT:   %.loc18_23.4: ref %Expect.3 = temporary %.loc18_23.3, %Test.call.loc18
// CHECK:STDOUT:   %Test.ref.loc19: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc19: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc19_18: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_1.loc19_21: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int.left_shift.loc19: init Core.IntLiteral = call %LeftShift.ref.loc19(%int_1.loc19_18, %int_1.loc19_21) [template = constants.%int_2]
// CHECK:STDOUT:   %.loc19_23.1: Core.IntLiteral = value_of_initializer %int.left_shift.loc19 [template = constants.%int_2]
// CHECK:STDOUT:   %.loc19_23.2: Core.IntLiteral = converted %int.left_shift.loc19, %.loc19_23.1 [template = constants.%int_2]
// CHECK:STDOUT:   %Test.specific_fn.loc19: <specific function> = specific_function %Test.ref.loc19, @Test(constants.%int_2) [template = constants.%Test.specific_fn.3]
// CHECK:STDOUT:   %.loc19_23.3: ref %Expect.4 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc19: init %Expect.4 = call %Test.specific_fn.loc19() to %.loc19_23.3
// CHECK:STDOUT:   %Expect.ref.loc19: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_2.loc19: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %Expect.loc19: type = class_type @Expect, @Expect(constants.%int_2) [template = constants.%Expect.4]
// CHECK:STDOUT:   %.loc19_23.4: ref %Expect.4 = temporary %.loc19_23.3, %Test.call.loc19
// CHECK:STDOUT:   %Test.ref.loc20: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc20: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_2.loc20: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int.left_shift.loc20: init Core.IntLiteral = call %LeftShift.ref.loc20(%int_2.loc20, %int_1.loc20) [template = constants.%int_4]
// CHECK:STDOUT:   %.loc20_23.1: Core.IntLiteral = value_of_initializer %int.left_shift.loc20 [template = constants.%int_4]
// CHECK:STDOUT:   %.loc20_23.2: Core.IntLiteral = converted %int.left_shift.loc20, %.loc20_23.1 [template = constants.%int_4]
// CHECK:STDOUT:   %Test.specific_fn.loc20: <specific function> = specific_function %Test.ref.loc20, @Test(constants.%int_4) [template = constants.%Test.specific_fn.4]
// CHECK:STDOUT:   %.loc20_23.3: ref %Expect.5 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc20: init %Expect.5 = call %Test.specific_fn.loc20() to %.loc20_23.3
// CHECK:STDOUT:   %Expect.ref.loc20: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_4.loc20: Core.IntLiteral = int_value 4 [template = constants.%int_4]
// CHECK:STDOUT:   %Expect.loc20: type = class_type @Expect, @Expect(constants.%int_4) [template = constants.%Expect.5]
// CHECK:STDOUT:   %.loc20_23.4: ref %Expect.5 = temporary %.loc20_23.3, %Test.call.loc20
// CHECK:STDOUT:   %Test.ref.loc21: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc21: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_2.loc21: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %int.left_shift.loc21: init Core.IntLiteral = call %LeftShift.ref.loc21(%int_1.loc21, %int_2.loc21) [template = constants.%int_4]
// CHECK:STDOUT:   %.loc21_23.1: Core.IntLiteral = value_of_initializer %int.left_shift.loc21 [template = constants.%int_4]
// CHECK:STDOUT:   %.loc21_23.2: Core.IntLiteral = converted %int.left_shift.loc21, %.loc21_23.1 [template = constants.%int_4]
// CHECK:STDOUT:   %Test.specific_fn.loc21: <specific function> = specific_function %Test.ref.loc21, @Test(constants.%int_4) [template = constants.%Test.specific_fn.4]
// CHECK:STDOUT:   %.loc21_23.3: ref %Expect.5 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc21: init %Expect.5 = call %Test.specific_fn.loc21() to %.loc21_23.3
// CHECK:STDOUT:   %Expect.ref.loc21: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_4.loc21: Core.IntLiteral = int_value 4 [template = constants.%int_4]
// CHECK:STDOUT:   %Expect.loc21: type = class_type @Expect, @Expect(constants.%int_4) [template = constants.%Expect.5]
// CHECK:STDOUT:   %.loc21_23.4: ref %Expect.5 = temporary %.loc21_23.3, %Test.call.loc21
// CHECK:STDOUT:   %Test.ref.loc22: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc22: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_3.loc22: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %int_2.loc22: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %int.left_shift.loc22: init Core.IntLiteral = call %LeftShift.ref.loc22(%int_3.loc22, %int_2.loc22) [template = constants.%int_12]
// CHECK:STDOUT:   %.loc22_23.1: Core.IntLiteral = value_of_initializer %int.left_shift.loc22 [template = constants.%int_12]
// CHECK:STDOUT:   %.loc22_23.2: Core.IntLiteral = converted %int.left_shift.loc22, %.loc22_23.1 [template = constants.%int_12]
// CHECK:STDOUT:   %Test.specific_fn.loc22: <specific function> = specific_function %Test.ref.loc22, @Test(constants.%int_12) [template = constants.%Test.specific_fn.5]
// CHECK:STDOUT:   %.loc22_23.3: ref %Expect.6 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc22: init %Expect.6 = call %Test.specific_fn.loc22() to %.loc22_23.3
// CHECK:STDOUT:   %Expect.ref.loc22: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [template = constants.%int_12]
// CHECK:STDOUT:   %Expect.loc22: type = class_type @Expect, @Expect(constants.%int_12) [template = constants.%Expect.6]
// CHECK:STDOUT:   %.loc22_23.4: ref %Expect.6 = temporary %.loc22_23.3, %Test.call.loc22
// CHECK:STDOUT:   %Test.ref.loc23: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc23: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc23: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_30.loc23: Core.IntLiteral = int_value 30 [template = constants.%int_30]
// CHECK:STDOUT:   %int.left_shift.loc23: init Core.IntLiteral = call %LeftShift.ref.loc23(%int_1.loc23, %int_30.loc23) [template = constants.%int_1073741824]
// CHECK:STDOUT:   %.loc23_24.1: Core.IntLiteral = value_of_initializer %int.left_shift.loc23 [template = constants.%int_1073741824]
// CHECK:STDOUT:   %.loc23_24.2: Core.IntLiteral = converted %int.left_shift.loc23, %.loc23_24.1 [template = constants.%int_1073741824]
// CHECK:STDOUT:   %Test.specific_fn.loc23: <specific function> = specific_function %Test.ref.loc23, @Test(constants.%int_1073741824) [template = constants.%Test.specific_fn.6]
// CHECK:STDOUT:   %.loc23_24.3: ref %Expect.7 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc23: init %Expect.7 = call %Test.specific_fn.loc23() to %.loc23_24.3
// CHECK:STDOUT:   %Expect.ref.loc23: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_1073741824: Core.IntLiteral = int_value 1073741824 [template = constants.%int_1073741824]
// CHECK:STDOUT:   %Expect.loc23: type = class_type @Expect, @Expect(constants.%int_1073741824) [template = constants.%Expect.7]
// CHECK:STDOUT:   %.loc23_24.4: ref %Expect.7 = temporary %.loc23_24.3, %Test.call.loc23
// CHECK:STDOUT:   %Test.ref.loc24: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc24: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5]
// CHECK:STDOUT:   %int_2.loc24: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %int.left_shift.loc24: init Core.IntLiteral = call %LeftShift.ref.loc24(%int_5, %int_2.loc24) [template = constants.%int_20]
// CHECK:STDOUT:   %.loc24_23.1: Core.IntLiteral = value_of_initializer %int.left_shift.loc24 [template = constants.%int_20]
// CHECK:STDOUT:   %.loc24_23.2: Core.IntLiteral = converted %int.left_shift.loc24, %.loc24_23.1 [template = constants.%int_20]
// CHECK:STDOUT:   %Test.specific_fn.loc24: <specific function> = specific_function %Test.ref.loc24, @Test(constants.%int_20) [template = constants.%Test.specific_fn.7]
// CHECK:STDOUT:   %.loc24_23.3: ref %Expect.8 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc24: init %Expect.8 = call %Test.specific_fn.loc24() to %.loc24_23.3
// CHECK:STDOUT:   %Expect.ref.loc24: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [template = constants.%int_20]
// CHECK:STDOUT:   %Expect.loc24: type = class_type @Expect, @Expect(constants.%int_20) [template = constants.%Expect.8]
// CHECK:STDOUT:   %.loc24_23.4: ref %Expect.8 = temporary %.loc24_23.3, %Test.call.loc24
// CHECK:STDOUT:   %Test.ref.loc27: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc27: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc27_19: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc27_18: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc27_18: <bound method> = bound_method %int_1.loc27_19, %impl.elem0.loc27_18 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc27_18: init Core.IntLiteral = call %Op.bound.loc27_18(%int_1.loc27_19) [template = constants.%int_-1]
// CHECK:STDOUT:   %int_0.loc27: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc27_18.1: Core.IntLiteral = value_of_initializer %int.snegate.loc27_18 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc27_18.2: Core.IntLiteral = converted %int.snegate.loc27_18, %.loc27_18.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.left_shift.loc27: init Core.IntLiteral = call %LeftShift.ref.loc27(%.loc27_18.2, %int_0.loc27) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc27_24.1: Core.IntLiteral = value_of_initializer %int.left_shift.loc27 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc27_24.2: Core.IntLiteral = converted %int.left_shift.loc27, %.loc27_24.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %Test.specific_fn.loc27: <specific function> = specific_function %Test.ref.loc27, @Test(constants.%int_-1) [template = constants.%Test.specific_fn.8]
// CHECK:STDOUT:   %.loc27_24.3: ref %Expect.9 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc27: init %Expect.9 = call %Test.specific_fn.loc27() to %.loc27_24.3
// CHECK:STDOUT:   %Expect.ref.loc27: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_1.loc27_37: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc27_36: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc27_36: <bound method> = bound_method %int_1.loc27_37, %impl.elem0.loc27_36 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc27_36: init Core.IntLiteral = call %Op.bound.loc27_36(%int_1.loc27_37) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc27_38.1: Core.IntLiteral = value_of_initializer %int.snegate.loc27_36 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc27_38.2: Core.IntLiteral = converted %int.snegate.loc27_36, %.loc27_38.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %Expect.loc27: type = class_type @Expect, @Expect(constants.%int_-1) [template = constants.%Expect.9]
// CHECK:STDOUT:   %.loc27_24.4: ref %Expect.9 = temporary %.loc27_24.3, %Test.call.loc27
// CHECK:STDOUT:   %Test.ref.loc28: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc28: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc28_19: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc28_18: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc28_18: <bound method> = bound_method %int_1.loc28_19, %impl.elem0.loc28_18 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc28_18: init Core.IntLiteral = call %Op.bound.loc28_18(%int_1.loc28_19) [template = constants.%int_-1]
// CHECK:STDOUT:   %int_1.loc28_22: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc28_18.1: Core.IntLiteral = value_of_initializer %int.snegate.loc28_18 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc28_18.2: Core.IntLiteral = converted %int.snegate.loc28_18, %.loc28_18.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.left_shift.loc28: init Core.IntLiteral = call %LeftShift.ref.loc28(%.loc28_18.2, %int_1.loc28_22) [template = constants.%int_-2]
// CHECK:STDOUT:   %.loc28_24.1: Core.IntLiteral = value_of_initializer %int.left_shift.loc28 [template = constants.%int_-2]
// CHECK:STDOUT:   %.loc28_24.2: Core.IntLiteral = converted %int.left_shift.loc28, %.loc28_24.1 [template = constants.%int_-2]
// CHECK:STDOUT:   %Test.specific_fn.loc28: <specific function> = specific_function %Test.ref.loc28, @Test(constants.%int_-2) [template = constants.%Test.specific_fn.9]
// CHECK:STDOUT:   %.loc28_24.3: ref %Expect.10 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc28: init %Expect.10 = call %Test.specific_fn.loc28() to %.loc28_24.3
// CHECK:STDOUT:   %Expect.ref.loc28: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_2.loc28: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %impl.elem0.loc28_36: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc28_36: <bound method> = bound_method %int_2.loc28, %impl.elem0.loc28_36 [template = constants.%Op.bound.2]
// CHECK:STDOUT:   %int.snegate.loc28_36: init Core.IntLiteral = call %Op.bound.loc28_36(%int_2.loc28) [template = constants.%int_-2]
// CHECK:STDOUT:   %.loc28_38.1: Core.IntLiteral = value_of_initializer %int.snegate.loc28_36 [template = constants.%int_-2]
// CHECK:STDOUT:   %.loc28_38.2: Core.IntLiteral = converted %int.snegate.loc28_36, %.loc28_38.1 [template = constants.%int_-2]
// CHECK:STDOUT:   %Expect.loc28: type = class_type @Expect, @Expect(constants.%int_-2) [template = constants.%Expect.10]
// CHECK:STDOUT:   %.loc28_24.4: ref %Expect.10 = temporary %.loc28_24.3, %Test.call.loc28
// CHECK:STDOUT:   %Test.ref.loc29: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc29: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_2.loc29: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %impl.elem0.loc29_18: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc29_18: <bound method> = bound_method %int_2.loc29, %impl.elem0.loc29_18 [template = constants.%Op.bound.2]
// CHECK:STDOUT:   %int.snegate.loc29_18: init Core.IntLiteral = call %Op.bound.loc29_18(%int_2.loc29) [template = constants.%int_-2]
// CHECK:STDOUT:   %int_1.loc29: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc29_18.1: Core.IntLiteral = value_of_initializer %int.snegate.loc29_18 [template = constants.%int_-2]
// CHECK:STDOUT:   %.loc29_18.2: Core.IntLiteral = converted %int.snegate.loc29_18, %.loc29_18.1 [template = constants.%int_-2]
// CHECK:STDOUT:   %int.left_shift.loc29: init Core.IntLiteral = call %LeftShift.ref.loc29(%.loc29_18.2, %int_1.loc29) [template = constants.%int_-4]
// CHECK:STDOUT:   %.loc29_24.1: Core.IntLiteral = value_of_initializer %int.left_shift.loc29 [template = constants.%int_-4]
// CHECK:STDOUT:   %.loc29_24.2: Core.IntLiteral = converted %int.left_shift.loc29, %.loc29_24.1 [template = constants.%int_-4]
// CHECK:STDOUT:   %Test.specific_fn.loc29: <specific function> = specific_function %Test.ref.loc29, @Test(constants.%int_-4) [template = constants.%Test.specific_fn.10]
// CHECK:STDOUT:   %.loc29_24.3: ref %Expect.11 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc29: init %Expect.11 = call %Test.specific_fn.loc29() to %.loc29_24.3
// CHECK:STDOUT:   %Expect.ref.loc29: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_4.loc29: Core.IntLiteral = int_value 4 [template = constants.%int_4]
// CHECK:STDOUT:   %impl.elem0.loc29_36: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc29_36: <bound method> = bound_method %int_4.loc29, %impl.elem0.loc29_36 [template = constants.%Op.bound.3]
// CHECK:STDOUT:   %int.snegate.loc29_36: init Core.IntLiteral = call %Op.bound.loc29_36(%int_4.loc29) [template = constants.%int_-4]
// CHECK:STDOUT:   %.loc29_38.1: Core.IntLiteral = value_of_initializer %int.snegate.loc29_36 [template = constants.%int_-4]
// CHECK:STDOUT:   %.loc29_38.2: Core.IntLiteral = converted %int.snegate.loc29_36, %.loc29_38.1 [template = constants.%int_-4]
// CHECK:STDOUT:   %Expect.loc29: type = class_type @Expect, @Expect(constants.%int_-4) [template = constants.%Expect.11]
// CHECK:STDOUT:   %.loc29_24.4: ref %Expect.11 = temporary %.loc29_24.3, %Test.call.loc29
// CHECK:STDOUT:   %Test.ref.loc30: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc30: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_3.loc30: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %impl.elem0.loc30_18: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc30_18: <bound method> = bound_method %int_3.loc30, %impl.elem0.loc30_18 [template = constants.%Op.bound.4]
// CHECK:STDOUT:   %int.snegate.loc30_18: init Core.IntLiteral = call %Op.bound.loc30_18(%int_3.loc30) [template = constants.%int_-3]
// CHECK:STDOUT:   %int_1.loc30: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc30_18.1: Core.IntLiteral = value_of_initializer %int.snegate.loc30_18 [template = constants.%int_-3]
// CHECK:STDOUT:   %.loc30_18.2: Core.IntLiteral = converted %int.snegate.loc30_18, %.loc30_18.1 [template = constants.%int_-3]
// CHECK:STDOUT:   %int.left_shift.loc30: init Core.IntLiteral = call %LeftShift.ref.loc30(%.loc30_18.2, %int_1.loc30) [template = constants.%int_-6]
// CHECK:STDOUT:   %.loc30_24.1: Core.IntLiteral = value_of_initializer %int.left_shift.loc30 [template = constants.%int_-6]
// CHECK:STDOUT:   %.loc30_24.2: Core.IntLiteral = converted %int.left_shift.loc30, %.loc30_24.1 [template = constants.%int_-6]
// CHECK:STDOUT:   %Test.specific_fn.loc30: <specific function> = specific_function %Test.ref.loc30, @Test(constants.%int_-6) [template = constants.%Test.specific_fn.11]
// CHECK:STDOUT:   %.loc30_24.3: ref %Expect.12 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc30: init %Expect.12 = call %Test.specific_fn.loc30() to %.loc30_24.3
// CHECK:STDOUT:   %Expect.ref.loc30: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [template = constants.%int_6]
// CHECK:STDOUT:   %impl.elem0.loc30_36: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc30_36: <bound method> = bound_method %int_6, %impl.elem0.loc30_36 [template = constants.%Op.bound.5]
// CHECK:STDOUT:   %int.snegate.loc30_36: init Core.IntLiteral = call %Op.bound.loc30_36(%int_6) [template = constants.%int_-6]
// CHECK:STDOUT:   %.loc30_38.1: Core.IntLiteral = value_of_initializer %int.snegate.loc30_36 [template = constants.%int_-6]
// CHECK:STDOUT:   %.loc30_38.2: Core.IntLiteral = converted %int.snegate.loc30_36, %.loc30_38.1 [template = constants.%int_-6]
// CHECK:STDOUT:   %Expect.loc30: type = class_type @Expect, @Expect(constants.%int_-6) [template = constants.%Expect.12]
// CHECK:STDOUT:   %.loc30_24.4: ref %Expect.12 = temporary %.loc30_24.3, %Test.call.loc30
// CHECK:STDOUT:   %Test.ref.loc33: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc33: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_4294967295: Core.IntLiteral = int_value 4294967295 [template = constants.%int_4294967295]
// CHECK:STDOUT:   %int_1.loc33: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int.left_shift.loc33: init Core.IntLiteral = call %LeftShift.ref.loc33(%int_4294967295, %int_1.loc33) [template = constants.%int_8589934590]
// CHECK:STDOUT:   %.loc33_33.1: Core.IntLiteral = value_of_initializer %int.left_shift.loc33 [template = constants.%int_8589934590]
// CHECK:STDOUT:   %.loc33_33.2: Core.IntLiteral = converted %int.left_shift.loc33, %.loc33_33.1 [template = constants.%int_8589934590]
// CHECK:STDOUT:   %Test.specific_fn.loc33: <specific function> = specific_function %Test.ref.loc33, @Test(constants.%int_8589934590) [template = constants.%Test.specific_fn.12]
// CHECK:STDOUT:   %.loc33_33.3: ref %Expect.13 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc33: init %Expect.13 = call %Test.specific_fn.loc33() to %.loc33_33.3
// CHECK:STDOUT:   %Expect.ref.loc33: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_8589934590: Core.IntLiteral = int_value 8589934590 [template = constants.%int_8589934590]
// CHECK:STDOUT:   %Expect.loc33: type = class_type @Expect, @Expect(constants.%int_8589934590) [template = constants.%Expect.13]
// CHECK:STDOUT:   %.loc33_33.4: ref %Expect.13 = temporary %.loc33_33.3, %Test.call.loc33
// CHECK:STDOUT:   %Test.ref.loc34: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc34: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_4294967294: Core.IntLiteral = int_value 4294967294 [template = constants.%int_4294967294]
// CHECK:STDOUT:   %int_1.loc34: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int.left_shift.loc34: init Core.IntLiteral = call %LeftShift.ref.loc34(%int_4294967294, %int_1.loc34) [template = constants.%int_8589934588]
// CHECK:STDOUT:   %.loc34_33.1: Core.IntLiteral = value_of_initializer %int.left_shift.loc34 [template = constants.%int_8589934588]
// CHECK:STDOUT:   %.loc34_33.2: Core.IntLiteral = converted %int.left_shift.loc34, %.loc34_33.1 [template = constants.%int_8589934588]
// CHECK:STDOUT:   %Test.specific_fn.loc34: <specific function> = specific_function %Test.ref.loc34, @Test(constants.%int_8589934588) [template = constants.%Test.specific_fn.13]
// CHECK:STDOUT:   %.loc34_33.3: ref %Expect.14 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc34: init %Expect.14 = call %Test.specific_fn.loc34() to %.loc34_33.3
// CHECK:STDOUT:   %Expect.ref.loc34: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_8589934588: Core.IntLiteral = int_value 8589934588 [template = constants.%int_8589934588]
// CHECK:STDOUT:   %Expect.loc34: type = class_type @Expect, @Expect(constants.%int_8589934588) [template = constants.%Expect.14]
// CHECK:STDOUT:   %.loc34_33.4: ref %Expect.14 = temporary %.loc34_33.3, %Test.call.loc34
// CHECK:STDOUT:   %Test.ref.loc35: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc35: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_2882400001: Core.IntLiteral = int_value 2882400001 [template = constants.%int_2882400001]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [template = constants.%int_8]
// CHECK:STDOUT:   %int.left_shift.loc35: init Core.IntLiteral = call %LeftShift.ref.loc35(%int_2882400001, %int_8) [template = constants.%int_737894400256]
// CHECK:STDOUT:   %.loc35_33.1: Core.IntLiteral = value_of_initializer %int.left_shift.loc35 [template = constants.%int_737894400256]
// CHECK:STDOUT:   %.loc35_33.2: Core.IntLiteral = converted %int.left_shift.loc35, %.loc35_33.1 [template = constants.%int_737894400256]
// CHECK:STDOUT:   %Test.specific_fn.loc35: <specific function> = specific_function %Test.ref.loc35, @Test(constants.%int_737894400256) [template = constants.%Test.specific_fn.14]
// CHECK:STDOUT:   %.loc35_33.3: ref %Expect.15 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc35: init %Expect.15 = call %Test.specific_fn.loc35() to %.loc35_33.3
// CHECK:STDOUT:   %Expect.ref.loc35: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_737894400256: Core.IntLiteral = int_value 737894400256 [template = constants.%int_737894400256]
// CHECK:STDOUT:   %Expect.loc35: type = class_type @Expect, @Expect(constants.%int_737894400256) [template = constants.%Expect.15]
// CHECK:STDOUT:   %.loc35_33.4: ref %Expect.15 = temporary %.loc35_33.3, %Test.call.loc35
// CHECK:STDOUT:   %Test.ref.loc36: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc36: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_9223372036854775807: Core.IntLiteral = int_value 9223372036854775807 [template = constants.%int_9223372036854775807]
// CHECK:STDOUT:   %int_1.loc36: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int.left_shift.loc36: init Core.IntLiteral = call %LeftShift.ref.loc36(%int_9223372036854775807, %int_1.loc36) [template = constants.%int_18446744073709551614]
// CHECK:STDOUT:   %.loc36_43.1: Core.IntLiteral = value_of_initializer %int.left_shift.loc36 [template = constants.%int_18446744073709551614]
// CHECK:STDOUT:   %.loc36_43.2: Core.IntLiteral = converted %int.left_shift.loc36, %.loc36_43.1 [template = constants.%int_18446744073709551614]
// CHECK:STDOUT:   %Test.specific_fn.loc36: <specific function> = specific_function %Test.ref.loc36, @Test(constants.%int_18446744073709551614) [template = constants.%Test.specific_fn.15]
// CHECK:STDOUT:   %.loc36_43.3: ref %Expect.16 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc36: init %Expect.16 = call %Test.specific_fn.loc36() to %.loc36_43.3
// CHECK:STDOUT:   %Expect.ref.loc36: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_18446744073709551614: Core.IntLiteral = int_value 18446744073709551614 [template = constants.%int_18446744073709551614]
// CHECK:STDOUT:   %Expect.loc36: type = class_type @Expect, @Expect(constants.%int_18446744073709551614) [template = constants.%Expect.16]
// CHECK:STDOUT:   %.loc36_43.4: ref %Expect.16 = temporary %.loc36_43.3, %Test.call.loc36
// CHECK:STDOUT:   %Test.ref.loc37: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc37: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_18446744073709551615: Core.IntLiteral = int_value 18446744073709551615 [template = constants.%int_18446744073709551615]
// CHECK:STDOUT:   %int_1.loc37: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int.left_shift.loc37: init Core.IntLiteral = call %LeftShift.ref.loc37(%int_18446744073709551615, %int_1.loc37) [template = constants.%int_36893488147419103230]
// CHECK:STDOUT:   %.loc37_43.1: Core.IntLiteral = value_of_initializer %int.left_shift.loc37 [template = constants.%int_36893488147419103230]
// CHECK:STDOUT:   %.loc37_43.2: Core.IntLiteral = converted %int.left_shift.loc37, %.loc37_43.1 [template = constants.%int_36893488147419103230]
// CHECK:STDOUT:   %Test.specific_fn.loc37: <specific function> = specific_function %Test.ref.loc37, @Test(constants.%int_36893488147419103230) [template = constants.%Test.specific_fn.16]
// CHECK:STDOUT:   %.loc37_43.3: ref %Expect.17 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc37: init %Expect.17 = call %Test.specific_fn.loc37() to %.loc37_43.3
// CHECK:STDOUT:   %Expect.ref.loc37: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_36893488147419103230: Core.IntLiteral = int_value 36893488147419103230 [template = constants.%int_36893488147419103230]
// CHECK:STDOUT:   %Expect.loc37: type = class_type @Expect, @Expect(constants.%int_36893488147419103230) [template = constants.%Expect.17]
// CHECK:STDOUT:   %.loc37_43.4: ref %Expect.17 = temporary %.loc37_43.3, %Test.call.loc37
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%N) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(@Test.%N.loc7_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%N) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%N
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_0) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_0
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_0
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_0) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_0
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_1) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_1
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_1
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_1) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_1
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_2) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_2
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_2
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_2) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_2
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_4) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_4
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_4
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_4) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_4
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_12) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_12
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_12
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_12) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_12
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_12
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_1073741824) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_1073741824
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_1073741824
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_1073741824) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_1073741824
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_1073741824
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_20) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_20
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_20
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_20) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_20
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_20
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_-1) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_-1
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_-1
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-1) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_-1
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_-1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_-2) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_-2
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_-2
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.10
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-2) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_-2
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_-2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_-4) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_-4
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_-4
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.11
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-4) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_-4
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_-4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_-6) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_-6
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_-6
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.12
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-6) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_-6
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_-6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_8589934590) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_8589934590
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_8589934590
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.13
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_8589934590) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_8589934590
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_8589934590
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_8589934588) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_8589934588
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_8589934588
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.14
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_8589934588) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_8589934588
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_8589934588
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_737894400256) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_737894400256
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_737894400256
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.15
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_737894400256) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_737894400256
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_737894400256
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_18446744073709551614) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_18446744073709551614
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_18446744073709551614
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.16
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_18446744073709551614) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_18446744073709551614
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_18446744073709551614
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_36893488147419103230) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_36893488147419103230
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_36893488147419103230
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.17
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_36893488147419103230) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_36893488147419103230
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_36893488147419103230
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32.1) [template]
// CHECK:STDOUT:   %LeftShift.type.1: type = fn_type @LeftShift.1 [template]
// CHECK:STDOUT:   %LeftShift: %LeftShift.type.1 = struct_value () [template]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %LeftShiftLit.type: type = fn_type @LeftShiftLit [template]
// CHECK:STDOUT:   %LeftShiftLit: %LeftShiftLit.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_31.1: Core.IntLiteral = int_value 31 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32.1) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.19: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32.1) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_31.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32.1) [template]
// CHECK:STDOUT:   %int_31.2: %i32 = int_value 31 [template]
// CHECK:STDOUT:   %int_-2147483648: %i32 = int_value -2147483648 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_32.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32.1) [template]
// CHECK:STDOUT:   %int_32.2: %i32 = int_value 32 [template]
// CHECK:STDOUT:   %int_33.1: Core.IntLiteral = int_value 33 [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_33.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.2(%int_32.1) [template]
// CHECK:STDOUT:   %int_33.2: %i32 = int_value 33 [template]
// CHECK:STDOUT:   %int_1000.1: Core.IntLiteral = int_value 1000 [template]
// CHECK:STDOUT:   %Convert.bound.5: <bound method> = bound_method %int_1000.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.5: <specific function> = specific_function %Convert.bound.5, @Convert.2(%int_32.1) [template]
// CHECK:STDOUT:   %int_1000.2: %i32 = int_value 1000 [template]
// CHECK:STDOUT:   %int_0.1: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %int_0.2: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.bound.6: <bound method> = bound_method %int_0.2, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.6: <specific function> = specific_function %Convert.bound.6, @Convert.2(%int_32.1) [template]
// CHECK:STDOUT:   %Op.type.13: type = fn_type @Op.13 [template]
// CHECK:STDOUT:   %Op.type.14: type = fn_type @Op.14 [template]
// CHECK:STDOUT:   %Op.14: %Op.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.20: <witness> = interface_witness (%Op.14) [template]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %int_1.1, %Op.14 [template]
// CHECK:STDOUT:   %int_-1.1: Core.IntLiteral = int_value -1 [template]
// CHECK:STDOUT:   %Convert.bound.7: <bound method> = bound_method %int_-1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.7: <specific function> = specific_function %Convert.bound.7, @Convert.2(%int_32.1) [template]
// CHECK:STDOUT:   %int_-1.2: %i32 = int_value -1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .IntLiteral = %import_ref.5
// CHECK:STDOUT:     .ImplicitAs = %import_ref.6
// CHECK:STDOUT:     .Negate = %import_ref.230
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .LeftShift = %LeftShift.decl
// CHECK:STDOUT:     .LeftShiftLit = %LeftShiftLit.decl
// CHECK:STDOUT:     .size_1 = @__global_init.%size_1
// CHECK:STDOUT:     .size_2 = @__global_init.%size_2
// CHECK:STDOUT:     .size_3 = @__global_init.%size_3
// CHECK:STDOUT:     .overflow_1 = @__global_init.%overflow_1
// CHECK:STDOUT:     .overflow_2 = @__global_init.%overflow_2
// CHECK:STDOUT:     .no_overflow_1 = @__global_init.%no_overflow_1
// CHECK:STDOUT:     .no_overflow_2 = @__global_init.%no_overflow_2
// CHECK:STDOUT:     .negative = @__global_init.%negative
// CHECK:STDOUT:     .negative_zero = @__global_init.%negative_zero
// CHECK:STDOUT:     .negative_lit = @__global_init.%negative_lit
// CHECK:STDOUT:     .negative_lit_zero = @__global_init.%negative_lit_zero
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %LeftShift.decl: %LeftShift.type.1 = fn_decl @LeftShift.1 [template = constants.%LeftShift] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc4_33: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:     %i32.loc4_33: type = class_type @Int, @Int(constants.%int_32.1) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc4_17: type = splice_block %i32.loc4_17 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc4_17: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:       %i32.loc4_17: type = class_type @Int, @Int(constants.%int_32.1) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc4_25: type = splice_block %i32.loc4_25 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc4_25: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:       %i32.loc4_25: type = class_type @Int, @Int(constants.%int_32.1) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %LeftShiftLit.decl: %LeftShiftLit.type = fn_decl @LeftShiftLit [template = constants.%LeftShiftLit] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc5_50: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc5_54: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.5 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc5_66: init type = call %IntLiteral.ref.loc5_54() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc5_66.1: type = value_of_initializer %int_literal.make_type.loc5_66 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc5_66.2: type = converted %int_literal.make_type.loc5_66, %.loc5_66.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc5_36.3: type = splice_block %.loc5_36.2 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc5_20: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc5_24: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.5 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc5_36: init type = call %IntLiteral.ref.loc5_24() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc5_36.1: type = value_of_initializer %int_literal.make_type.loc5_36 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc5_36.2: type = converted %int_literal.make_type.loc5_36, %.loc5_36.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc5_42: type = splice_block %i32 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32.1) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param2
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LeftShift.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.left_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LeftShiftLit(%a.param_patt: Core.IntLiteral, %b.param_patt: %i32) -> Core.IntLiteral = "int.left_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %LeftShift.ref.loc8: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_31.loc8: Core.IntLiteral = int_value 31 [template = constants.%int_31.1]
// CHECK:STDOUT:   %impl.elem0.loc8_29: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc8_29: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8_29 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc8_29: <specific function> = specific_function %Convert.bound.loc8_29, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc8_29: init %i32 = call %Convert.specific_fn.loc8_29(%int_1.loc8) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc8_29.1: %i32 = value_of_initializer %int.convert_checked.loc8_29 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc8_29.2: %i32 = converted %int_1.loc8, %.loc8_29.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc8_32: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc8_32: <bound method> = bound_method %int_31.loc8, %impl.elem0.loc8_32 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc8_32: <specific function> = specific_function %Convert.bound.loc8_32, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc8_32: init %i32 = call %Convert.specific_fn.loc8_32(%int_31.loc8) [template = constants.%int_31.2]
// CHECK:STDOUT:   %.loc8_32.1: %i32 = value_of_initializer %int.convert_checked.loc8_32 [template = constants.%int_31.2]
// CHECK:STDOUT:   %.loc8_32.2: %i32 = converted %int_31.loc8, %.loc8_32.1 [template = constants.%int_31.2]
// CHECK:STDOUT:   %int.left_shift.loc8: init %i32 = call %LeftShift.ref.loc8(%.loc8_29.2, %.loc8_32.2) [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %.loc8_35.1: %i32 = value_of_initializer %int.left_shift.loc8 [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %.loc8_35.2: %i32 = converted %int.left_shift.loc8, %.loc8_35.1 [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %size_1: %i32 = bind_name size_1, %.loc8_35.2
// CHECK:STDOUT:   %LeftShift.ref.loc13: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_32.loc13: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:   %impl.elem0.loc13_29: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc13_29: <bound method> = bound_method %int_1.loc13, %impl.elem0.loc13_29 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_29: <specific function> = specific_function %Convert.bound.loc13_29, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc13_29: init %i32 = call %Convert.specific_fn.loc13_29(%int_1.loc13) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_29.1: %i32 = value_of_initializer %int.convert_checked.loc13_29 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_29.2: %i32 = converted %int_1.loc13, %.loc13_29.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc13_32: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc13_32: <bound method> = bound_method %int_32.loc13, %impl.elem0.loc13_32 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_32: <specific function> = specific_function %Convert.bound.loc13_32, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc13_32: init %i32 = call %Convert.specific_fn.loc13_32(%int_32.loc13) [template = constants.%int_32.2]
// CHECK:STDOUT:   %.loc13_32.1: %i32 = value_of_initializer %int.convert_checked.loc13_32 [template = constants.%int_32.2]
// CHECK:STDOUT:   %.loc13_32.2: %i32 = converted %int_32.loc13, %.loc13_32.1 [template = constants.%int_32.2]
// CHECK:STDOUT:   %int.left_shift.loc13: init %i32 = call %LeftShift.ref.loc13(%.loc13_29.2, %.loc13_32.2) [template = <error>]
// CHECK:STDOUT:   %.loc13_35.1: %i32 = value_of_initializer %int.left_shift.loc13 [template = <error>]
// CHECK:STDOUT:   %.loc13_35.2: %i32 = converted %int.left_shift.loc13, %.loc13_35.1 [template = <error>]
// CHECK:STDOUT:   %size_2: %i32 = bind_name size_2, %.loc13_35.2
// CHECK:STDOUT:   %LeftShift.ref.loc18: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_33: Core.IntLiteral = int_value 33 [template = constants.%int_33.1]
// CHECK:STDOUT:   %impl.elem0.loc18_29: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc18_29: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18_29 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc18_29: <specific function> = specific_function %Convert.bound.loc18_29, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc18_29: init %i32 = call %Convert.specific_fn.loc18_29(%int_1.loc18) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc18_29.1: %i32 = value_of_initializer %int.convert_checked.loc18_29 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc18_29.2: %i32 = converted %int_1.loc18, %.loc18_29.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc18_32: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc18_32: <bound method> = bound_method %int_33, %impl.elem0.loc18_32 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc18_32: <specific function> = specific_function %Convert.bound.loc18_32, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc18_32: init %i32 = call %Convert.specific_fn.loc18_32(%int_33) [template = constants.%int_33.2]
// CHECK:STDOUT:   %.loc18_32.1: %i32 = value_of_initializer %int.convert_checked.loc18_32 [template = constants.%int_33.2]
// CHECK:STDOUT:   %.loc18_32.2: %i32 = converted %int_33, %.loc18_32.1 [template = constants.%int_33.2]
// CHECK:STDOUT:   %int.left_shift.loc18: init %i32 = call %LeftShift.ref.loc18(%.loc18_29.2, %.loc18_32.2) [template = <error>]
// CHECK:STDOUT:   %.loc18_35.1: %i32 = value_of_initializer %int.left_shift.loc18 [template = <error>]
// CHECK:STDOUT:   %.loc18_35.2: %i32 = converted %int.left_shift.loc18, %.loc18_35.1 [template = <error>]
// CHECK:STDOUT:   %size_3: %i32 = bind_name size_3, %.loc18_35.2
// CHECK:STDOUT:   %LeftShift.ref.loc21: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1000.loc21: Core.IntLiteral = int_value 1000 [template = constants.%int_1000.1]
// CHECK:STDOUT:   %int_31.loc21: Core.IntLiteral = int_value 31 [template = constants.%int_31.1]
// CHECK:STDOUT:   %impl.elem0.loc21_33: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc21_33: <bound method> = bound_method %int_1000.loc21, %impl.elem0.loc21_33 [template = constants.%Convert.bound.5]
// CHECK:STDOUT:   %Convert.specific_fn.loc21_33: <specific function> = specific_function %Convert.bound.loc21_33, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %int.convert_checked.loc21_33: init %i32 = call %Convert.specific_fn.loc21_33(%int_1000.loc21) [template = constants.%int_1000.2]
// CHECK:STDOUT:   %.loc21_33.1: %i32 = value_of_initializer %int.convert_checked.loc21_33 [template = constants.%int_1000.2]
// CHECK:STDOUT:   %.loc21_33.2: %i32 = converted %int_1000.loc21, %.loc21_33.1 [template = constants.%int_1000.2]
// CHECK:STDOUT:   %impl.elem0.loc21_39: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc21_39: <bound method> = bound_method %int_31.loc21, %impl.elem0.loc21_39 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc21_39: <specific function> = specific_function %Convert.bound.loc21_39, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc21_39: init %i32 = call %Convert.specific_fn.loc21_39(%int_31.loc21) [template = constants.%int_31.2]
// CHECK:STDOUT:   %.loc21_39.1: %i32 = value_of_initializer %int.convert_checked.loc21_39 [template = constants.%int_31.2]
// CHECK:STDOUT:   %.loc21_39.2: %i32 = converted %int_31.loc21, %.loc21_39.1 [template = constants.%int_31.2]
// CHECK:STDOUT:   %int.left_shift.loc21: init %i32 = call %LeftShift.ref.loc21(%.loc21_33.2, %.loc21_39.2) [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc21_42.1: %i32 = value_of_initializer %int.left_shift.loc21 [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc21_42.2: %i32 = converted %int.left_shift.loc21, %.loc21_42.1 [template = constants.%int_0.1]
// CHECK:STDOUT:   %overflow_1: %i32 = bind_name overflow_1, %.loc21_42.2
// CHECK:STDOUT:   %LeftShift.ref.loc26: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1000.loc26: Core.IntLiteral = int_value 1000 [template = constants.%int_1000.1]
// CHECK:STDOUT:   %int_32.loc26: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:   %impl.elem0.loc26_33: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc26_33: <bound method> = bound_method %int_1000.loc26, %impl.elem0.loc26_33 [template = constants.%Convert.bound.5]
// CHECK:STDOUT:   %Convert.specific_fn.loc26_33: <specific function> = specific_function %Convert.bound.loc26_33, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %int.convert_checked.loc26_33: init %i32 = call %Convert.specific_fn.loc26_33(%int_1000.loc26) [template = constants.%int_1000.2]
// CHECK:STDOUT:   %.loc26_33.1: %i32 = value_of_initializer %int.convert_checked.loc26_33 [template = constants.%int_1000.2]
// CHECK:STDOUT:   %.loc26_33.2: %i32 = converted %int_1000.loc26, %.loc26_33.1 [template = constants.%int_1000.2]
// CHECK:STDOUT:   %impl.elem0.loc26_39: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc26_39: <bound method> = bound_method %int_32.loc26, %impl.elem0.loc26_39 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc26_39: <specific function> = specific_function %Convert.bound.loc26_39, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc26_39: init %i32 = call %Convert.specific_fn.loc26_39(%int_32.loc26) [template = constants.%int_32.2]
// CHECK:STDOUT:   %.loc26_39.1: %i32 = value_of_initializer %int.convert_checked.loc26_39 [template = constants.%int_32.2]
// CHECK:STDOUT:   %.loc26_39.2: %i32 = converted %int_32.loc26, %.loc26_39.1 [template = constants.%int_32.2]
// CHECK:STDOUT:   %int.left_shift.loc26: init %i32 = call %LeftShift.ref.loc26(%.loc26_33.2, %.loc26_39.2) [template = <error>]
// CHECK:STDOUT:   %.loc26_42.1: %i32 = value_of_initializer %int.left_shift.loc26 [template = <error>]
// CHECK:STDOUT:   %.loc26_42.2: %i32 = converted %int.left_shift.loc26, %.loc26_42.1 [template = <error>]
// CHECK:STDOUT:   %overflow_2: %i32 = bind_name overflow_2, %.loc26_42.2
// CHECK:STDOUT:   %LeftShift.ref.loc29: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_0.loc29: Core.IntLiteral = int_value 0 [template = constants.%int_0.2]
// CHECK:STDOUT:   %int_31.loc29: Core.IntLiteral = int_value 31 [template = constants.%int_31.1]
// CHECK:STDOUT:   %impl.elem0.loc29_36: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc29_36: <bound method> = bound_method %int_0.loc29, %impl.elem0.loc29_36 [template = constants.%Convert.bound.6]
// CHECK:STDOUT:   %Convert.specific_fn.loc29_36: <specific function> = specific_function %Convert.bound.loc29_36, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.6]
// CHECK:STDOUT:   %int.convert_checked.loc29_36: init %i32 = call %Convert.specific_fn.loc29_36(%int_0.loc29) [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc29_36.1: %i32 = value_of_initializer %int.convert_checked.loc29_36 [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc29_36.2: %i32 = converted %int_0.loc29, %.loc29_36.1 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc29_39: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc29_39: <bound method> = bound_method %int_31.loc29, %impl.elem0.loc29_39 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc29_39: <specific function> = specific_function %Convert.bound.loc29_39, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc29_39: init %i32 = call %Convert.specific_fn.loc29_39(%int_31.loc29) [template = constants.%int_31.2]
// CHECK:STDOUT:   %.loc29_39.1: %i32 = value_of_initializer %int.convert_checked.loc29_39 [template = constants.%int_31.2]
// CHECK:STDOUT:   %.loc29_39.2: %i32 = converted %int_31.loc29, %.loc29_39.1 [template = constants.%int_31.2]
// CHECK:STDOUT:   %int.left_shift.loc29: init %i32 = call %LeftShift.ref.loc29(%.loc29_36.2, %.loc29_39.2) [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc29_42.1: %i32 = value_of_initializer %int.left_shift.loc29 [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc29_42.2: %i32 = converted %int.left_shift.loc29, %.loc29_42.1 [template = constants.%int_0.1]
// CHECK:STDOUT:   %no_overflow_1: %i32 = bind_name no_overflow_1, %.loc29_42.2
// CHECK:STDOUT:   %LeftShift.ref.loc34: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_0.loc34: Core.IntLiteral = int_value 0 [template = constants.%int_0.2]
// CHECK:STDOUT:   %int_32.loc34: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:   %impl.elem0.loc34_36: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc34_36: <bound method> = bound_method %int_0.loc34, %impl.elem0.loc34_36 [template = constants.%Convert.bound.6]
// CHECK:STDOUT:   %Convert.specific_fn.loc34_36: <specific function> = specific_function %Convert.bound.loc34_36, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.6]
// CHECK:STDOUT:   %int.convert_checked.loc34_36: init %i32 = call %Convert.specific_fn.loc34_36(%int_0.loc34) [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc34_36.1: %i32 = value_of_initializer %int.convert_checked.loc34_36 [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc34_36.2: %i32 = converted %int_0.loc34, %.loc34_36.1 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc34_39: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc34_39: <bound method> = bound_method %int_32.loc34, %impl.elem0.loc34_39 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc34_39: <specific function> = specific_function %Convert.bound.loc34_39, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc34_39: init %i32 = call %Convert.specific_fn.loc34_39(%int_32.loc34) [template = constants.%int_32.2]
// CHECK:STDOUT:   %.loc34_39.1: %i32 = value_of_initializer %int.convert_checked.loc34_39 [template = constants.%int_32.2]
// CHECK:STDOUT:   %.loc34_39.2: %i32 = converted %int_32.loc34, %.loc34_39.1 [template = constants.%int_32.2]
// CHECK:STDOUT:   %int.left_shift.loc34: init %i32 = call %LeftShift.ref.loc34(%.loc34_36.2, %.loc34_39.2) [template = <error>]
// CHECK:STDOUT:   %.loc34_42.1: %i32 = value_of_initializer %int.left_shift.loc34 [template = <error>]
// CHECK:STDOUT:   %.loc34_42.2: %i32 = converted %int.left_shift.loc34, %.loc34_42.1 [template = <error>]
// CHECK:STDOUT:   %no_overflow_2: %i32 = bind_name no_overflow_2, %.loc34_42.2
// CHECK:STDOUT:   %LeftShift.ref.loc41: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc41_31: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_1.loc41_35: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc41_34.1: %Op.type.13 = interface_witness_access constants.%interface.20, element0 [template = constants.%Op.14]
// CHECK:STDOUT:   %Op.bound.loc41: <bound method> = bound_method %int_1.loc41_35, %impl.elem0.loc41_34.1 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc41: init Core.IntLiteral = call %Op.bound.loc41(%int_1.loc41_35) [template = constants.%int_-1.1]
// CHECK:STDOUT:   %impl.elem0.loc41_31: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc41_31: <bound method> = bound_method %int_1.loc41_31, %impl.elem0.loc41_31 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc41_31: <specific function> = specific_function %Convert.bound.loc41_31, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc41_31: init %i32 = call %Convert.specific_fn.loc41_31(%int_1.loc41_31) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc41_31.1: %i32 = value_of_initializer %int.convert_checked.loc41_31 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc41_31.2: %i32 = converted %int_1.loc41_31, %.loc41_31.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc41_34.2: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc41_34: <bound method> = bound_method %int.snegate.loc41, %impl.elem0.loc41_34.2 [template = constants.%Convert.bound.7]
// CHECK:STDOUT:   %Convert.specific_fn.loc41_34: <specific function> = specific_function %Convert.bound.loc41_34, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.7]
// CHECK:STDOUT:   %.loc41_34.1: Core.IntLiteral = value_of_initializer %int.snegate.loc41 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc41_34.2: Core.IntLiteral = converted %int.snegate.loc41, %.loc41_34.1 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int.convert_checked.loc41_34: init %i32 = call %Convert.specific_fn.loc41_34(%.loc41_34.2) [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc41_34.3: %i32 = value_of_initializer %int.convert_checked.loc41_34 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc41_34.4: %i32 = converted %int.snegate.loc41, %.loc41_34.3 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %int.left_shift.loc41: init %i32 = call %LeftShift.ref.loc41(%.loc41_31.2, %.loc41_34.4) [template = <error>]
// CHECK:STDOUT:   %.loc41_37.1: %i32 = value_of_initializer %int.left_shift.loc41 [template = <error>]
// CHECK:STDOUT:   %.loc41_37.2: %i32 = converted %int.left_shift.loc41, %.loc41_37.1 [template = <error>]
// CHECK:STDOUT:   %negative: %i32 = bind_name negative, %.loc41_37.2
// CHECK:STDOUT:   %LeftShift.ref.loc46: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_0.loc46: Core.IntLiteral = int_value 0 [template = constants.%int_0.2]
// CHECK:STDOUT:   %int_1.loc46: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc46_39.1: %Op.type.13 = interface_witness_access constants.%interface.20, element0 [template = constants.%Op.14]
// CHECK:STDOUT:   %Op.bound.loc46: <bound method> = bound_method %int_1.loc46, %impl.elem0.loc46_39.1 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc46: init Core.IntLiteral = call %Op.bound.loc46(%int_1.loc46) [template = constants.%int_-1.1]
// CHECK:STDOUT:   %impl.elem0.loc46_36: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc46_36: <bound method> = bound_method %int_0.loc46, %impl.elem0.loc46_36 [template = constants.%Convert.bound.6]
// CHECK:STDOUT:   %Convert.specific_fn.loc46_36: <specific function> = specific_function %Convert.bound.loc46_36, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.6]
// CHECK:STDOUT:   %int.convert_checked.loc46_36: init %i32 = call %Convert.specific_fn.loc46_36(%int_0.loc46) [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc46_36.1: %i32 = value_of_initializer %int.convert_checked.loc46_36 [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc46_36.2: %i32 = converted %int_0.loc46, %.loc46_36.1 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc46_39.2: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc46_39: <bound method> = bound_method %int.snegate.loc46, %impl.elem0.loc46_39.2 [template = constants.%Convert.bound.7]
// CHECK:STDOUT:   %Convert.specific_fn.loc46_39: <specific function> = specific_function %Convert.bound.loc46_39, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.7]
// CHECK:STDOUT:   %.loc46_39.1: Core.IntLiteral = value_of_initializer %int.snegate.loc46 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc46_39.2: Core.IntLiteral = converted %int.snegate.loc46, %.loc46_39.1 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int.convert_checked.loc46_39: init %i32 = call %Convert.specific_fn.loc46_39(%.loc46_39.2) [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc46_39.3: %i32 = value_of_initializer %int.convert_checked.loc46_39 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc46_39.4: %i32 = converted %int.snegate.loc46, %.loc46_39.3 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %int.left_shift.loc46: init %i32 = call %LeftShift.ref.loc46(%.loc46_36.2, %.loc46_39.4) [template = <error>]
// CHECK:STDOUT:   %.loc46_42.1: %i32 = value_of_initializer %int.left_shift.loc46 [template = <error>]
// CHECK:STDOUT:   %.loc46_42.2: %i32 = converted %int.left_shift.loc46, %.loc46_42.1 [template = <error>]
// CHECK:STDOUT:   %negative_zero: %i32 = bind_name negative_zero, %.loc46_42.2
// CHECK:STDOUT:   %LeftShiftLit.ref.loc51: %LeftShiftLit.type = name_ref LeftShiftLit, file.%LeftShiftLit.decl [template = constants.%LeftShiftLit]
// CHECK:STDOUT:   %int_1.loc51_52: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_1.loc51_56: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc51_55.1: %Op.type.13 = interface_witness_access constants.%interface.20, element0 [template = constants.%Op.14]
// CHECK:STDOUT:   %Op.bound.loc51: <bound method> = bound_method %int_1.loc51_56, %impl.elem0.loc51_55.1 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc51: init Core.IntLiteral = call %Op.bound.loc51(%int_1.loc51_56) [template = constants.%int_-1.1]
// CHECK:STDOUT:   %impl.elem0.loc51_55.2: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc51: <bound method> = bound_method %int.snegate.loc51, %impl.elem0.loc51_55.2 [template = constants.%Convert.bound.7]
// CHECK:STDOUT:   %Convert.specific_fn.loc51: <specific function> = specific_function %Convert.bound.loc51, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.7]
// CHECK:STDOUT:   %.loc51_55.1: Core.IntLiteral = value_of_initializer %int.snegate.loc51 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc51_55.2: Core.IntLiteral = converted %int.snegate.loc51, %.loc51_55.1 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int.convert_checked.loc51: init %i32 = call %Convert.specific_fn.loc51(%.loc51_55.2) [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc51_55.3: %i32 = value_of_initializer %int.convert_checked.loc51 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc51_55.4: %i32 = converted %int.snegate.loc51, %.loc51_55.3 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %int.left_shift.loc51: init Core.IntLiteral = call %LeftShiftLit.ref.loc51(%int_1.loc51_52, %.loc51_55.4) [template = <error>]
// CHECK:STDOUT:   %.loc51_58.1: Core.IntLiteral = value_of_initializer %int.left_shift.loc51 [template = <error>]
// CHECK:STDOUT:   %.loc51_58.2: Core.IntLiteral = converted %int.left_shift.loc51, %.loc51_58.1 [template = <error>]
// CHECK:STDOUT:   %negative_lit: Core.IntLiteral = bind_name negative_lit, %.loc51_58.2
// CHECK:STDOUT:   %LeftShiftLit.ref.loc56: %LeftShiftLit.type = name_ref LeftShiftLit, file.%LeftShiftLit.decl [template = constants.%LeftShiftLit]
// CHECK:STDOUT:   %int_0.loc56: Core.IntLiteral = int_value 0 [template = constants.%int_0.2]
// CHECK:STDOUT:   %int_1.loc56: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc56_60.1: %Op.type.13 = interface_witness_access constants.%interface.20, element0 [template = constants.%Op.14]
// CHECK:STDOUT:   %Op.bound.loc56: <bound method> = bound_method %int_1.loc56, %impl.elem0.loc56_60.1 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc56: init Core.IntLiteral = call %Op.bound.loc56(%int_1.loc56) [template = constants.%int_-1.1]
// CHECK:STDOUT:   %impl.elem0.loc56_60.2: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc56: <bound method> = bound_method %int.snegate.loc56, %impl.elem0.loc56_60.2 [template = constants.%Convert.bound.7]
// CHECK:STDOUT:   %Convert.specific_fn.loc56: <specific function> = specific_function %Convert.bound.loc56, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.7]
// CHECK:STDOUT:   %.loc56_60.1: Core.IntLiteral = value_of_initializer %int.snegate.loc56 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc56_60.2: Core.IntLiteral = converted %int.snegate.loc56, %.loc56_60.1 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int.convert_checked.loc56: init %i32 = call %Convert.specific_fn.loc56(%.loc56_60.2) [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc56_60.3: %i32 = value_of_initializer %int.convert_checked.loc56 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc56_60.4: %i32 = converted %int.snegate.loc56, %.loc56_60.3 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %int.left_shift.loc56: init Core.IntLiteral = call %LeftShiftLit.ref.loc56(%int_0.loc56, %.loc56_60.4) [template = <error>]
// CHECK:STDOUT:   %.loc56_63.1: Core.IntLiteral = value_of_initializer %int.left_shift.loc56 [template = <error>]
// CHECK:STDOUT:   %.loc56_63.2: Core.IntLiteral = converted %int.left_shift.loc56, %.loc56_63.1 [template = <error>]
// CHECK:STDOUT:   %negative_lit_zero: Core.IntLiteral = bind_name negative_lit_zero, %.loc56_63.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_literal_overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %LeftShift.type.1: type = fn_type @LeftShift.1 [template]
// CHECK:STDOUT:   %LeftShift: %LeftShift.type.1 = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_1000000000: Core.IntLiteral = int_value 1000000000 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.19: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Op.type.13: type = fn_type @Op.13 [template]
// CHECK:STDOUT:   %Op.type.14: type = fn_type @Op.14 [template]
// CHECK:STDOUT:   %Op.14: %Op.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.20: <witness> = interface_witness (%Op.14) [template]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %int_1, %Op.14 [template]
// CHECK:STDOUT:   %int_-1: Core.IntLiteral = int_value -1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IntLiteral = %import_ref.1
// CHECK:STDOUT:     .Int = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.6
// CHECK:STDOUT:     .Negate = %import_ref.230
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .LeftShift = %LeftShift.decl
// CHECK:STDOUT:     .bad = @__global_init.%bad
// CHECK:STDOUT:     .bad_negative = @__global_init.%bad_negative
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %LeftShift.decl: %LeftShift.type.1 = fn_decl @LeftShift.1 [template = constants.%LeftShift] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: Core.IntLiteral = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: Core.IntLiteral = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc4_61: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_65: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_77: init type = call %IntLiteral.ref.loc4_65() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_77.1: type = value_of_initializer %int_literal.make_type.loc4_77 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_77.2: type = converted %int_literal.make_type.loc4_77, %.loc4_77.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc4_33.3: type = splice_block %.loc4_33.2 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc4_17: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc4_21: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc4_33: init type = call %IntLiteral.ref.loc4_21() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_33.1: type = value_of_initializer %int_literal.make_type.loc4_33 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_33.2: type = converted %int_literal.make_type.loc4_33, %.loc4_33.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: Core.IntLiteral = value_param runtime_param1
// CHECK:STDOUT:     %.loc4_55.3: type = splice_block %.loc4_55.2 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc4_39: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc4_43: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc4_55: init type = call %IntLiteral.ref.loc4_43() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_55.1: type = value_of_initializer %int_literal.make_type.loc4_55 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_55.2: type = converted %int_literal.make_type.loc4_55, %.loc4_55.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: Core.IntLiteral = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param2
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LeftShift.1(%a.param_patt: Core.IntLiteral, %b.param_patt: Core.IntLiteral) -> Core.IntLiteral = "int.left_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %LeftShift.ref.loc10: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc10: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_1000000000.loc10: Core.IntLiteral = int_value 1000000000 [template = constants.%int_1000000000]
// CHECK:STDOUT:   %int.left_shift.loc10: init Core.IntLiteral = call %LeftShift.ref.loc10(%int_1.loc10, %int_1000000000.loc10) [template = <error>]
// CHECK:STDOUT:   %impl.elem0.loc10: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc10: <bound method> = bound_method %int.left_shift.loc10, %impl.elem0.loc10 [template = <error>]
// CHECK:STDOUT:   %Convert.specific_fn.loc10: <specific function> = specific_function %Convert.bound.loc10, @Convert.2(constants.%int_32) [template = <error>]
// CHECK:STDOUT:   %.loc10_42.1: Core.IntLiteral = value_of_initializer %int.left_shift.loc10 [template = <error>]
// CHECK:STDOUT:   %.loc10_42.2: Core.IntLiteral = converted %int.left_shift.loc10, %.loc10_42.1 [template = <error>]
// CHECK:STDOUT:   %int.convert_checked.loc10: init %i32 = call %Convert.specific_fn.loc10(%.loc10_42.2) [template = <error>]
// CHECK:STDOUT:   %.loc10_43.1: %i32 = value_of_initializer %int.convert_checked.loc10 [template = <error>]
// CHECK:STDOUT:   %.loc10_43.2: %i32 = converted %int.left_shift.loc10, %.loc10_43.1 [template = <error>]
// CHECK:STDOUT:   %bad: %i32 = bind_name bad, %.loc10_43.2
// CHECK:STDOUT:   %LeftShift.ref.loc16: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc16: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc16_35: %Op.type.13 = interface_witness_access constants.%interface.20, element0 [template = constants.%Op.14]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %int_1.loc16, %impl.elem0.loc16_35 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate: init Core.IntLiteral = call %Op.bound(%int_1.loc16) [template = constants.%int_-1]
// CHECK:STDOUT:   %int_1000000000.loc16: Core.IntLiteral = int_value 1000000000 [template = constants.%int_1000000000]
// CHECK:STDOUT:   %.loc16_35.1: Core.IntLiteral = value_of_initializer %int.snegate [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc16_35.2: Core.IntLiteral = converted %int.snegate, %.loc16_35.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.left_shift.loc16: init Core.IntLiteral = call %LeftShift.ref.loc16(%.loc16_35.2, %int_1000000000.loc16) [template = <error>]
// CHECK:STDOUT:   %impl.elem0.loc16_53: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc16: <bound method> = bound_method %int.left_shift.loc16, %impl.elem0.loc16_53 [template = <error>]
// CHECK:STDOUT:   %Convert.specific_fn.loc16: <specific function> = specific_function %Convert.bound.loc16, @Convert.2(constants.%int_32) [template = <error>]
// CHECK:STDOUT:   %.loc16_52.1: Core.IntLiteral = value_of_initializer %int.left_shift.loc16 [template = <error>]
// CHECK:STDOUT:   %.loc16_52.2: Core.IntLiteral = converted %int.left_shift.loc16, %.loc16_52.1 [template = <error>]
// CHECK:STDOUT:   %int.convert_checked.loc16: init %i32 = call %Convert.specific_fn.loc16(%.loc16_52.2) [template = <error>]
// CHECK:STDOUT:   %.loc16_53.1: %i32 = value_of_initializer %int.convert_checked.loc16 [template = <error>]
// CHECK:STDOUT:   %.loc16_53.2: %i32 = converted %int.left_shift.loc16, %.loc16_53.1 [template = <error>]
// CHECK:STDOUT:   %bad_negative: %i32 = bind_name bad_negative, %.loc16_53.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_comp_time_only_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %LeftShiftByLit.type: type = fn_type @LeftShiftByLit [template]
// CHECK:STDOUT:   %LeftShiftByLit: %LeftShiftByLit.type = struct_value () [template]
// CHECK:STDOUT:   %LeftShiftOfLit.type: type = fn_type @LeftShiftOfLit [template]
// CHECK:STDOUT:   %LeftShiftOfLit: %LeftShiftOfLit.type = struct_value () [template]
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %int_34.1: Core.IntLiteral = int_value 34 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.19: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_34.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_34.2: %i32 = int_value 34 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .IntLiteral = %import_ref.5
// CHECK:STDOUT:     .ImplicitAs = %import_ref.6
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .LeftShiftByLit = %LeftShiftByLit.decl
// CHECK:STDOUT:     .LeftShiftOfLit = %LeftShiftOfLit.decl
// CHECK:STDOUT:     .a_lit = %a_lit
// CHECK:STDOUT:     .an_i32 = %an_i32
// CHECK:STDOUT:     .bad1 = @__global_init.%bad1
// CHECK:STDOUT:     .bad2 = @__global_init.%bad2
// CHECK:STDOUT:     .bad3 = @__global_init.%bad3
// CHECK:STDOUT:     .bad4 = @__global_init.%bad4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %LeftShiftByLit.decl: %LeftShiftByLit.type = fn_decl @LeftShiftByLit [template = constants.%LeftShiftByLit] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: Core.IntLiteral = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: Core.IntLiteral = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc4_52: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_52: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc4_22: type = splice_block %i32.loc4_22 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc4_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc4_22: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: Core.IntLiteral = value_param runtime_param1
// CHECK:STDOUT:     %.loc4_46.3: type = splice_block %.loc4_46.2 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.5 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_46.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_46.2: type = converted %int_literal.make_type, %.loc4_46.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: Core.IntLiteral = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %LeftShiftOfLit.decl: %LeftShiftOfLit.type = fn_decl @LeftShiftOfLit [template = constants.%LeftShiftOfLit] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc5_52: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc5_56: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.5 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc5_68: init type = call %IntLiteral.ref.loc5_56() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc5_68.1: type = value_of_initializer %int_literal.make_type.loc5_68 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc5_68.2: type = converted %int_literal.make_type.loc5_68, %.loc5_68.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc5_38.3: type = splice_block %.loc5_38.2 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc5_22: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc5_26: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.5 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc5_38: init type = call %IntLiteral.ref.loc5_26() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc5_38.1: type = value_of_initializer %int_literal.make_type.loc5_38 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc5_38.2: type = converted %int_literal.make_type.loc5_38, %.loc5_38.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc5_44: type = splice_block %i32 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param2
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_lit.var: ref Core.IntLiteral = var a_lit
// CHECK:STDOUT:   %a_lit: ref Core.IntLiteral = bind_name a_lit, %a_lit.var
// CHECK:STDOUT:   %an_i32.var: ref %i32 = var an_i32
// CHECK:STDOUT:   %an_i32: ref %i32 = bind_name an_i32, %an_i32.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LeftShiftByLit(%a.param_patt: %i32, %b.param_patt: Core.IntLiteral) -> %i32 = "int.left_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LeftShiftOfLit(%a.param_patt: Core.IntLiteral, %b.param_patt: %i32) -> Core.IntLiteral = "int.left_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_12.loc7: Core.IntLiteral = int_value 12 [template = constants.%int_12]
// CHECK:STDOUT:   assign file.%a_lit.var, %int_12.loc7
// CHECK:STDOUT:   %int_34: Core.IntLiteral = int_value 34 [template = constants.%int_34.1]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_34, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_34) [template = constants.%int_34.2]
// CHECK:STDOUT:   %.loc8: init %i32 = converted %int_34, %int.convert_checked [template = constants.%int_34.2]
// CHECK:STDOUT:   assign file.%an_i32.var, %.loc8
// CHECK:STDOUT:   %LeftShiftByLit.ref.loc18: %LeftShiftByLit.type = name_ref LeftShiftByLit, file.%LeftShiftByLit.decl [template = constants.%LeftShiftByLit]
// CHECK:STDOUT:   %an_i32.ref.loc18: ref %i32 = name_ref an_i32, file.%an_i32
// CHECK:STDOUT:   %a_lit.ref.loc18: ref Core.IntLiteral = name_ref a_lit, file.%a_lit
// CHECK:STDOUT:   %.loc18_32: %i32 = bind_value %an_i32.ref.loc18
// CHECK:STDOUT:   %.loc18_40: Core.IntLiteral = bind_value %a_lit.ref.loc18
// CHECK:STDOUT:   %int.left_shift.loc18: init %i32 = call %LeftShiftByLit.ref.loc18(%.loc18_32, %.loc18_40)
// CHECK:STDOUT:   %.loc18_46.1: %i32 = value_of_initializer %int.left_shift.loc18
// CHECK:STDOUT:   %.loc18_46.2: %i32 = converted %int.left_shift.loc18, %.loc18_46.1
// CHECK:STDOUT:   %bad1: %i32 = bind_name bad1, %.loc18_46.2
// CHECK:STDOUT:   %LeftShiftOfLit.ref.loc28: %LeftShiftOfLit.type = name_ref LeftShiftOfLit, file.%LeftShiftOfLit.decl [template = constants.%LeftShiftOfLit]
// CHECK:STDOUT:   %a_lit.ref.loc28: ref Core.IntLiteral = name_ref a_lit, file.%a_lit
// CHECK:STDOUT:   %an_i32.ref.loc28: ref %i32 = name_ref an_i32, file.%an_i32
// CHECK:STDOUT:   %.loc28_46: Core.IntLiteral = bind_value %a_lit.ref.loc28
// CHECK:STDOUT:   %.loc28_53: %i32 = bind_value %an_i32.ref.loc28
// CHECK:STDOUT:   %int.left_shift.loc28: init Core.IntLiteral = call %LeftShiftOfLit.ref.loc28(%.loc28_46, %.loc28_53)
// CHECK:STDOUT:   %.loc28_60.1: Core.IntLiteral = value_of_initializer %int.left_shift.loc28
// CHECK:STDOUT:   %.loc28_60.2: Core.IntLiteral = converted %int.left_shift.loc28, %.loc28_60.1
// CHECK:STDOUT:   %bad2: Core.IntLiteral = bind_name bad2, %.loc28_60.2
// CHECK:STDOUT:   %LeftShiftByLit.ref.loc38: %LeftShiftByLit.type = name_ref LeftShiftByLit, file.%LeftShiftByLit.decl [template = constants.%LeftShiftByLit]
// CHECK:STDOUT:   %an_i32.ref.loc38: ref %i32 = name_ref an_i32, file.%an_i32
// CHECK:STDOUT:   %int_12.loc38: Core.IntLiteral = int_value 12 [template = constants.%int_12]
// CHECK:STDOUT:   %.loc38_32: %i32 = bind_value %an_i32.ref.loc38
// CHECK:STDOUT:   %int.left_shift.loc38: init %i32 = call %LeftShiftByLit.ref.loc38(%.loc38_32, %int_12.loc38)
// CHECK:STDOUT:   %.loc38_43.1: %i32 = value_of_initializer %int.left_shift.loc38
// CHECK:STDOUT:   %.loc38_43.2: %i32 = converted %int.left_shift.loc38, %.loc38_43.1
// CHECK:STDOUT:   %bad3: %i32 = bind_name bad3, %.loc38_43.2
// CHECK:STDOUT:   %LeftShiftOfLit.ref.loc47: %LeftShiftOfLit.type = name_ref LeftShiftOfLit, file.%LeftShiftOfLit.decl [template = constants.%LeftShiftOfLit]
// CHECK:STDOUT:   %int_12.loc47: Core.IntLiteral = int_value 12 [template = constants.%int_12]
// CHECK:STDOUT:   %an_i32.ref.loc47: ref %i32 = name_ref an_i32, file.%an_i32
// CHECK:STDOUT:   %.loc47_50: %i32 = bind_value %an_i32.ref.loc47
// CHECK:STDOUT:   %int.left_shift.loc47: init Core.IntLiteral = call %LeftShiftOfLit.ref.loc47(%int_12.loc47, %.loc47_50)
// CHECK:STDOUT:   %.loc47_57.1: Core.IntLiteral = value_of_initializer %int.left_shift.loc47
// CHECK:STDOUT:   %.loc47_57.2: Core.IntLiteral = converted %int.left_shift.loc47, %.loc47_57.1
// CHECK:STDOUT:   %bad4: Core.IntLiteral = bind_name bad4, %.loc47_57.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
