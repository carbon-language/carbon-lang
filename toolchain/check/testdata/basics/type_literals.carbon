// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/type_literals.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/type_literals.carbon

// --- iN.carbon
library "[[@TEST_NAME]]";

var test_i8: i8;
var test_i16: i16;
var test_i64: i64;

// --- fail_iN_bad_width.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_iN_bad_width.carbon:[[@LINE+4]]:14: error: name `i0` not found [NameNotFound]
// CHECK:STDERR: var test_i0: i0;
// CHECK:STDERR:              ^~
// CHECK:STDERR:
var test_i0: i0;
// CHECK:STDERR: fail_iN_bad_width.carbon:[[@LINE+4]]:14: error: bit width of integer type literal must be a multiple of 8; use `Core.Int(1)` instead [IntWidthNotMultipleOf8]
// CHECK:STDERR: var test_i1: i1;
// CHECK:STDERR:              ^~
// CHECK:STDERR:
var test_i1: i1;
// CHECK:STDERR: fail_iN_bad_width.carbon:[[@LINE+7]]:15: error: bit width of integer type literal must be a multiple of 8; use `Core.Int(15)` instead [IntWidthNotMultipleOf8]
// CHECK:STDERR: var test_i15: i15;
// CHECK:STDERR:               ^~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:13:9: error: integer type width of 1000000000 is greater than the maximum supported width of 8388608 [IntWidthTooLarge]
// CHECK:STDERR:   adapt MakeInt(N);
// CHECK:STDERR:         ^~~~~~~~~~
var test_i15: i15;
// CHECK:STDERR: fail_iN_bad_width.carbon:[[@LINE+4]]:23: note: in `i1000000000` used here [ResolvingSpecificHere]
// CHECK:STDERR: var test_i1000000000: i1000000000;
// CHECK:STDERR:                       ^~~~~~~~~~~
// CHECK:STDERR:
var test_i1000000000: i1000000000;
// TODO: This diagnostic is not very good.
// CHECK:STDERR: fail_iN_bad_width.carbon:[[@LINE+8]]:33: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: var test_i10000000000000000000: i10000000000000000000;
// CHECK:STDERR:                                 ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_iN_bad_width.carbon:[[@LINE+4]]:34: error: found a type literal with a bit width using 20 digits, which is greater than the limit of 18 [TooManyTypeBitWidthDigits]
// CHECK:STDERR: var test_i10000000000000000000: i10000000000000000000;
// CHECK:STDERR:                                  ^
// CHECK:STDERR:
var test_i10000000000000000000: i10000000000000000000;

// --- uN.carbon
library "[[@TEST_NAME]]";

var test_u8: u8;
var test_u16: u16;
var test_u64: u64;

// --- fail_uN_bad_width.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_uN_bad_width.carbon:[[@LINE+4]]:14: error: name `u0` not found [NameNotFound]
// CHECK:STDERR: var test_u0: u0;
// CHECK:STDERR:              ^~
// CHECK:STDERR:
var test_u0: u0;
// CHECK:STDERR: fail_uN_bad_width.carbon:[[@LINE+4]]:14: error: bit width of integer type literal must be a multiple of 8; use `Core.UInt(1)` instead [IntWidthNotMultipleOf8]
// CHECK:STDERR: var test_u1: u1;
// CHECK:STDERR:              ^~
// CHECK:STDERR:
var test_u1: u1;
// CHECK:STDERR: fail_uN_bad_width.carbon:[[@LINE+7]]:15: error: bit width of integer type literal must be a multiple of 8; use `Core.UInt(15)` instead [IntWidthNotMultipleOf8]
// CHECK:STDERR: var test_u15: u15;
// CHECK:STDERR:               ^~~
// CHECK:STDERR:
// CHECK:STDERR: {{.*}}/prelude/types/uint.carbon:14:9: error: integer type width of 1000000000 is greater than the maximum supported width of 8388608 [IntWidthTooLarge]
// CHECK:STDERR:   adapt MakeUInt(N);
// CHECK:STDERR:         ^~~~~~~~~~~
var test_u15: u15;
// CHECK:STDERR: fail_uN_bad_width.carbon:[[@LINE+4]]:23: note: in `u1000000000` used here [ResolvingSpecificHere]
// CHECK:STDERR: var test_u1000000000: u1000000000;
// CHECK:STDERR:                       ^~~~~~~~~~~
// CHECK:STDERR:
var test_u1000000000: u1000000000;
// TODO: This diagnostic is not very good.
// CHECK:STDERR: fail_uN_bad_width.carbon:[[@LINE+8]]:33: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: var test_u10000000000000000000: u10000000000000000000;
// CHECK:STDERR:                                 ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_uN_bad_width.carbon:[[@LINE+4]]:34: error: found a type literal with a bit width using 20 digits, which is greater than the limit of 18 [TooManyTypeBitWidthDigits]
// CHECK:STDERR: var test_u10000000000000000000: u10000000000000000000;
// CHECK:STDERR:                                  ^
// CHECK:STDERR:
var test_u10000000000000000000: u10000000000000000000;

// --- fail_fN_bad_width.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_fN_bad_width.carbon:[[@LINE+4]]:14: error: name `f0` not found [NameNotFound]
// CHECK:STDERR: var test_f0: f0;
// CHECK:STDERR:              ^~
// CHECK:STDERR:
var test_f0: f0;
// CHECK:STDERR: fail_fN_bad_width.carbon:[[@LINE+4]]:14: error: semantics TODO: `Currently only f64 is allowed` [SemanticsTodo]
// CHECK:STDERR: var test_f1: f1;
// CHECK:STDERR:              ^~
// CHECK:STDERR:
var test_f1: f1;
var test_f15: f15;
var test_f100: f100;
var test_f1000000000: f1000000000;
var test_f1000000000000: f1000000000000;

// --- fail_fN_todo_unsupported.carbon
library "[[@TEST_NAME]]";

// TODO: Some or all of these should eventually work.
// CHECK:STDERR: fail_fN_todo_unsupported.carbon:[[@LINE+4]]:15: error: semantics TODO: `Currently only f64 is allowed` [SemanticsTodo]
// CHECK:STDERR: var test_f16: f16;
// CHECK:STDERR:               ^~~
// CHECK:STDERR:
var test_f16: f16;
var test_f32: f32;
var test_f128: f128;

// CHECK:STDOUT: --- iN.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %i8: type = class_type @Int, @Int(%int_8) [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .test_i8 = %test_i8
// CHECK:STDOUT:     .test_i16 = %test_i16
// CHECK:STDOUT:     .test_i64 = %test_i64
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_i8.patt: %i8 = binding_pattern test_i8
// CHECK:STDOUT:     %.loc3_1: %i8 = var_pattern %test_i8.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_i8.var: ref %i8 = var test_i8
// CHECK:STDOUT:   %.loc3_14: type = splice_block %i8 [concrete = constants.%i8] {
// CHECK:STDOUT:     %int_8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8]
// CHECK:STDOUT:     %i8: type = class_type @Int, @Int(constants.%int_8) [concrete = constants.%i8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_i8: ref %i8 = bind_name test_i8, %test_i8.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_i16.patt: %i16 = binding_pattern test_i16
// CHECK:STDOUT:     %.loc4_1: %i16 = var_pattern %test_i16.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_i16.var: ref %i16 = var test_i16
// CHECK:STDOUT:   %.loc4_15: type = splice_block %i16 [concrete = constants.%i16] {
// CHECK:STDOUT:     %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:     %i16: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_i16: ref %i16 = bind_name test_i16, %test_i16.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_i64.patt: %i64 = binding_pattern test_i64
// CHECK:STDOUT:     %.loc5_1: %i64 = var_pattern %test_i64.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_i64.var: ref %i64 = var test_i64
// CHECK:STDOUT:   %.loc5_15: type = splice_block %i64 [concrete = constants.%i64] {
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %i64: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_i64: ref %i64 = bind_name test_i64, %test_i64.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_iN_bad_width.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- uN.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %u8: type = class_type @UInt, @UInt(%int_8) [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %u16: type = class_type @UInt, @UInt(%int_16) [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .UInt = %Core.UInt
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .test_u8 = %test_u8
// CHECK:STDOUT:     .test_u16 = %test_u16
// CHECK:STDOUT:     .test_u64 = %test_u64
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_u8.patt: %u8 = binding_pattern test_u8
// CHECK:STDOUT:     %.loc3_1: %u8 = var_pattern %test_u8.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_u8.var: ref %u8 = var test_u8
// CHECK:STDOUT:   %.loc3_14: type = splice_block %u8 [concrete = constants.%u8] {
// CHECK:STDOUT:     %int_8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8]
// CHECK:STDOUT:     %u8: type = class_type @UInt, @UInt(constants.%int_8) [concrete = constants.%u8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_u8: ref %u8 = bind_name test_u8, %test_u8.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_u16.patt: %u16 = binding_pattern test_u16
// CHECK:STDOUT:     %.loc4_1: %u16 = var_pattern %test_u16.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_u16.var: ref %u16 = var test_u16
// CHECK:STDOUT:   %.loc4_15: type = splice_block %u16 [concrete = constants.%u16] {
// CHECK:STDOUT:     %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:     %u16: type = class_type @UInt, @UInt(constants.%int_16) [concrete = constants.%u16]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_u16: ref %u16 = bind_name test_u16, %test_u16.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %test_u64.patt: %u64 = binding_pattern test_u64
// CHECK:STDOUT:     %.loc5_1: %u64 = var_pattern %test_u64.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_u64.var: ref %u64 = var test_u64
// CHECK:STDOUT:   %.loc5_15: type = splice_block %u64 [concrete = constants.%u64] {
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %u64: type = class_type @UInt, @UInt(constants.%int_64) [concrete = constants.%u64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %test_u64: ref %u64 = bind_name test_u64, %test_u64.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_uN_bad_width.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_fN_bad_width.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_fN_todo_unsupported.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
