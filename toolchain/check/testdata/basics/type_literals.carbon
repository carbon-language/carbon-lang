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
// CHECK:STDERR: fail_iN_bad_width.carbon:[[@LINE+4]]:15: error: bit width of integer type literal must be a multiple of 8; use `Core.Int(15)` instead [IntWidthNotMultipleOf8]
// CHECK:STDERR: var test_i15: i15;
// CHECK:STDERR:               ^~~
// CHECK:STDERR:
var test_i15: i15;
// CHECK:STDERR: fail_iN_bad_width.carbon:[[@LINE+4]]:23: error: integer type width of 1000000000 is greater than the maximum supported width of 8388608 [IntWidthTooLarge]
// CHECK:STDERR: var test_i1000000000: i1000000000;
// CHECK:STDERR:                       ^~~~~~~~~~~
// CHECK:STDERR:
var test_i1000000000: i1000000000;
// TODO: This diagnostic is not very good.
// CHECK:STDERR: fail_iN_bad_width.carbon:[[@LINE+12]]:33: error: expected expression [ExpectedExpr]
// CHECK:STDERR: var test_i10000000000000000000: i10000000000000000000;
// CHECK:STDERR:                                 ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_iN_bad_width.carbon:[[@LINE+8]]:33: error: semantics TODO: `HandleInvalidParse` [SemanticsTodo]
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
// CHECK:STDERR: fail_uN_bad_width.carbon:[[@LINE+4]]:15: error: bit width of integer type literal must be a multiple of 8; use `Core.UInt(15)` instead [IntWidthNotMultipleOf8]
// CHECK:STDERR: var test_u15: u15;
// CHECK:STDERR:               ^~~
// CHECK:STDERR:
var test_u15: u15;
// CHECK:STDERR: fail_uN_bad_width.carbon:[[@LINE+4]]:23: error: integer type width of 1000000000 is greater than the maximum supported width of 8388608 [IntWidthTooLarge]
// CHECK:STDERR: var test_u1000000000: u1000000000;
// CHECK:STDERR:                       ^~~~~~~~~~~
// CHECK:STDERR:
var test_u1000000000: u1000000000;
// TODO: This diagnostic is not very good.
// CHECK:STDERR: fail_uN_bad_width.carbon:[[@LINE+12]]:33: error: expected expression [ExpectedExpr]
// CHECK:STDERR: var test_u10000000000000000000: u10000000000000000000;
// CHECK:STDERR:                                 ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_uN_bad_width.carbon:[[@LINE+8]]:33: error: semantics TODO: `HandleInvalidParse` [SemanticsTodo]
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
// CHECK:STDERR: fail_fN_todo_unsupported.carbon:[[@LINE+3]]:15: error: semantics TODO: `Currently only f64 is allowed` [SemanticsTodo]
// CHECK:STDERR: var test_f16: f16;
// CHECK:STDERR:               ^~~
var test_f16: f16;
var test_f32: f32;
var test_f128: f128;

// CHECK:STDOUT: --- iN.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.BigInt = int_literal 8 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.4: Core.BigInt = int_literal 16 [template]
// CHECK:STDOUT:   %.5: type = int_type signed, %.4 [template]
// CHECK:STDOUT:   %.6: Core.BigInt = int_literal 64 [template]
// CHECK:STDOUT:   %.7: type = int_type signed, %.6 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int.type = import_ref Core//prelude/types, inst+30, loaded [template = constants.%Int]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .test_i8 = %test_i8
// CHECK:STDOUT:     .test_i16 = %test_i16
// CHECK:STDOUT:     .test_i64 = %test_i64
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %.loc3_14.1: Core.BigInt = int_literal 8 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc3: init type = call constants.%Int(%.loc3_14.1) [template = constants.%.3]
// CHECK:STDOUT:   %.loc3_14.2: type = value_of_initializer %int.make_type_signed.loc3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc3_14.3: type = converted %int.make_type_signed.loc3, %.loc3_14.2 [template = constants.%.3]
// CHECK:STDOUT:   %test_i8.var: ref %.3 = var test_i8
// CHECK:STDOUT:   %test_i8: ref %.3 = bind_name test_i8, %test_i8.var
// CHECK:STDOUT:   %.loc4_15.1: Core.BigInt = int_literal 16 [template = constants.%.4]
// CHECK:STDOUT:   %int.make_type_signed.loc4: init type = call constants.%Int(%.loc4_15.1) [template = constants.%.5]
// CHECK:STDOUT:   %.loc4_15.2: type = value_of_initializer %int.make_type_signed.loc4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc4_15.3: type = converted %int.make_type_signed.loc4, %.loc4_15.2 [template = constants.%.5]
// CHECK:STDOUT:   %test_i16.var: ref %.5 = var test_i16
// CHECK:STDOUT:   %test_i16: ref %.5 = bind_name test_i16, %test_i16.var
// CHECK:STDOUT:   %.loc5_15.1: Core.BigInt = int_literal 64 [template = constants.%.6]
// CHECK:STDOUT:   %int.make_type_signed.loc5: init type = call constants.%Int(%.loc5_15.1) [template = constants.%.7]
// CHECK:STDOUT:   %.loc5_15.2: type = value_of_initializer %int.make_type_signed.loc5 [template = constants.%.7]
// CHECK:STDOUT:   %.loc5_15.3: type = converted %int.make_type_signed.loc5, %.loc5_15.2 [template = constants.%.7]
// CHECK:STDOUT:   %test_i64.var: ref %.7 = var test_i64
// CHECK:STDOUT:   %test_i64: ref %.7 = bind_name test_i64, %test_i64.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%size.param_patt: Core.BigInt) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_iN_bad_width.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.BigInt = int_literal 1 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.4: Core.BigInt = int_literal 15 [template]
// CHECK:STDOUT:   %.5: type = int_type signed, %.4 [template]
// CHECK:STDOUT:   %.6: Core.BigInt = int_literal 1000000000 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%size.param_patt: Core.BigInt) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: --- uN.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.BigInt = int_literal 8 [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = int_type unsigned, %.1 [template]
// CHECK:STDOUT:   %.4: Core.BigInt = int_literal 16 [template]
// CHECK:STDOUT:   %.5: type = int_type unsigned, %.4 [template]
// CHECK:STDOUT:   %.6: Core.BigInt = int_literal 64 [template]
// CHECK:STDOUT:   %.7: type = int_type unsigned, %.6 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .UInt = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %UInt.type = import_ref Core//prelude/types, inst+45, loaded [template = constants.%UInt]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .test_u8 = %test_u8
// CHECK:STDOUT:     .test_u16 = %test_u16
// CHECK:STDOUT:     .test_u64 = %test_u64
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %.loc3_14.1: Core.BigInt = int_literal 8 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_unsigned.loc3: init type = call constants.%UInt(%.loc3_14.1) [template = constants.%.3]
// CHECK:STDOUT:   %.loc3_14.2: type = value_of_initializer %int.make_type_unsigned.loc3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc3_14.3: type = converted %int.make_type_unsigned.loc3, %.loc3_14.2 [template = constants.%.3]
// CHECK:STDOUT:   %test_u8.var: ref %.3 = var test_u8
// CHECK:STDOUT:   %test_u8: ref %.3 = bind_name test_u8, %test_u8.var
// CHECK:STDOUT:   %.loc4_15.1: Core.BigInt = int_literal 16 [template = constants.%.4]
// CHECK:STDOUT:   %int.make_type_unsigned.loc4: init type = call constants.%UInt(%.loc4_15.1) [template = constants.%.5]
// CHECK:STDOUT:   %.loc4_15.2: type = value_of_initializer %int.make_type_unsigned.loc4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc4_15.3: type = converted %int.make_type_unsigned.loc4, %.loc4_15.2 [template = constants.%.5]
// CHECK:STDOUT:   %test_u16.var: ref %.5 = var test_u16
// CHECK:STDOUT:   %test_u16: ref %.5 = bind_name test_u16, %test_u16.var
// CHECK:STDOUT:   %.loc5_15.1: Core.BigInt = int_literal 64 [template = constants.%.6]
// CHECK:STDOUT:   %int.make_type_unsigned.loc5: init type = call constants.%UInt(%.loc5_15.1) [template = constants.%.7]
// CHECK:STDOUT:   %.loc5_15.2: type = value_of_initializer %int.make_type_unsigned.loc5 [template = constants.%.7]
// CHECK:STDOUT:   %.loc5_15.3: type = converted %int.make_type_unsigned.loc5, %.loc5_15.2 [template = constants.%.7]
// CHECK:STDOUT:   %test_u64.var: ref %.7 = var test_u64
// CHECK:STDOUT:   %test_u64: ref %.7 = bind_name test_u64, %test_u64.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UInt(%size.param_patt: Core.BigInt) -> type = "int.make_type_unsigned";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_uN_bad_width.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.BigInt = int_literal 1 [template]
// CHECK:STDOUT:   %UInt.type: type = fn_type @UInt [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %UInt: %UInt.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = int_type unsigned, %.1 [template]
// CHECK:STDOUT:   %.4: Core.BigInt = int_literal 15 [template]
// CHECK:STDOUT:   %.5: type = int_type unsigned, %.4 [template]
// CHECK:STDOUT:   %.6: Core.BigInt = int_literal 1000000000 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UInt(%size.param_patt: Core.BigInt) -> type = "int.make_type_unsigned";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_fN_bad_width.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_fN_todo_unsupported.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
