// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- iN.carbon
library "iN" api;

var test_i8: i8;
var test_i16: i16;
var test_i64: i64;

// --- fail_iN_bad_width.carbon
library "fail_iN_bad_width" api;

// CHECK:STDERR: fail_iN_bad_width.carbon:[[@LINE+4]]:14: ERROR: Name `i0` not found.
// CHECK:STDERR: var test_i0: i0;
// CHECK:STDERR:              ^~
// CHECK:STDERR:
var test_i0: i0;
// CHECK:STDERR: fail_iN_bad_width.carbon:[[@LINE+4]]:14: ERROR: Bit width of integer type literal must be a multiple of 8. Use `Core.Int(1)` instead.
// CHECK:STDERR: var test_i1: i1;
// CHECK:STDERR:              ^~
// CHECK:STDERR:
var test_i1: i1;
// CHECK:STDERR: fail_iN_bad_width.carbon:[[@LINE+4]]:15: ERROR: Bit width of integer type literal must be a multiple of 8. Use `Core.Int(15)` instead.
// CHECK:STDERR: var test_i15: i15;
// CHECK:STDERR:               ^~~
// CHECK:STDERR:
var test_i15: i15;
// CHECK:STDERR: fail_iN_bad_width.carbon:[[@LINE+4]]:23: ERROR: Integer type width of 1000000000 is greater than the maximum supported width of 8388608.
// CHECK:STDERR: var test_i1000000000: i1000000000;
// CHECK:STDERR:                       ^~~~~~~~~~~
// CHECK:STDERR:
var test_i1000000000: i1000000000;
// TODO: This diagnostic is not very good.
// CHECK:STDERR: fail_iN_bad_width.carbon:[[@LINE+4]]:33: ERROR: Integer literal with value 10000000000000000000 does not fit in i32.
// CHECK:STDERR: var test_i10000000000000000000: i10000000000000000000;
// CHECK:STDERR:                                 ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var test_i10000000000000000000: i10000000000000000000;

// --- uN.carbon
library "uN" api;

var test_u8: u8;
var test_u16: u16;
var test_u64: u64;

// --- fail_uN_bad_width.carbon
library "fail_uN_bad_width" api;

// CHECK:STDERR: fail_uN_bad_width.carbon:[[@LINE+4]]:14: ERROR: Name `u0` not found.
// CHECK:STDERR: var test_u0: u0;
// CHECK:STDERR:              ^~
// CHECK:STDERR:
var test_u0: u0;
// CHECK:STDERR: fail_uN_bad_width.carbon:[[@LINE+4]]:14: ERROR: Bit width of integer type literal must be a multiple of 8. Use `Core.UInt(1)` instead.
// CHECK:STDERR: var test_u1: u1;
// CHECK:STDERR:              ^~
// CHECK:STDERR:
var test_u1: u1;
// CHECK:STDERR: fail_uN_bad_width.carbon:[[@LINE+4]]:15: ERROR: Bit width of integer type literal must be a multiple of 8. Use `Core.UInt(15)` instead.
// CHECK:STDERR: var test_u15: u15;
// CHECK:STDERR:               ^~~
// CHECK:STDERR:
var test_u15: u15;
// CHECK:STDERR: fail_uN_bad_width.carbon:[[@LINE+4]]:23: ERROR: Integer type width of 1000000000 is greater than the maximum supported width of 8388608.
// CHECK:STDERR: var test_u1000000000: u1000000000;
// CHECK:STDERR:                       ^~~~~~~~~~~
// CHECK:STDERR:
var test_u1000000000: u1000000000;
// TODO: This diagnostic is not very good.
// CHECK:STDERR: fail_uN_bad_width.carbon:[[@LINE+4]]:33: ERROR: Integer literal with value 10000000000000000000 does not fit in i32.
// CHECK:STDERR: var test_u10000000000000000000: u10000000000000000000;
// CHECK:STDERR:                                 ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var test_u10000000000000000000: u10000000000000000000;

// --- fail_fN_bad_width.carbon
library "fail_fN_bad_width" api;

// CHECK:STDERR: fail_fN_bad_width.carbon:[[@LINE+4]]:14: ERROR: Name `f0` not found.
// CHECK:STDERR: var test_f0: f0;
// CHECK:STDERR:              ^~
// CHECK:STDERR:
var test_f0: f0;
// CHECK:STDERR: fail_fN_bad_width.carbon:[[@LINE+4]]:14: ERROR: Semantics TODO: `Currently only f64 is allowed`.
// CHECK:STDERR: var test_f1: f1;
// CHECK:STDERR:              ^~
// CHECK:STDERR:
var test_f1: f1;
var test_f15: f15;
var test_f100: f100;
var test_f1000000000: f1000000000;
var test_f1000000000000: f1000000000000;

// --- fail_fN_todo_unsupported.carbon
library "fail_fN_todo_unsupported" api;

// TODO: Some or all of these should eventually work.
// CHECK:STDERR: fail_fN_todo_unsupported.carbon:[[@LINE+3]]:15: ERROR: Semantics TODO: `Currently only f64 is allowed`.
// CHECK:STDERR: var test_f16: f16;
// CHECK:STDERR:               ^~~
var test_f16: f16;
var test_f32: f32;
var test_f128: f128;

// CHECK:STDOUT: --- iN.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 8 [template]
// CHECK:STDOUT:   %.2: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 16 [template]
// CHECK:STDOUT:   %.4: type = int_type signed, %.3 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 64 [template]
// CHECK:STDOUT:   %.6: type = int_type signed, %.5 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .test_i8 = %test_i8
// CHECK:STDOUT:     .test_i16 = %test_i16
// CHECK:STDOUT:     .test_i64 = %test_i64
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc3_14.1: i32 = int_literal 8 [template = constants.%.1]
// CHECK:STDOUT:   %.loc3_14.2: type = int_type signed, %.loc3_14.1 [template = constants.%.2]
// CHECK:STDOUT:   %test_i8.var: ref i8 = var test_i8
// CHECK:STDOUT:   %test_i8: ref i8 = bind_name test_i8, %test_i8.var
// CHECK:STDOUT:   %.loc4_15.1: i32 = int_literal 16 [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_15.2: type = int_type signed, %.loc4_15.1 [template = constants.%.4]
// CHECK:STDOUT:   %test_i16.var: ref i16 = var test_i16
// CHECK:STDOUT:   %test_i16: ref i16 = bind_name test_i16, %test_i16.var
// CHECK:STDOUT:   %.loc5_15.1: i32 = int_literal 64 [template = constants.%.5]
// CHECK:STDOUT:   %.loc5_15.2: type = int_type signed, %.loc5_15.1 [template = constants.%.6]
// CHECK:STDOUT:   %test_i64.var: ref i64 = var test_i64
// CHECK:STDOUT:   %test_i64: ref i64 = bind_name test_i64, %test_i64.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_iN_bad_width.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.2: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 15 [template]
// CHECK:STDOUT:   %.4: type = int_type signed, %.3 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1000000000 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .test_i0 = %test_i0
// CHECK:STDOUT:     .test_i1 = %test_i1
// CHECK:STDOUT:     .test_i15 = %test_i15
// CHECK:STDOUT:     .test_i1000000000 = %test_i1000000000
// CHECK:STDOUT:     .test_i10000000000000000000 = %test_i10000000000000000000
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %i0.ref: <error> = name_ref i0, <error> [template = <error>]
// CHECK:STDOUT:   %test_i0.var: ref <error> = var test_i0
// CHECK:STDOUT:   %test_i0: ref <error> = bind_name test_i0, %test_i0.var
// CHECK:STDOUT:   %.loc12_14.1: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc12_14.2: type = int_type signed, %.loc12_14.1 [template = constants.%.2]
// CHECK:STDOUT:   %test_i1.var: ref i1 = var test_i1
// CHECK:STDOUT:   %test_i1: ref i1 = bind_name test_i1, %test_i1.var
// CHECK:STDOUT:   %.loc17_15.1: i32 = int_literal 15 [template = constants.%.3]
// CHECK:STDOUT:   %.loc17_15.2: type = int_type signed, %.loc17_15.1 [template = constants.%.4]
// CHECK:STDOUT:   %test_i15.var: ref i15 = var test_i15
// CHECK:STDOUT:   %test_i15: ref i15 = bind_name test_i15, %test_i15.var
// CHECK:STDOUT:   %.loc22_23.1: i32 = int_literal 1000000000 [template = constants.%.5]
// CHECK:STDOUT:   %.loc22_23.2: type = int_type signed, %.loc22_23.1 [template = <error>]
// CHECK:STDOUT:   %test_i1000000000.var: ref <error> = var test_i1000000000
// CHECK:STDOUT:   %test_i1000000000: ref <error> = bind_name test_i1000000000, %test_i1000000000.var
// CHECK:STDOUT:   %.loc28: type = int_type signed, <error> [template = <error>]
// CHECK:STDOUT:   %test_i10000000000000000000.var: ref <error> = var test_i10000000000000000000
// CHECK:STDOUT:   %test_i10000000000000000000: ref <error> = bind_name test_i10000000000000000000, %test_i10000000000000000000.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- uN.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 8 [template]
// CHECK:STDOUT:   %.2: type = int_type unsigned, %.1 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 16 [template]
// CHECK:STDOUT:   %.4: type = int_type unsigned, %.3 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 64 [template]
// CHECK:STDOUT:   %.6: type = int_type unsigned, %.5 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .test_u8 = %test_u8
// CHECK:STDOUT:     .test_u16 = %test_u16
// CHECK:STDOUT:     .test_u64 = %test_u64
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc3_14.1: i32 = int_literal 8 [template = constants.%.1]
// CHECK:STDOUT:   %.loc3_14.2: type = int_type unsigned, %.loc3_14.1 [template = constants.%.2]
// CHECK:STDOUT:   %test_u8.var: ref u8 = var test_u8
// CHECK:STDOUT:   %test_u8: ref u8 = bind_name test_u8, %test_u8.var
// CHECK:STDOUT:   %.loc4_15.1: i32 = int_literal 16 [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_15.2: type = int_type unsigned, %.loc4_15.1 [template = constants.%.4]
// CHECK:STDOUT:   %test_u16.var: ref u16 = var test_u16
// CHECK:STDOUT:   %test_u16: ref u16 = bind_name test_u16, %test_u16.var
// CHECK:STDOUT:   %.loc5_15.1: i32 = int_literal 64 [template = constants.%.5]
// CHECK:STDOUT:   %.loc5_15.2: type = int_type unsigned, %.loc5_15.1 [template = constants.%.6]
// CHECK:STDOUT:   %test_u64.var: ref u64 = var test_u64
// CHECK:STDOUT:   %test_u64: ref u64 = bind_name test_u64, %test_u64.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_uN_bad_width.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.2: type = int_type unsigned, %.1 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 15 [template]
// CHECK:STDOUT:   %.4: type = int_type unsigned, %.3 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1000000000 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .test_u0 = %test_u0
// CHECK:STDOUT:     .test_u1 = %test_u1
// CHECK:STDOUT:     .test_u15 = %test_u15
// CHECK:STDOUT:     .test_u1000000000 = %test_u1000000000
// CHECK:STDOUT:     .test_u10000000000000000000 = %test_u10000000000000000000
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %u0.ref: <error> = name_ref u0, <error> [template = <error>]
// CHECK:STDOUT:   %test_u0.var: ref <error> = var test_u0
// CHECK:STDOUT:   %test_u0: ref <error> = bind_name test_u0, %test_u0.var
// CHECK:STDOUT:   %.loc12_14.1: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc12_14.2: type = int_type unsigned, %.loc12_14.1 [template = constants.%.2]
// CHECK:STDOUT:   %test_u1.var: ref u1 = var test_u1
// CHECK:STDOUT:   %test_u1: ref u1 = bind_name test_u1, %test_u1.var
// CHECK:STDOUT:   %.loc17_15.1: i32 = int_literal 15 [template = constants.%.3]
// CHECK:STDOUT:   %.loc17_15.2: type = int_type unsigned, %.loc17_15.1 [template = constants.%.4]
// CHECK:STDOUT:   %test_u15.var: ref u15 = var test_u15
// CHECK:STDOUT:   %test_u15: ref u15 = bind_name test_u15, %test_u15.var
// CHECK:STDOUT:   %.loc22_23.1: i32 = int_literal 1000000000 [template = constants.%.5]
// CHECK:STDOUT:   %.loc22_23.2: type = int_type unsigned, %.loc22_23.1 [template = <error>]
// CHECK:STDOUT:   %test_u1000000000.var: ref <error> = var test_u1000000000
// CHECK:STDOUT:   %test_u1000000000: ref <error> = bind_name test_u1000000000, %test_u1000000000.var
// CHECK:STDOUT:   %.loc28: type = int_type unsigned, <error> [template = <error>]
// CHECK:STDOUT:   %test_u10000000000000000000.var: ref <error> = var test_u10000000000000000000
// CHECK:STDOUT:   %test_u10000000000000000000: ref <error> = bind_name test_u10000000000000000000, %test_u10000000000000000000.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_fN_bad_width.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_fN_todo_unsupported.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
