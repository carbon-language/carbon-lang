// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_numeric_literal_overflow.carbon:[[@LINE+4]]:14: ERROR: Integer literal with value 39999999999999999993 does not fit in i32.
// CHECK:STDERR: let a: i32 = 39999999999999999993;
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let a: i32 = 39999999999999999993;

// CHECK:STDERR: fail_numeric_literal_overflow.carbon:[[@LINE+4]]:14: ERROR: Integer literal with value 2147483648 does not fit in i32.
// CHECK:STDERR: let b: i32 = 2_147_483_648;
// CHECK:STDERR:              ^~~~~~~~~~~~~
// CHECK:STDERR:
let b: i32 = 2_147_483_648;

// CHECK:STDERR: fail_numeric_literal_overflow.carbon:[[@LINE+4]]:14: ERROR: Integer literal with value 2147483648 does not fit in i32.
// CHECK:STDERR: let c: i32 = 0x8000_0000;
// CHECK:STDERR:              ^~~~~~~~~~~
// CHECK:STDERR:
let c: i32 = 0x8000_0000;

// CHECK:STDERR: fail_numeric_literal_overflow.carbon:[[@LINE+4]]:14: ERROR: Real mantissa with value 399999999999999999930 does not fit in i64.
// CHECK:STDERR: let d: f64 = 39999999999999999993.0e3;
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let d: f64 = 39999999999999999993.0e3;

// CHECK:STDERR: fail_numeric_literal_overflow.carbon:[[@LINE+3]]:14: ERROR: Real exponent with value 39999999999999999992 does not fit in i64.
// CHECK:STDERR: let e: f64 = 5.0e39999999999999999993;
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~~~
let e: f64 = 5.0e39999999999999999993;

// CHECK:STDOUT: --- fail_numeric_literal_overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %a: i32 = bind_name a, <error>
// CHECK:STDOUT:   %b: i32 = bind_name b, <error>
// CHECK:STDOUT:   %c: i32 = bind_name c, <error>
// CHECK:STDOUT:   %d: f64 = bind_name d, <error>
// CHECK:STDOUT:   %e: f64 = bind_name e, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
