// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/numeric_literals.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/numeric_literals.carbon

// --- literal_values.carbon

library "[[@TEST_NAME]]";

fn F() {
  // 8 and 9 trigger special behavior in APInt when mishandling signed versus
  // unsigned, so we pay extra attention to those.
  var ints: array(i32, 6) = (
    //@dump-sem-ir-begin
    8,
    9,
    0x8,
    0b1000,
    2147483647,
    0x7FFFFFFF,
    //@dump-sem-ir-end
  );
  var floats: array(f64, 6) = (
    //@dump-sem-ir-begin
    0.9,
    8.0,
    80.0,
    1.0e7,
    1.0e8,
    1.0e-8
    //@dump-sem-ir-end
  );
}

// --- fail_overflow_very_large.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_overflow_very_large.carbon:[[@LINE+4]]:14: error: integer value 39999999999999999993 too large for type `i32` [IntTooLargeForType]
// CHECK:STDERR: let a: i32 = 39999999999999999993;
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let a: i32 = 39999999999999999993;

// --- fail_overflow_boundary.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_overflow_boundary.carbon:[[@LINE+4]]:14: error: integer value 2147483648 too large for type `i32` [IntTooLargeForType]
// CHECK:STDERR: let b: i32 = 2_147_483_648;
// CHECK:STDERR:              ^~~~~~~~~~~~~
// CHECK:STDERR:
let b: i32 = 2_147_483_648;

// --- fail_overflow_boundary_hex.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_overflow_boundary_hex.carbon:[[@LINE+4]]:14: error: integer value 2147483648 too large for type `i32` [IntTooLargeForType]
// CHECK:STDERR: let c: i32 = 0x8000_0000;
// CHECK:STDERR:              ^~~~~~~~~~~
// CHECK:STDERR:
let c: i32 = 0x8000_0000;

// --- fail_overflow_very_large_mantissa.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_overflow_very_large_mantissa.carbon:[[@LINE+4]]:14: error: real mantissa with value 399999999999999999930 does not fit in i64 [RealMantissaTooLargeForI64]
// CHECK:STDERR: let d: f64 = 39999999999999999993.0e3;
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let d: f64 = 39999999999999999993.0e3;

// --- fail_overflow_very_large_exponent.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_overflow_very_large_exponent.carbon:[[@LINE+4]]:14: error: real exponent with value 39999999999999999992 does not fit in i64 [RealExponentTooLargeForI64]
// CHECK:STDERR: let e: f64 = 5.0e39999999999999999993;
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let e: f64 = 5.0e39999999999999999993;

// CHECK:STDOUT: --- literal_values.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_8.b85: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %int_9.988: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %int_2147483647.d89: Core.IntLiteral = int_value 2147483647 [concrete]
// CHECK:STDOUT:   %float.8ca: Core.FloatLiteral = float_value 0.90000000000000002 [concrete]
// CHECK:STDOUT:   %float.fde: Core.FloatLiteral = float_value 8 [concrete]
// CHECK:STDOUT:   %float.7a6: Core.FloatLiteral = float_value 80 [concrete]
// CHECK:STDOUT:   %float.23b: Core.FloatLiteral = float_value 1.0E+7 [concrete]
// CHECK:STDOUT:   %float.b98: Core.FloatLiteral = float_value 1.0E+8 [concrete]
// CHECK:STDOUT:   %float.34d: Core.FloatLiteral = float_value 1.0E-8 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %int_8.loc9: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9.988]
// CHECK:STDOUT:   %int_8.loc11: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %int_8.loc12: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %int_2147483647.loc13: Core.IntLiteral = int_value 2147483647 [concrete = constants.%int_2147483647.d89]
// CHECK:STDOUT:   %int_2147483647.loc14: Core.IntLiteral = int_value 2147483647 [concrete = constants.%int_2147483647.d89]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %float.loc19: Core.FloatLiteral = float_value 0.90000000000000002 [concrete = constants.%float.8ca]
// CHECK:STDOUT:   %float.loc20: Core.FloatLiteral = float_value 8 [concrete = constants.%float.fde]
// CHECK:STDOUT:   %float.loc21: Core.FloatLiteral = float_value 80 [concrete = constants.%float.7a6]
// CHECK:STDOUT:   %float.loc22: Core.FloatLiteral = float_value 1.0E+7 [concrete = constants.%float.23b]
// CHECK:STDOUT:   %float.loc23: Core.FloatLiteral = float_value 1.0E+8 [concrete = constants.%float.b98]
// CHECK:STDOUT:   %float.loc24: Core.FloatLiteral = float_value 1.0E-8 [concrete = constants.%float.34d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
