// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/primitives/numeric_literals.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/primitives/numeric_literals.carbon

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

// --- very_large_mantissa.carbon

library "[[@TEST_NAME]]";

let large: f64 = 99999999999999999999999999999999999999999999999999999999999999999999999.0e237;
let max_trunc: f64 = 17976931348623157081452742373170435679807056752584499659891747680315726.0e238;
let max: f64 = 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.0;

let max_hex: f64 = 0x1.FFFFFFFFFFFFFp+1023;

// The largest number that wouldn't round to have an out-of-bounds exponent.
// This is 2^1024 - 2^54 - epsilon.
// TODO: The design says that we should treat this as an error. Decide if that's
// right or if we want to round before checking for overflow.
let max_rounds_down: f64 = 179769313486231580793728971405303415079934132710037826936173778980444968292764750946649017977587207096330286416692887910946555547851940402630657488671505820681908902000708383676273854845817711531764475730270069855571366959622842914819860834936475292719074168444365510704342711559699508093042880177904174497791.9999999999999;
let max_rounds_down_trunc: f64 = 17976931348623158079372897140530341507993413271003782693617377898044496.8e238;

let max_rounds_down_hex: f64 = 0x1.FFFFFFFFFFFFF7FFFFFFFFFFp+1023;

// --- fail_overflow_very_large_mantissa.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_overflow_very_large_mantissa.carbon:[[@LINE+4]]:22: error: value 399999999999999999999999999999999999999999999999999999999999999999999990*10^237 too large for floating-point type `f64` [FloatTooLargeForType]
// CHECK:STDERR: let too_large: f64 = 39999999999999999999999999999999999999999999999999999999999999999999999.0e238;
// CHECK:STDERR:                      ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let too_large: f64 = 39999999999999999999999999999999999999999999999999999999999999999999999.0e238;

// TODO: These are larger than DBL_MAX but would still round to DBL_MAX if f64
// had a wider exponent field. The design says these are invalid.
let above_max_trunc: f64 = 17976931348623157081452742373170435679807056752584499659891747680315727.0e238;
let max_plus_one: f64 = 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858369.0;
let max_plus_a_bit_hex: f64 = 0x1.FFFFFFFFFFFFF00000000000000000000001p+1023;

// The smallest number that would round to having an over-large exponent.
// CHECK:STDERR: fail_overflow_very_large_mantissa.carbon:[[@LINE+4]]:30: error: value 1797693134862315807937289714053034150799341327100378269361737789804449682927647509466490179775872070963302864166928879109465555478519404026306574886715058206819089020007083836762738548458177115317644757302700698555713669596228429148198608349364752927190741684443655107043427115596995080930428801779041744977920*10^-1 too large for floating-point type `f64` [FloatTooLargeForType]
// CHECK:STDERR: let min_rounds_to_inf: f64 = 179769313486231580793728971405303415079934132710037826936173778980444968292764750946649017977587207096330286416692887910946555547851940402630657488671505820681908902000708383676273854845817711531764475730270069855571366959622842914819860834936475292719074168444365510704342711559699508093042880177904174497792.0;
// CHECK:STDERR:                              ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let min_rounds_to_inf: f64 = 179769313486231580793728971405303415079934132710037826936173778980444968292764750946649017977587207096330286416692887910946555547851940402630657488671505820681908902000708383676273854845817711531764475730270069855571366959622842914819860834936475292719074168444365510704342711559699508093042880177904174497792.0;

// CHECK:STDERR: fail_overflow_very_large_mantissa.carbon:[[@LINE+4]]:41: error: value 179769313486231580793728971405303415079934132710037826936173778980444969*10^237 too large for floating-point type `f64` [FloatTooLargeForType]
// CHECK:STDERR: let min_rounds_to_inf_rounded_up: f64 = 17976931348623158079372897140530341507993413271003782693617377898044496.9e238;
// CHECK:STDERR:                                         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let min_rounds_to_inf_rounded_up: f64 = 17976931348623158079372897140530341507993413271003782693617377898044496.9e238;

// CHECK:STDERR: fail_overflow_very_large_mantissa.carbon:[[@LINE+4]]:34: error: value 144115188075855864*2^967 too large for floating-point type `f64` [FloatTooLargeForType]
// CHECK:STDERR: let min_rounds_to_inf_hex: f64 = 0x1.FFFFFFFFFFFFF8p+1023;
// CHECK:STDERR:                                  ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let min_rounds_to_inf_hex: f64 = 0x1.FFFFFFFFFFFFF8p+1023;

// --- fail_overflow_very_large_exponent.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_overflow_very_large_exponent.carbon:[[@LINE+4]]:14: error: value 50*10^39999999999999999992 too large for floating-point type `f64` [FloatTooLargeForType]
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
// CHECK:STDOUT:   %float.7d0: Core.FloatLiteral = float_literal_value 9e-1 [concrete]
// CHECK:STDOUT:   %float.4bc: Core.FloatLiteral = float_literal_value 80e-1 [concrete]
// CHECK:STDOUT:   %float.24e: Core.FloatLiteral = float_literal_value 800e-1 [concrete]
// CHECK:STDOUT:   %float.4d9: Core.FloatLiteral = float_literal_value 10e6 [concrete]
// CHECK:STDOUT:   %float.150: Core.FloatLiteral = float_literal_value 10e7 [concrete]
// CHECK:STDOUT:   %float.07e: Core.FloatLiteral = float_literal_value 10e-9 [concrete]
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
// CHECK:STDOUT:   %float.loc19: Core.FloatLiteral = float_literal_value 9e-1 [concrete = constants.%float.7d0]
// CHECK:STDOUT:   %float.loc20: Core.FloatLiteral = float_literal_value 80e-1 [concrete = constants.%float.4bc]
// CHECK:STDOUT:   %float.loc21: Core.FloatLiteral = float_literal_value 800e-1 [concrete = constants.%float.24e]
// CHECK:STDOUT:   %float.loc22: Core.FloatLiteral = float_literal_value 10e6 [concrete = constants.%float.4d9]
// CHECK:STDOUT:   %float.loc23: Core.FloatLiteral = float_literal_value 10e7 [concrete = constants.%float.150]
// CHECK:STDOUT:   %float.loc24: Core.FloatLiteral = float_literal_value 10e-9 [concrete = constants.%float.07e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
