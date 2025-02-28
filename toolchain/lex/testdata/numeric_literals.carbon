// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lex/testdata/numeric_literals.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lex/testdata/numeric_literals.carbon

// --- valid.carbon
// CHECK:STDOUT: - filename: valid.carbon
// CHECK:STDOUT:   tokens:


fn F() {
// CHECK:STDOUT:   - { index:  1, kind:               "Fn", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:       "Identifier", line: {{ *}}[[@LINE-2]], column:  4, indent: 1, spelling: "F", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:        "OpenParen", line: {{ *}}[[@LINE-3]], column:  5, indent: 1, spelling: "(", closing_token: 4 }
// CHECK:STDOUT:   - { index:  4, kind:       "CloseParen", line: {{ *}}[[@LINE-4]], column:  6, indent: 1, spelling: ")", opening_token: 3 }
// CHECK:STDOUT:   - { index:  5, kind:   "OpenCurlyBrace", line: {{ *}}[[@LINE-5]], column:  8, indent: 1, spelling: "{", closing_token: 56, has_leading_space: true }
  // 8 and 9 trigger special behavior in APInt when mishandling signed versus
  // unsigned, so we pay extra attention to those.
  var ints: array(i32, 5) = (
  // CHECK:STDOUT:   - { index:  6, kind:              "Var", line: {{ *}}[[@LINE-1]], column:  3, indent: 3, spelling: "var", has_leading_space: true }
  // CHECK:STDOUT:   - { index:  7, kind:       "Identifier", line: {{ *}}[[@LINE-2]], column:  7, indent: 3, spelling: "ints", identifier: 1, has_leading_space: true }
  // CHECK:STDOUT:   - { index:  8, kind:            "Colon", line: {{ *}}[[@LINE-3]], column: 11, indent: 3, spelling: ":" }
  // CHECK:STDOUT:   - { index:  9, kind:            "Array", line: {{ *}}[[@LINE-4]], column: 13, indent: 3, spelling: "array", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 10, kind:        "OpenParen", line: {{ *}}[[@LINE-5]], column: 18, indent: 3, spelling: "(", closing_token: 14 }
  // CHECK:STDOUT:   - { index: 11, kind:   "IntTypeLiteral", line: {{ *}}[[@LINE-6]], column: 19, indent: 3, spelling: "i32" }
  // CHECK:STDOUT:   - { index: 12, kind:            "Comma", line: {{ *}}[[@LINE-7]], column: 22, indent: 3, spelling: "," }
  // CHECK:STDOUT:   - { index: 13, kind:       "IntLiteral", line: {{ *}}[[@LINE-8]], column: 24, indent: 3, spelling: "5", value: "5", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 14, kind:       "CloseParen", line: {{ *}}[[@LINE-9]], column: 25, indent: 3, spelling: ")", opening_token: 10 }
  // CHECK:STDOUT:   - { index: 15, kind:            "Equal", line: {{ *}}[[@LINE-10]], column: 27, indent: 3, spelling: "=", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 16, kind:        "OpenParen", line: {{ *}}[[@LINE-11]], column: 29, indent: 3, spelling: "(", closing_token: 27, has_leading_space: true }
    8,
    // CHECK:STDOUT:   - { index: 17, kind:       "IntLiteral", line: {{ *}}[[@LINE-1]], column:  5, indent: 5, spelling: "8", value: "8", has_leading_space: true }
    // CHECK:STDOUT:   - { index: 18, kind:            "Comma", line: {{ *}}[[@LINE-2]], column:  6, indent: 5, spelling: "," }
    9,
    // CHECK:STDOUT:   - { index: 19, kind:       "IntLiteral", line: {{ *}}[[@LINE-1]], column:  5, indent: 5, spelling: "9", value: "9", has_leading_space: true }
    // CHECK:STDOUT:   - { index: 20, kind:            "Comma", line: {{ *}}[[@LINE-2]], column:  6, indent: 5, spelling: "," }
    0x8,
    // CHECK:STDOUT:   - { index: 21, kind:       "IntLiteral", line: {{ *}}[[@LINE-1]], column:  5, indent: 5, spelling: "0x8", value: "8", has_leading_space: true }
    // CHECK:STDOUT:   - { index: 22, kind:            "Comma", line: {{ *}}[[@LINE-2]], column:  8, indent: 5, spelling: "," }
    0b1000,
    // CHECK:STDOUT:   - { index: 23, kind:       "IntLiteral", line: {{ *}}[[@LINE-1]], column:  5, indent: 5, spelling: "0b1000", value: "8", has_leading_space: true }
    // CHECK:STDOUT:   - { index: 24, kind:            "Comma", line: {{ *}}[[@LINE-2]], column: 11, indent: 5, spelling: "," }
    39999999999999999993,
    // CHECK:STDOUT:   - { index: 25, kind:       "IntLiteral", line: {{ *}}[[@LINE-1]], column:  5, indent: 5, spelling: "39999999999999999993", value: "39999999999999999993", has_leading_space: true }
    // CHECK:STDOUT:   - { index: 26, kind:            "Comma", line: {{ *}}[[@LINE-2]], column: 25, indent: 5, spelling: "," }
  );
  // CHECK:STDOUT:   - { index: 27, kind:       "CloseParen", line: {{ *}}[[@LINE-1]], column:  3, indent: 3, spelling: ")", opening_token: 16, has_leading_space: true }
  // CHECK:STDOUT:   - { index: 28, kind:             "Semi", line: {{ *}}[[@LINE-2]], column:  4, indent: 3, spelling: ";" }
  var floats: array(f64, 7) = (
  // CHECK:STDOUT:   - { index: 29, kind:              "Var", line: {{ *}}[[@LINE-1]], column:  3, indent: 3, spelling: "var", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 30, kind:       "Identifier", line: {{ *}}[[@LINE-2]], column:  7, indent: 3, spelling: "floats", identifier: 2, has_leading_space: true }
  // CHECK:STDOUT:   - { index: 31, kind:            "Colon", line: {{ *}}[[@LINE-3]], column: 13, indent: 3, spelling: ":" }
  // CHECK:STDOUT:   - { index: 32, kind:            "Array", line: {{ *}}[[@LINE-4]], column: 15, indent: 3, spelling: "array", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 33, kind:        "OpenParen", line: {{ *}}[[@LINE-5]], column: 20, indent: 3, spelling: "(", closing_token: 37 }
  // CHECK:STDOUT:   - { index: 34, kind: "FloatTypeLiteral", line: {{ *}}[[@LINE-6]], column: 21, indent: 3, spelling: "f64" }
  // CHECK:STDOUT:   - { index: 35, kind:            "Comma", line: {{ *}}[[@LINE-7]], column: 24, indent: 3, spelling: "," }
  // CHECK:STDOUT:   - { index: 36, kind:       "IntLiteral", line: {{ *}}[[@LINE-8]], column: 26, indent: 3, spelling: "7", value: "7", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 37, kind:       "CloseParen", line: {{ *}}[[@LINE-9]], column: 27, indent: 3, spelling: ")", opening_token: 33 }
  // CHECK:STDOUT:   - { index: 38, kind:            "Equal", line: {{ *}}[[@LINE-10]], column: 29, indent: 3, spelling: "=", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 39, kind:        "OpenParen", line: {{ *}}[[@LINE-11]], column: 31, indent: 3, spelling: "(", closing_token: 54, has_leading_space: true }
    0.9,
    // CHECK:STDOUT:   - { index: 40, kind:      "RealLiteral", line: {{ *}}[[@LINE-1]], column:  5, indent: 5, spelling: "0.9", value: "9*10^-1", has_leading_space: true }
    // CHECK:STDOUT:   - { index: 41, kind:            "Comma", line: {{ *}}[[@LINE-2]], column:  8, indent: 5, spelling: "," }
    8.0,
    // CHECK:STDOUT:   - { index: 42, kind:      "RealLiteral", line: {{ *}}[[@LINE-1]], column:  5, indent: 5, spelling: "8.0", value: "80*10^-1", has_leading_space: true }
    // CHECK:STDOUT:   - { index: 43, kind:            "Comma", line: {{ *}}[[@LINE-2]], column:  8, indent: 5, spelling: "," }
    80.0,
    // CHECK:STDOUT:   - { index: 44, kind:      "RealLiteral", line: {{ *}}[[@LINE-1]], column:  5, indent: 5, spelling: "80.0", value: "800*10^-1", has_leading_space: true }
    // CHECK:STDOUT:   - { index: 45, kind:            "Comma", line: {{ *}}[[@LINE-2]], column:  9, indent: 5, spelling: "," }
    1.0e7,
    // CHECK:STDOUT:   - { index: 46, kind:      "RealLiteral", line: {{ *}}[[@LINE-1]], column:  5, indent: 5, spelling: "1.0e7", value: "10*10^6", has_leading_space: true }
    // CHECK:STDOUT:   - { index: 47, kind:            "Comma", line: {{ *}}[[@LINE-2]], column: 10, indent: 5, spelling: "," }
    1.0e8,
    // CHECK:STDOUT:   - { index: 48, kind:      "RealLiteral", line: {{ *}}[[@LINE-1]], column:  5, indent: 5, spelling: "1.0e8", value: "10*10^7", has_leading_space: true }
    // CHECK:STDOUT:   - { index: 49, kind:            "Comma", line: {{ *}}[[@LINE-2]], column: 10, indent: 5, spelling: "," }
    1.0e-8,
    // CHECK:STDOUT:   - { index: 50, kind:      "RealLiteral", line: {{ *}}[[@LINE-1]], column:  5, indent: 5, spelling: "1.0e-8", value: "10*10^-9", has_leading_space: true }
    // CHECK:STDOUT:   - { index: 51, kind:            "Comma", line: {{ *}}[[@LINE-2]], column: 11, indent: 5, spelling: "," }
    39999999999999999993.0e39999999999999999993,
    // CHECK:STDOUT:   - { index: 52, kind:      "RealLiteral", line: {{ *}}[[@LINE-1]], column:  5, indent: 5, spelling: "39999999999999999993.0e39999999999999999993", value: "399999999999999999930*10^39999999999999999992", has_leading_space: true }
    // CHECK:STDOUT:   - { index: 53, kind:            "Comma", line: {{ *}}[[@LINE-2]], column: 48, indent: 5, spelling: "," }
  );
  // CHECK:STDOUT:   - { index: 54, kind:       "CloseParen", line: {{ *}}[[@LINE-1]], column:  3, indent: 3, spelling: ")", opening_token: 39, has_leading_space: true }
  // CHECK:STDOUT:   - { index: 55, kind:             "Semi", line: {{ *}}[[@LINE-2]], column:  4, indent: 3, spelling: ";" }
}
// CHECK:STDOUT:   - { index: 56, kind:  "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "}", opening_token: 5, has_leading_space: true }


// --- fail_binary_real.carbon
// CHECK:STDOUT: - filename: fail_binary_real.carbon
// CHECK:STDOUT:   tokens:

// CHECK:STDERR: fail_binary_real.carbon:[[@LINE+4]]:4: error: binary real number literals are not supported [BinaryRealLiteral]
// CHECK:STDERR: 0b1.0
// CHECK:STDERR:    ^
// CHECK:STDERR:
0b1.0
// CHECK:STDOUT:   - { index: 1, kind: "RealLiteral", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "0b1.0", value: "2*2^-1", has_leading_space: true }

// --- fail_wrong_real_exponent.carbon
// CHECK:STDOUT: - filename: fail_wrong_real_exponent.carbon
// CHECK:STDOUT:   tokens:

// CHECK:STDERR: fail_wrong_real_exponent.carbon:[[@LINE+4]]:4: error: expected 'e' to introduce exponent [WrongRealLiteralExponent]
// CHECK:STDERR: 1.0r3
// CHECK:STDERR:    ^
// CHECK:STDERR:
1.0r3
// CHECK:STDOUT:   - { index: 1, kind:     "Error", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "1.0r3", has_leading_space: true }

// --- fail_invalid_digit.carbon
// CHECK:STDOUT: - filename: fail_invalid_digit.carbon
// CHECK:STDOUT:   tokens:

// CHECK:STDERR: fail_invalid_digit.carbon:[[@LINE+4]]:4: error: invalid digit 'a' in hexadecimal numeric literal [InvalidDigit]
// CHECK:STDERR: 0x1a
// CHECK:STDERR:    ^
// CHECK:STDERR:
0x1a
// CHECK:STDOUT:   - { index: 1, kind:     "Error", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "0x1a", has_leading_space: true }

// --- fail_invalid_separator.carbon
// CHECK:STDOUT: - filename: fail_invalid_separator.carbon
// CHECK:STDOUT:   tokens:

// CHECK:STDERR: fail_invalid_separator.carbon:[[@LINE+4]]:2: error: misplaced digit separator in numeric literal [InvalidDigitSeparator]
// CHECK:STDERR: 1__2
// CHECK:STDERR:  ^
// CHECK:STDERR:
1__2
// CHECK:STDOUT:   - { index: 1, kind: "IntLiteral", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "1__2", value: "12", has_leading_space: true }

// --- fail_unknown_base.carbon
// CHECK:STDOUT: - filename: fail_unknown_base.carbon
// CHECK:STDOUT:   tokens:

// CHECK:STDERR: fail_unknown_base.carbon:[[@LINE+4]]:1: error: unknown base specifier in numeric literal [UnknownBaseSpecifier]
// CHECK:STDERR: 05
// CHECK:STDERR: ^
// CHECK:STDERR:
05
// CHECK:STDOUT:   - { index: 1, kind:     "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "05", has_leading_space: true }
