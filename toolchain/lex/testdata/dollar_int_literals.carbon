// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lex/testdata/dollar_int_literals.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lex/testdata/dollar_int_literals.carbon

// --- valid.carbon
// CHECK:STDOUT: - filename: valid.carbon
// CHECK:STDOUT:   tokens:
$0
// CHECK:STDOUT:   - { index: 1, kind: "DollarIntLiteral", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "$0", has_leading_space: true }
$1
// CHECK:STDOUT:   - { index: 2, kind: "DollarIntLiteral", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "$1", has_leading_space: true }
$42
// CHECK:STDOUT:   - { index: 3, kind: "DollarIntLiteral", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "$42", has_leading_space: true }
$100
// CHECK:STDOUT:   - { index: 4, kind: "DollarIntLiteral", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "$100", has_leading_space: true }
$999999999999
// CHECK:STDOUT:   - { index: 5, kind: "DollarIntLiteral", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "$999999999999", has_leading_space: true }

// --- fail_invalid.carbon
// CHECK:STDOUT: - filename: fail_invalid.carbon
// CHECK:STDOUT:   tokens:

// CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:2: error: Expected digits after `$` [DollarIntLiteralMissingNumber]
// CHECK:STDERR: $
// CHECK:STDERR:  ^
// CHECK:STDERR:
$
// CHECK:STDOUT:   - { index:  1, kind:      "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "$", has_leading_space: true }
// CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:2: error: Expected digits after `$` [DollarIntLiteralMissingNumber]
// CHECK:STDERR: $ 1
// CHECK:STDERR:  ^
// CHECK:STDERR:
$ 1
// CHECK:STDOUT:   - { index:  2, kind:      "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "$", has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind: "IntLiteral", line: {{ *}}[[@LINE-2]], column:   3, indent: 1, spelling: "1", value: "1", has_leading_space: true }
// CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:2: error: Positional parameters can only contain digits after `$` [InvalidCharacterInDollarIntLiteral]
// CHECK:STDERR: $foo
// CHECK:STDERR:  ^
// CHECK:STDERR:
$foo
// CHECK:STDOUT:   - { index:  4, kind:      "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "$foo", has_leading_space: true }
// CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:2: error: Positional parameters can only contain digits after `$` [InvalidCharacterInDollarIntLiteral]
// CHECK:STDERR: $1a32
// CHECK:STDERR:  ^
// CHECK:STDERR:
$1a32
// CHECK:STDOUT:   - { index:  5, kind:      "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "$1a32", has_leading_space: true }
// CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:2: error: Leading zeroes are not allowed in positional parameters [DollarIntLiteralLeadingZero]
// CHECK:STDERR: $00
// CHECK:STDERR:  ^
// CHECK:STDERR:
$00
// CHECK:STDOUT:   - { index:  6, kind:      "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "$00", has_leading_space: true }
// CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:2: error: Leading zeroes are not allowed in positional parameters [DollarIntLiteralLeadingZero]
// CHECK:STDERR: $025321
// CHECK:STDERR:  ^
// CHECK:STDERR:
$025321
// CHECK:STDOUT:   - { index:  7, kind:      "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "$025321", has_leading_space: true }
// CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:2: error: Leading zeroes are not allowed in positional parameters [DollarIntLiteralLeadingZero]
// CHECK:STDERR: $0x12
// CHECK:STDERR:  ^
// CHECK:STDERR:
$0x12
// CHECK:STDOUT:   - { index:  8, kind:      "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "$0x12", has_leading_space: true }
// CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:2: error: Leading zeroes are not allowed in positional parameters [DollarIntLiteralLeadingZero]
// CHECK:STDERR: $0b1100
// CHECK:STDERR:  ^
// CHECK:STDERR:
$0b1100
// CHECK:STDOUT:   - { index:  9, kind:      "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "$0b1100", has_leading_space: true }
// CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:2: error: invalid digit '$' in decimal numeric literal [InvalidDigit]
// CHECK:STDERR: 2$63
// CHECK:STDERR:  ^
// CHECK:STDERR:
2$63
// CHECK:STDOUT:   - { index: 10, kind:      "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "2$63", has_leading_space: true }
// CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:4: error: `$` is only allowed at the start of a positional parameter [CharacterOnlyAllowedAtStart]
// CHECK:STDERR: abc$37
// CHECK:STDERR:    ^
// CHECK:STDERR:
abc$37
// CHECK:STDOUT:   - { index: 11, kind:      "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "abc$37" }
// CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:6: error: `$` is only allowed at the start of a positional parameter [CharacterOnlyAllowedAtStart]
// CHECK:STDERR: r#foo$0
// CHECK:STDERR:      ^
// CHECK:STDERR:
r#foo$0
// CHECK:STDOUT:   - { index: 12, kind:      "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "r#foo$0" }
// CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:2: error: found a positional parameter using 19 digits, which is greater than the limit of 18 [TooManyDollarIntDigits]
// CHECK:STDERR: $9999999999999999999
// CHECK:STDERR:  ^
// CHECK:STDERR:
$9999999999999999999
// CHECK:STDOUT:   - { index: 13, kind:      "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "$9999999999999999999", has_leading_space: true }
