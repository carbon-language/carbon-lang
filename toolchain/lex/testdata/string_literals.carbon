// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lex/testdata/string_literals.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lex/testdata/string_literals.carbon

// --- fail_decimal_escape.carbon
// CHECK:STDOUT: - filename: fail_decimal_escape.carbon
// CHECK:STDOUT:   tokens:

// CHECK:STDERR: fail_decimal_escape.carbon:[[@LINE+4]]:4: error: decimal digit follows `\0` escape sequence. Use `\x00` instead of `\0` if the next character is a digit [DecimalEscapeSequence]
// CHECK:STDERR: "\01"
// CHECK:STDERR:    ^
// CHECK:STDERR:
"\01"
// CHECK:STDOUT:   - { index: 1, kind: "StringLiteral", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "\"\\01\"", value: "\x001", has_leading_space: true }

// --- fail_unterminated.carbon
// CHECK:STDOUT: - filename: fail_unterminated.carbon
// CHECK:STDOUT:   tokens:

// CHECK:STDERR: fail_unterminated.carbon:[[@LINE+4]]:1: error: string is missing a terminator [UnterminatedString]
// CHECK:STDERR: "s
// CHECK:STDERR: ^
// CHECK:STDERR:
"s
// CHECK:STDOUT:   - { index: 1, kind:     "Error", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "\"s", has_leading_space: true }


// --- fail_unknown_escape.carbon
// CHECK:STDOUT: - filename: fail_unknown_escape.carbon
// CHECK:STDOUT:   tokens:

// CHECK:STDERR: fail_unknown_escape.carbon:[[@LINE+4]]:3: error: unrecognized escape sequence `w` [UnknownEscapeSequence]
// CHECK:STDERR: "\w"
// CHECK:STDERR:   ^
// CHECK:STDERR:
"\w"
// CHECK:STDOUT:   - { index: 1, kind: "StringLiteral", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "\"\\w\"", value: "w", has_leading_space: true }

// --- fail_unicode_escape_missing_braced_digits.carbon
// CHECK:STDOUT: - filename: fail_unicode_escape_missing_braced_digits.carbon
// CHECK:STDOUT:   tokens:

// CHECK:STDERR: fail_unicode_escape_missing_braced_digits.carbon:[[@LINE+4]]:4: error: escape sequence `\u` must be followed by a braced sequence of uppercase hexadecimal digits, for example `\u{70AD}}` [UnicodeEscapeMissingBracedDigits]
// CHECK:STDERR: "\u"
// CHECK:STDERR:    ^
// CHECK:STDERR:
"\u"
// CHECK:STDOUT:   - { index: 1, kind: "StringLiteral", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "\"\\u\"", value: "u", has_leading_space: true }

// --- fail_hex_escape_missing_digits.carbon
// CHECK:STDOUT: - filename: fail_hex_escape_missing_digits.carbon
// CHECK:STDOUT:   tokens:

// CHECK:STDERR: fail_hex_escape_missing_digits.carbon:[[@LINE+4]]:4: error: escape sequence `\x` must be followed by two uppercase hexadecimal digits, for example `\x0F` [HexadecimalEscapeMissingDigits]
// CHECK:STDERR: "\x"
// CHECK:STDERR:    ^
// CHECK:STDERR:
"\x"
// CHECK:STDOUT:   - { index: 1, kind: "StringLiteral", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "\"\\x\"", value: "x", has_leading_space: true }

// --- fail_literal_tab_in_string.carbon
// CHECK:STDOUT: - filename: fail_literal_tab_in_string.carbon
// CHECK:STDOUT:   tokens:

// CHECK:STDERR: fail_literal_tab_in_string.carbon:[[@LINE+4]]:2: error: whitespace other than plain space must be expressed with an escape sequence in a string literal [InvalidHorizontalWhitespaceInString]
// CHECK:STDERR: "{{\t}}"
// CHECK:STDERR:  ^
// CHECK:STDERR:
"	"
// CHECK:STDOUT:   - { index: 1, kind: "StringLiteral", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "\"\t\"", value: "\t", has_leading_space: true }

// --- fail_unicode_surrogate.carbon
// CHECK:STDOUT: - filename: fail_unicode_surrogate.carbon
// CHECK:STDOUT:   tokens:

// CHECK:STDERR: fail_unicode_surrogate.carbon:[[@LINE+4]]:5: error: code point specified by `\u{...}}` escape is a surrogate character [UnicodeEscapeSurrogate]
// CHECK:STDERR: "\u{D800}"
// CHECK:STDERR:     ^
// CHECK:STDERR:
"\u{D800}"
// CHECK:STDOUT:   - { index: 1, kind: "StringLiteral", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "\"\\u{D800}\"", value: "u{D800}", has_leading_space: true }

// --- fail_unicode_length.carbon
// CHECK:STDOUT: - filename: fail_unicode_length.carbon
// CHECK:STDOUT:   tokens:

// CHECK:STDERR: fail_unicode_length.carbon:[[@LINE+4]]:5: error: code point specified by `\u{...}}` escape is greater than 0x10FFFF [UnicodeEscapeTooLarge]
// CHECK:STDERR: "\u{FFFFFFFF}"
// CHECK:STDERR:     ^
// CHECK:STDERR:
"\u{FFFFFFFF}"
// CHECK:STDOUT:   - { index: 1, kind: "StringLiteral", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "\"\\u{FFFFFFFF}\"", value: "u{FFFFFFFF}", has_leading_space: true }
