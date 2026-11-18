// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lex/testdata/fail_malformed_escapes.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lex/testdata/fail_malformed_escapes.carbon

// Each diagnostic is located where the escape stops being well-formed, and
// marks the escape up to there.

// --- fail_hex_escape_wrong_digit.carbon
// CHECK:STDOUT: - filename: fail_hex_escape_wrong_digit.carbon
// CHECK:STDOUT:   tokens:

// CHECK:STDERR:        ╭─┤ fail_hex_escape_wrong_digit.carbon:[[@LINE+8]]:4
// CHECK:STDERR: ->     │ "\xG1"
// CHECK:STDERR:        ·  ─╴╻
// CHECK:STDERR:        ·    ┗━━┫ error: escape sequence `\x` must be followed by two uppercase
// CHECK:STDERR:        ·       ┃        hexadecimal digits, for example `\x0F`
// CHECK:STDERR:        ·       ┃        [HexadecimalEscapeMissingDigits]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
"\xG1"
// CHECK:STDOUT:   - { index: 1, kind: "StringLiteral", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "\"\\xG1\"", value: "xG1", has_leading_space: true }

// --- fail_hex_escape_one_digit.carbon
// CHECK:STDOUT: - filename: fail_hex_escape_one_digit.carbon
// CHECK:STDOUT:   tokens:

// CHECK:STDERR:        ╭─┤ fail_hex_escape_one_digit.carbon:[[@LINE+8]]:5
// CHECK:STDERR: ->     │ "\x0g"
// CHECK:STDERR:        ·  ──╴╻
// CHECK:STDERR:        ·     ┗━━┫ error: escape sequence `\x` must be followed by two uppercase
// CHECK:STDERR:        ·        ┃        hexadecimal digits, for example `\x0F`
// CHECK:STDERR:        ·        ┃        [HexadecimalEscapeMissingDigits]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
"\x0g"
// CHECK:STDOUT:   - { index: 1, kind: "StringLiteral", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "\"\\x0g\"", value: "x0g", has_leading_space: true }

// --- fail_unicode_escape_empty_braces.carbon
// CHECK:STDOUT: - filename: fail_unicode_escape_empty_braces.carbon
// CHECK:STDOUT:   tokens:

// CHECK:STDERR:        ╭─┤ fail_unicode_escape_empty_braces.carbon:[[@LINE+8]]:5
// CHECK:STDERR: ->     │ "\u{}"
// CHECK:STDERR:        ·  ──╴╻
// CHECK:STDERR:        ·     ┗━━┫ error: escape sequence `\u` must be followed by a braced sequence of
// CHECK:STDERR:        ·        ┃        uppercase hexadecimal digits, for example `\u{70AD}}`
// CHECK:STDERR:        ·        ┃        [UnicodeEscapeMissingBracedDigits]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
"\u{}"
// CHECK:STDOUT:   - { index: 1, kind: "StringLiteral", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "\"\\u{}\"", value: "u{}", has_leading_space: true }

// --- fail_unicode_escape_unclosed.carbon
// CHECK:STDOUT: - filename: fail_unicode_escape_unclosed.carbon
// CHECK:STDOUT:   tokens:

// CHECK:STDERR:        ╭─┤ fail_unicode_escape_unclosed.carbon:[[@LINE+8]]:7
// CHECK:STDERR: ->     │ "\u{12"
// CHECK:STDERR:        ·  ────╴╻
// CHECK:STDERR:        ·       ┗━━┫ error: escape sequence `\u` must be followed by a braced sequence of
// CHECK:STDERR:        ·          ┃        uppercase hexadecimal digits, for example `\u{70AD}}`
// CHECK:STDERR:        ·          ┃        [UnicodeEscapeMissingBracedDigits]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
"\u{12"
// CHECK:STDOUT:   - { index: 1, kind: "StringLiteral", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "\"\\u{12\"", value: "u{12", has_leading_space: true }
