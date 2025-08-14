// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lex/testdata/char_literals.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lex/testdata/char_literals.carbon

// --- valid.carbon
// CHECK:STDOUT: - filename: valid.carbon
// CHECK:STDOUT:   tokens:

'a'
// CHECK:STDOUT:   - { index:  1, kind: "CharLiteral", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "'a'", has_leading_space: true }
'\n'
// CHECK:STDOUT:   - { index:  2, kind: "CharLiteral", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "'\\n'", has_leading_space: true }
'\0'
// CHECK:STDOUT:   - { index:  3, kind: "CharLiteral", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "'\\0'", has_leading_space: true }
'\u{00}'
// CHECK:STDOUT:   - { index:  4, kind: "CharLiteral", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "'\\u{00}'", has_leading_space: true }
'\u{1F}'
// CHECK:STDOUT:   - { index:  5, kind: "CharLiteral", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "'\\u{1F}'", has_leading_space: true }
'\u{20}'
// CHECK:STDOUT:   - { index:  6, kind: "CharLiteral", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "'\\u{20}'", has_leading_space: true }
'\u{7F}'
// CHECK:STDOUT:   - { index:  7, kind: "CharLiteral", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "'\\u{7F}'", has_leading_space: true }
'\u{123}'
// CHECK:STDOUT:   - { index:  8, kind: "CharLiteral", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "'\\u{123}'", has_leading_space: true }

// --- fail_invalid.carbon
// CHECK:STDOUT: - filename: fail_invalid.carbon
// CHECK:STDOUT:   tokens:

// CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:1: error: escape sequence `\x` in character literal; specify as escape sequence `\u{70}` [CharLiteralHexEscape]
// CHECK:STDERR: '\x70'
// CHECK:STDERR: ^
// CHECK:STDERR:
'\x70'
// CHECK:STDOUT:   - { index:  1, kind:       "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "'\\x70'", has_leading_space: true }

// CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:1: error: incomplete UTF-8 [CharLiteralUnderflow]
// CHECK:STDERR: '\xC3'
// CHECK:STDERR: ^
// CHECK:STDERR:
'\xC3'
// CHECK:STDOUT:   - { index:  2, kind:       "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "'\\xC3'", has_leading_space: true }

// CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:1: error: escape sequence `\x` in character literal; specify as escape sequence `\u{E9}` [CharLiteralHexEscape]
// CHECK:STDERR: '\xC3\xA9'
// CHECK:STDERR: ^
// CHECK:STDERR:
'\xC3\xA9'
// CHECK:STDOUT:   - { index:  3, kind:       "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "'\\xC3\\xA9'", has_leading_space: true }

// CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:1: error: invalid UTF-8 character [CharLiteralInvalidUTF8]
// CHECK:STDERR: '\xC3\xFF'
// CHECK:STDERR: ^
// CHECK:STDERR:
'\xC3\xFF'
// CHECK:STDOUT:   - { index:  4, kind:       "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "'\\xC3\\xFF'", has_leading_space: true }

// CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:1: error: empty character literal [CharLiteralEmpty]
// CHECK:STDERR: ''
// CHECK:STDERR: ^
// CHECK:STDERR:
''
// CHECK:STDOUT:   - { index:  5, kind:       "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "''", has_leading_space: true }

// CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:1: error: too many characters [CharLiteralOverflow]
// CHECK:STDERR: 'abcde'
// CHECK:STDERR: ^
// CHECK:STDERR:
'abcde'
// CHECK:STDOUT:   - { index:  6, kind:       "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "'abcde'", has_leading_space: true }

// CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:1: error: unexpected `#` before character literal [CharLiteralRaw]
// CHECK:STDERR: #'a'#
// CHECK:STDERR: ^
// CHECK:STDERR:
#'a'#
// CHECK:STDOUT:   - { index:  7, kind: "CharLiteral", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "#'a'#", has_leading_space: true }

// CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:1: error: character literal is missing a terminator [UnterminatedString]
// CHECK:STDERR: '
// CHECK:STDERR: ^
// CHECK:STDERR:
'
// CHECK:STDOUT:   - { index:  8, kind:       "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "'", has_leading_space: true }

// CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:1: error: character literal is missing a terminator [UnterminatedString]
// CHECK:STDERR: '\'
// CHECK:STDERR: ^
// CHECK:STDERR:
'\'
// CHECK:STDOUT:   - { index:  9, kind:       "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "'\\'", has_leading_space: true }

// CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:1: error: character literal is missing a terminator [UnterminatedString]
// CHECK:STDERR: '\
// CHECK:STDERR: ^
// CHECK:STDERR:
'\
// CHECK:STDOUT:   - { index: 10, kind:       "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "'\\", has_leading_space: true }

// This literal contains a raw tab character.
// CHECK:STDERR: fail_invalid.carbon:[[@LINE+4]]:2: error: whitespace other than plain space must be expressed with an escape sequence in a string literal [InvalidHorizontalWhitespaceInString]
// CHECK:STDERR: '{{\t}}'
// CHECK:STDERR:  ^
// CHECK:STDERR:
'	'
// CHECK:STDOUT:   - { index: 11, kind: "CharLiteral", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "'\t'", has_leading_space: true }
