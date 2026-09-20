// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lex/testdata/fail_char_literals_bad_encoding.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lex/testdata/fail_char_literals_bad_encoding.carbon
// CHECK:STDOUT: - filename: fail_char_literals_bad_encoding.carbon
// CHECK:STDOUT:   tokens:

// Be careful when operating on this file: it contains invalid UTF-8 sequences
// and Unicode control characters, and text editors are likely to corrupt it.

// CHECK:STDERR: fail_char_literals_bad_encoding.carbon:[[@LINE+2]]:1: error: control character in character literal; specify as escape sequence `\u{00}` [CharLiteralControlCharacter]
// CHECK:STDERR:
'[[@0x00]]'
// CHECK:STDOUT:   - { index:  1, kind:     "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "'\x00'", has_leading_space: true }

// CHECK:STDERR: fail_char_literals_bad_encoding.carbon:[[@LINE+2]]:1: error: control character in character literal; specify as escape sequence `\u{01}` [CharLiteralControlCharacter]
// CHECK:STDERR:
'[[@0x01]]'
// CHECK:STDOUT:   - { index:  2, kind:     "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "'\x01'", has_leading_space: true }

// CHECK:STDERR: fail_char_literals_bad_encoding.carbon:[[@LINE+2]]:1: error: control character in character literal; specify as escape sequence `\u{1F}` [CharLiteralControlCharacter]
// CHECK:STDERR:
'[[@0x1F]]'
// CHECK:STDOUT:   - { index:  3, kind:     "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "'\x1F'", has_leading_space: true }

// CHECK:STDERR: fail_char_literals_bad_encoding.carbon:[[@LINE+2]]:1: error: control character in character literal; specify as escape sequence `\u{7F}` [CharLiteralControlCharacter]
// CHECK:STDERR:
'[[@0x7F]]'
// CHECK:STDOUT:   - { index:  4, kind:     "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "'\x7F'", has_leading_space: true }

// 0xC2 0x80 is U+0080.
// CHECK:STDERR: fail_char_literals_bad_encoding.carbon:[[@LINE+2]]:1: error: control character in character literal; specify as escape sequence `\u{80}` [CharLiteralControlCharacter]
// CHECK:STDERR:
'[[@0xC2]][[@0x80]]'
// CHECK:STDOUT:   - { index:  5, kind:     "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "'\xC2\x80'", has_leading_space: true }

// 0xC2 0x9F is U+009F.
// CHECK:STDERR: fail_char_literals_bad_encoding.carbon:[[@LINE+2]]:1: error: control character in character literal; specify as escape sequence `\u{9F}` [CharLiteralControlCharacter]
// CHECK:STDERR:
'[[@0xC2]][[@0x9F]]'
// CHECK:STDOUT:   - { index:  6, kind:     "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "'\xC2\x9F'", has_leading_space: true }

// 0xC3 is invalid UTF-8 due to having a lead byte with no trail byte.
// CHECK:STDERR: fail_char_literals_bad_encoding.carbon:[[@LINE+2]]:1: error: incomplete UTF-8 [CharLiteralUnderflow]
// CHECK:STDERR:
'[[@0xC3]]'
// CHECK:STDOUT:   - { index:  7, kind:     "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "'\xC3'", has_leading_space: true }

// 0xC3 0xC3 is invalid UTF-8 due to having two lead bytes in a row.
// CHECK:STDERR: fail_char_literals_bad_encoding.carbon:[[@LINE+2]]:1: error: invalid UTF-8 character [CharLiteralInvalidUTF8]
// CHECK:STDERR:
'[[@0xC3]][[@0xC3]]'
// CHECK:STDOUT:   - { index:  8, kind:     "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "'\xC3\xC3'", has_leading_space: true }

// This is invalid UTF-8 due to being too large.
// CHECK:STDERR: fail_char_literals_bad_encoding.carbon:[[@LINE+2]]:1: error: invalid UTF-8 character [CharLiteralInvalidUTF8]
// CHECK:STDERR:
'[[@0xFF]][[@0x80]][[@0x80]][[@0x80]][[@0x80]][[@0x80]][[@0x80]][[@0x80]][[@0x80]]'
// CHECK:STDOUT:   - { index:  9, kind:     "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "'\xFF\x80\x80\x80\x80\x80\x80\x80\x80'", has_leading_space: true }

// 0xED 0xA0 0x80 would be U+D800 but is invalid UTF-8 due to encoding a high surrogate value.
// CHECK:STDERR: fail_char_literals_bad_encoding.carbon:[[@LINE+2]]:1: error: invalid UTF-8 character [CharLiteralInvalidUTF8]
// CHECK:STDERR:
'[[@0xED]][[@0xA0]][[@0x80]]'
// CHECK:STDOUT:   - { index: 10, kind:     "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "'\xED\xA0\x80'", has_leading_space: true }

// 0xED 0xBF 0xBF would be U+DFFF but is invalid UTF-8 due to encoding a low surrogate value.
// CHECK:STDERR: fail_char_literals_bad_encoding.carbon:[[@LINE+2]]:1: error: invalid UTF-8 character [CharLiteralInvalidUTF8]
// CHECK:STDERR:
'[[@0xED]][[@0xBF]][[@0xBF]]'
// CHECK:STDOUT:   - { index: 11, kind:     "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "'\xED\xBF\xBF'", has_leading_space: true }

// 0xED 0xA0 0xBD 0xED 0xBA 0x80 would be U+D83D U+DE80, but those are both surrogates.
// Treated as UTF-16, that would in turn produce U+1F680. But we don't do that.
// CHECK:STDERR: fail_char_literals_bad_encoding.carbon:[[@LINE+2]]:1: error: invalid UTF-8 character [CharLiteralInvalidUTF8]
// CHECK:STDERR:
'[[@0xED]][[@0xA0]][[@0xBD]][[@0xED]][[@0xBA]][[@0x80]]'
// CHECK:STDOUT:   - { index: 12, kind:     "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "'\xED\xA0\xBD\xED\xBA\x80'", has_leading_space: true }
