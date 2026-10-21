// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// A malformed `'''` introducer line is diagnosed and skipped as an error. It
// is never re-lexed as character literals: a character literal is not allowed
// to be empty, so `''` can only begin a block string literal.
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lex/testdata/fail_multiline_string_introducer.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lex/testdata/fail_multiline_string_introducer.carbon
// CHECK:STDOUT: - filename: fail_multiline_string_introducer.carbon
// CHECK:STDOUT:   tokens:

// A `'''` in the middle of a line cannot close a block string literal, so a
// single-line use is a malformed introducer.
// CHECK:STDERR: fail_multiline_string_introducer.carbon:[[@LINE+2]]:17: error: invalid multi-line string literal introducer; a file type indicator may not contain `'`, `#`, or `"`, and the content must begin on a new line [MultiLineStringInvalidIntroducer]
// CHECK:STDERR:
var s: String = '''not multi-line''';
// CHECK:STDOUT:   - { index:  1, kind:            "Var", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "var", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:     "Identifier", line: {{ *}}[[@LINE-2]], column:   5, indent: 1, spelling: "s", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:          "Colon", line: {{ *}}[[@LINE-3]], column:   6, indent: 1, spelling: ":" }
// CHECK:STDOUT:   - { index:  4, kind:     "Identifier", line: {{ *}}[[@LINE-4]], column:   8, indent: 1, spelling: "String", identifier: 1, has_leading_space: true }
// CHECK:STDOUT:   - { index:  5, kind:          "Equal", line: {{ *}}[[@LINE-5]], column:  15, indent: 1, spelling: "=", has_leading_space: true }
// CHECK:STDOUT:   - { index:  6, kind:          "Error", line: {{ *}}[[@LINE-6]], column:  17, indent: 1, spelling: "'''not multi-line''';", has_leading_space: true }

// A file type indicator may not contain `#`. The error covers the introducer
// line and lexing continues on the next line.
// CHECK:STDERR: fail_multiline_string_introducer.carbon:[[@LINE+2]]:17: error: invalid multi-line string literal introducer; a file type indicator may not contain `'`, `#`, or `"`, and the content must begin on a new line [MultiLineStringInvalidIntroducer]
// CHECK:STDERR:
var t: String = '''bad#type
// CHECK:STDOUT:   - { index:  7, kind:            "Var", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "var", has_leading_space: true }
// CHECK:STDOUT:   - { index:  8, kind:     "Identifier", line: {{ *}}[[@LINE-2]], column:   5, indent: 1, spelling: "t", identifier: 2, has_leading_space: true }
// CHECK:STDOUT:   - { index:  9, kind:          "Colon", line: {{ *}}[[@LINE-3]], column:   6, indent: 1, spelling: ":" }
// CHECK:STDOUT:   - { index: 10, kind:     "Identifier", line: {{ *}}[[@LINE-4]], column:   8, indent: 1, spelling: "String", identifier: 1, has_leading_space: true }
// CHECK:STDOUT:   - { index: 11, kind:          "Equal", line: {{ *}}[[@LINE-5]], column:  15, indent: 1, spelling: "=", has_leading_space: true }
// CHECK:STDOUT:   - { index: 12, kind:          "Error", line: {{ *}}[[@LINE-6]], column:  17, indent: 1, spelling: "'''bad#type", has_leading_space: true }
var u: i32 = 1;
// CHECK:STDOUT:   - { index: 13, kind:            "Var", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "var", has_leading_space: true }
// CHECK:STDOUT:   - { index: 14, kind:     "Identifier", line: {{ *}}[[@LINE-2]], column:   5, indent: 1, spelling: "u", identifier: 3, has_leading_space: true }
// CHECK:STDOUT:   - { index: 15, kind:          "Colon", line: {{ *}}[[@LINE-3]], column:   6, indent: 1, spelling: ":" }
// CHECK:STDOUT:   - { index: 16, kind: "IntTypeLiteral", line: {{ *}}[[@LINE-4]], column:   8, indent: 1, spelling: "i32", has_leading_space: true }
// CHECK:STDOUT:   - { index: 17, kind:          "Equal", line: {{ *}}[[@LINE-5]], column:  12, indent: 1, spelling: "=", has_leading_space: true }
// CHECK:STDOUT:   - { index: 18, kind:     "IntLiteral", line: {{ *}}[[@LINE-6]], column:  14, indent: 1, spelling: "1", value: "1", has_leading_space: true }
// CHECK:STDOUT:   - { index: 19, kind:           "Semi", line: {{ *}}[[@LINE-7]], column:  15, indent: 1, spelling: ";" }
