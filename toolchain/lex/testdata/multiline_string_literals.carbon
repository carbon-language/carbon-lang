// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// For multiline string literals, force autoupdates to the end so that we don't
// accidentally mix diagnostics with the literal.
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lex/testdata/multiline_string_literals.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lex/testdata/multiline_string_literals.carbon

// --- fail_indent_mismatch.carbon

  '''
a
  '''

// --- fail_quotes.carbon

"""
a
"""

// --- introducer_trailing_comment.carbon

// A trailing comment may follow the file type indicator on the introducer line.
// It is treated like trailing whitespace: it is part of the literal's introducer
// and produces no comment token, and it may contain `'`, `#`, and `"` even
// though the file type indicator may not.
//
// TODO: Surface this comment through the lexer's comment records.
var s: String = '''py // A trailing comment with ', #, and " in it.
  content
  ''';

// --- AUTOUPDATE-SPLIT

// CHECK:STDERR: fail_indent_mismatch.carbon:3:1: error: indentation does not match that of the closing `'''` in multi-line string literal [MismatchedIndentInString]
// CHECK:STDERR: fail_indent_mismatch.carbon:4:1-2: note: closing `'''` is indented to here [ClosingDelimiterIndent]
// CHECK:STDERR:
// CHECK:STDERR: fail_quotes.carbon:2:1: error: use `'''` delimiters for a multi-line string literal, not `"""` [MultiLineStringWithDoubleQuotes]
// CHECK:STDERR:
// CHECK:STDOUT: - filename: fail_indent_mismatch.carbon
// CHECK:STDOUT:   tokens:
// CHECK:STDOUT:   - { index: 1, kind: "StringLiteral", line: {{ *}}2, column: 3, indent: 3, spelling: "'''\na\n  '''", value: "a\n", has_leading_space: true }
// CHECK:STDOUT: - filename: fail_quotes.carbon
// CHECK:STDOUT:   tokens:
// CHECK:STDOUT:   - { index: 1, kind: "StringLiteral", line: {{ *}}2, column: 1, indent: 1, spelling: "\"\"\"\na\n\"\"\"", value: "a\n", has_leading_space: true }
// CHECK:STDOUT: - filename: introducer_trailing_comment.carbon
// CHECK:STDOUT:   tokens:
// CHECK:STDOUT:   - { index: 1, kind:           "Var", line: {{ *}}8, column:  1, indent: 1, spelling: "var", has_leading_space: true }
// CHECK:STDOUT:   - { index: 2, kind:    "Identifier", line: {{ *}}8, column:  5, indent: 1, spelling: "s", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index: 3, kind:         "Colon", line: {{ *}}8, column:  6, indent: 1, spelling: ":" }
// CHECK:STDOUT:   - { index: 4, kind:    "Identifier", line: {{ *}}8, column:  8, indent: 1, spelling: "String", identifier: 1, has_leading_space: true }
// CHECK:STDOUT:   - { index: 5, kind:         "Equal", line: {{ *}}8, column: 15, indent: 1, spelling: "=", has_leading_space: true }
// CHECK:STDOUT:   - { index: 6, kind: "StringLiteral", line: {{ *}}8, column: 17, indent: 1, spelling: "'''py // A trailing comment with ', #, and \" in it.\n  content\n  '''", value: "content\n", has_leading_space: true }
// CHECK:STDOUT:   - { index: 7, kind:          "Semi", line: {{ *}}10, column:  6, indent: 3, spelling: ";" }
