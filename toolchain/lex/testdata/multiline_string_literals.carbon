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

// --- AUTOUPDATE-SPLIT

// CHECK:STDERR: fail_indent_mismatch.carbon:3:1: error: indentation does not match that of the closing `'''` in multi-line string literal [MismatchedIndentInString]
// CHECK:STDERR: a
// CHECK:STDERR: ^
// CHECK:STDERR:
// CHECK:STDERR: fail_quotes.carbon:2:1: error: use `'''` delimiters for a multi-line string literal, not `"""` [MultiLineStringWithDoubleQuotes]
// CHECK:STDERR: """
// CHECK:STDERR: ^
// CHECK:STDERR:
// CHECK:STDOUT: - filename: fail_indent_mismatch.carbon
// CHECK:STDOUT:   tokens:
// CHECK:STDOUT:   - { index: 1, kind: "StringLiteral", line: {{ *}}2, column: 3, indent: 3, spelling: "'''\na\n  '''", value: "a\n", has_leading_space: true }
// CHECK:STDOUT: - filename: fail_quotes.carbon
// CHECK:STDOUT:   tokens:
// CHECK:STDOUT:   - { index: 1, kind: "StringLiteral", line: {{ *}}2, column: 1, indent: 1, spelling: "\"\"\"\na\n\"\"\"", value: "a\n", has_leading_space: true }
