// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

var s: String = '''
  error here: '''

// Put all autoupdates below because otherwise is puts a CHECK in the middle of
// the block string.
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lex/testdata/fail_block_string_second_line.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lex/testdata/fail_block_string_second_line.carbon
// CHECK:STDERR: fail_block_string_second_line.carbon:[[@LINE-10]]:3: error: only whitespace is permitted before the closing `'''` of a multi-line string [ContentBeforeStringTerminator]
// CHECK:STDERR:   error here: '''
// CHECK:STDERR:   ^
// CHECK:STDERR:
// CHECK:STDOUT: - filename: fail_block_string_second_line.carbon
// CHECK:STDOUT:   tokens:
// CHECK:STDOUT:   - { index: 1, kind:               "Var", line: {{ *}}[[@LINE-17]], column:   1, indent: 1, spelling: "var", has_leading_space: true }
// CHECK:STDOUT:   - { index: 2, kind:        "Identifier", line: {{ *}}[[@LINE-18]], column:   5, indent: 1, spelling: "s", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index: 3, kind:             "Colon", line: {{ *}}[[@LINE-19]], column:   6, indent: 1, spelling: ":" }
// CHECK:STDOUT:   - { index: 4, kind: "StringTypeLiteral", line: {{ *}}[[@LINE-20]], column:   8, indent: 1, spelling: "String", has_leading_space: true }
// CHECK:STDOUT:   - { index: 5, kind:             "Equal", line: {{ *}}[[@LINE-21]], column:  15, indent: 1, spelling: "=", has_leading_space: true }
// CHECK:STDOUT:   - { index: 6, kind:     "StringLiteral", line: {{ *}}[[@LINE-22]], column:  17, indent: 1, spelling: "'''\n  error here: '''", value: "error here: ", has_leading_space: true }
