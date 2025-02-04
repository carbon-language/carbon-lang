// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TODO: For this example, we should ideally report a missing `{` on the `if`
// line, for example based on the indentation of the surrounding lines.

fn F() {
  if (thing1)
    thing2;
  }
}

// The check lines are inserted at the end so that they don't disrupt the
// indentation of lines in the text.
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lex/testdata/fail_mismatched_brackets_2.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lex/testdata/fail_mismatched_brackets_2.carbon
// CHECK:STDERR: fail_mismatched_brackets_2.carbon:[[@LINE-9]]:1: error: closing symbol without a corresponding opening symbol [UnmatchedClosing]
// CHECK:STDERR: }
// CHECK:STDERR: ^
// CHECK:STDERR:
// CHECK:STDOUT: - filename: fail_mismatched_brackets_2.carbon
// CHECK:STDOUT:   tokens:
// CHECK:STDOUT:   - { index:  1, kind:              "Fn", line: {{ *}}[[@LINE-19]], column:   1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:      "Identifier", line: {{ *}}[[@LINE-20]], column:   4, indent: 1, spelling: "F", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:       "OpenParen", line: {{ *}}[[@LINE-21]], column:   5, indent: 1, spelling: "(", closing_token: 4 }
// CHECK:STDOUT:   - { index:  4, kind:      "CloseParen", line: {{ *}}[[@LINE-22]], column:   6, indent: 1, spelling: ")", opening_token: 3 }
// CHECK:STDOUT:   - { index:  5, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-23]], column:   8, indent: 1, spelling: "{", closing_token: 12, has_leading_space: true }
// CHECK:STDOUT:   - { index:  6, kind:              "If", line: {{ *}}[[@LINE-23]], column:   3, indent: 3, spelling: "if", has_leading_space: true }
// CHECK:STDOUT:   - { index:  7, kind:       "OpenParen", line: {{ *}}[[@LINE-24]], column:   6, indent: 3, spelling: "(", closing_token: 9, has_leading_space: true }
// CHECK:STDOUT:   - { index:  8, kind:      "Identifier", line: {{ *}}[[@LINE-25]], column:   7, indent: 3, spelling: "thing1", identifier: 1 }
// CHECK:STDOUT:   - { index:  9, kind:      "CloseParen", line: {{ *}}[[@LINE-26]], column:  13, indent: 3, spelling: ")", opening_token: 7 }
// CHECK:STDOUT:   - { index: 10, kind:      "Identifier", line: {{ *}}[[@LINE-26]], column:   5, indent: 5, spelling: "thing2", identifier: 2, has_leading_space: true }
// CHECK:STDOUT:   - { index: 11, kind:            "Semi", line: {{ *}}[[@LINE-27]], column:  11, indent: 5, spelling: ";" }
// CHECK:STDOUT:   - { index: 12, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-27]], column:   3, indent: 3, spelling: "}", opening_token: 5, has_leading_space: true }
// CHECK:STDOUT:   - { index: 13, kind:           "Error", line: {{ *}}[[@LINE-27]], column:   1, indent: 1, spelling: "}", has_leading_space: true }
