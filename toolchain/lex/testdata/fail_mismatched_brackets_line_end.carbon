// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lex/testdata/fail_mismatched_brackets_line_end.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lex/testdata/fail_mismatched_brackets_line_end.carbon
// CHECK:STDOUT: - filename: fail_mismatched_brackets_line_end.carbon
// CHECK:STDOUT:   tokens:

// A missing `{` goes at the end of the line of the token it follows, before the
// comment there.
fn F() {
// CHECK:STDOUT:   - { index:  1, kind:              "Fn", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:   4, indent: 1, spelling: "F", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:       "OpenParen", line: {{ *}}[[@LINE-3]], column:   5, indent: 1, spelling: "(", closing_token: 4 }
// CHECK:STDOUT:   - { index:  4, kind:      "CloseParen", line: {{ *}}[[@LINE-4]], column:   6, indent: 1, spelling: ")", opening_token: 3 }
// CHECK:STDOUT:   - { index:  5, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-5]], column:   8, indent: 1, spelling: "{", closing_token: 16, has_leading_space: true }
  if (true) // A comment.
  // CHECK:STDOUT:   - { index:  6, kind:              "If", line: {{ *}}[[@LINE-1]], column:   3, indent: 3, spelling: "if", has_leading_space: true }
  // CHECK:STDOUT:   - { index:  7, kind:       "OpenParen", line: {{ *}}[[@LINE-2]], column:   6, indent: 3, spelling: "(", closing_token: 9, has_leading_space: true }
  // CHECK:STDOUT:   - { index:  8, kind:            "True", line: {{ *}}[[@LINE-3]], column:   7, indent: 3, spelling: "true" }
  // CHECK:STDOUT:   - { index:  9, kind:      "CloseParen", line: {{ *}}[[@LINE-4]], column:  11, indent: 3, spelling: ")", opening_token: 7 }
  // CHECK:STDOUT:   - { index: 10, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-5]], column:  12, indent: 3, spelling: "{", closing_token: 15, has_leading_space: true, recovery: true }
    G();
    // CHECK:STDOUT:   - { index: 11, kind:      "Identifier", line: {{ *}}[[@LINE-1]], column:   5, indent: 5, spelling: "G", identifier: 1 }
    // CHECK:STDOUT:   - { index: 12, kind:       "OpenParen", line: {{ *}}[[@LINE-2]], column:   6, indent: 5, spelling: "(", closing_token: 13 }
    // CHECK:STDOUT:   - { index: 13, kind:      "CloseParen", line: {{ *}}[[@LINE-3]], column:   7, indent: 5, spelling: ")", opening_token: 12 }
    // CHECK:STDOUT:   - { index: 14, kind:            "Semi", line: {{ *}}[[@LINE-4]], column:   8, indent: 5, spelling: ";" }
  // CHECK:STDERR: fail_mismatched_brackets_line_end.carbon:[[@LINE+3]]:3: error: closing symbol without a corresponding opening symbol [UnmatchedClosing]
  // CHECK:STDERR: fail_mismatched_brackets_line_end.carbon:[[@LINE-12]]:12: note: possibly missing `{` here [PossiblyMissingBracketHere]
  // CHECK:STDERR:
  }
  // CHECK:STDOUT:   - { index: 15, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column:   3, indent: 3, spelling: "}", opening_token: 10, has_leading_space: true }
}
// CHECK:STDOUT:   - { index: 16, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "}", opening_token: 5, has_leading_space: true }
