// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lex/testdata/fail_mismatched_brackets.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lex/testdata/fail_mismatched_brackets.carbon

// --- top_level_brackets.carbon
// CHECK:STDOUT: - filename: top_level_brackets.carbon
// CHECK:STDOUT:   tokens:

// CHECK:STDERR: top_level_brackets.carbon:[[@LINE+4]]:1: error: closing symbol without a corresponding opening symbol [UnmatchedClosing]
// CHECK:STDERR: }
// CHECK:STDERR: ^
// CHECK:STDERR:
}
// CHECK:STDOUT:   - { index: 1, kind:      "Error", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "}", has_leading_space: true }

// CHECK:STDERR: top_level_brackets.carbon:[[@LINE+4]]:3: error: closing symbol without a corresponding opening symbol [UnmatchedClosing]
// CHECK:STDERR: ( } )
// CHECK:STDERR:   ^
// CHECK:STDERR:
( } )
// CHECK:STDOUT:   - { index: 2, kind:  "OpenParen", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "(", closing_token: 4, has_leading_space: true }
// CHECK:STDOUT:   - { index: 3, kind:      "Error", line: {{ *}}[[@LINE-2]], column: 3, indent: 1, spelling: "}", has_leading_space: true }
// CHECK:STDOUT:   - { index: 4, kind: "CloseParen", line: {{ *}}[[@LINE-3]], column: 5, indent: 1, spelling: ")", opening_token: 2, has_leading_space: true }

// CHECK:STDERR: top_level_brackets.carbon:[[@LINE+4]]:1: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
// CHECK:STDERR: [
// CHECK:STDERR: ^
// CHECK:STDERR:
[
// CHECK:STDOUT:   - { index: 5, kind:      "Error", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "[", has_leading_space: true }

// --- missing_brace_after_if.carbon
// CHECK:STDOUT: - filename: missing_brace_after_if.carbon
// CHECK:STDOUT:   tokens:

fn F() {
// CHECK:STDOUT:   - { index:  1, kind:              "Fn", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:   4, indent: 1, spelling: "F", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:       "OpenParen", line: {{ *}}[[@LINE-3]], column:   5, indent: 1, spelling: "(", closing_token: 4 }
// CHECK:STDOUT:   - { index:  4, kind:      "CloseParen", line: {{ *}}[[@LINE-4]], column:   6, indent: 1, spelling: ")", opening_token: 3 }
// CHECK:STDOUT:   - { index:  5, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-5]], column:   8, indent: 1, spelling: "{", closing_token: 14, has_leading_space: true }
  if (thing1)
  // CHECK:STDOUT:   - { index:  6, kind:              "If", line: {{ *}}[[@LINE-1]], column:   3, indent: 3, spelling: "if", has_leading_space: true }
  // CHECK:STDOUT:   - { index:  7, kind:       "OpenParen", line: {{ *}}[[@LINE-2]], column:   6, indent: 3, spelling: "(", closing_token: 9, has_leading_space: true }
  // CHECK:STDOUT:   - { index:  8, kind:      "Identifier", line: {{ *}}[[@LINE-3]], column:   7, indent: 3, spelling: "thing1", identifier: 1 }
  // CHECK:STDOUT:   - { index:  9, kind:      "CloseParen", line: {{ *}}[[@LINE-4]], column:  13, indent: 3, spelling: ")", opening_token: 7 }
  // CHECK:STDOUT:   - { index: 10, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-5]], column:  14, indent: 3, spelling: "{", closing_token: 13, has_leading_space: true, recovery: true }
    thing2;
    // CHECK:STDOUT:   - { index: 11, kind:      "Identifier", line: {{ *}}[[@LINE-1]], column:   5, indent: 5, spelling: "thing2", identifier: 2, has_leading_space: true }
    // CHECK:STDOUT:   - { index: 12, kind:            "Semi", line: {{ *}}[[@LINE-2]], column:  11, indent: 5, spelling: ";" }
  // CHECK:STDERR: missing_brace_after_if.carbon:[[@LINE+4]]:3: error: closing symbol without a corresponding opening symbol [UnmatchedClosing]
  // CHECK:STDERR:   }
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  }
  // CHECK:STDOUT:   - { index: 13, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column:   3, indent: 3, spelling: "}", opening_token: 10, has_leading_space: true }
}
// CHECK:STDOUT:   - { index: 14, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "}", opening_token: 5, has_leading_space: true }
