// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lex/testdata/fail_mismatched_brackets.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lex/testdata/fail_mismatched_brackets.carbon
// CHECK:STDOUT: - filename: fail_mismatched_brackets.carbon
// CHECK:STDOUT:   tokens:

// CHECK:STDERR: fail_mismatched_brackets.carbon:[[@LINE+4]]:1: error: closing symbol without a corresponding opening symbol [UnmatchedClosing]
// CHECK:STDERR: }
// CHECK:STDERR: ^
// CHECK:STDERR:
}
// CHECK:STDOUT:   - { index: 1, kind:      "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "}", has_leading_space: true }

// CHECK:STDERR: fail_mismatched_brackets.carbon:[[@LINE+4]]:3: error: closing symbol without a corresponding opening symbol [UnmatchedClosing]
// CHECK:STDERR: ( } )
// CHECK:STDERR:   ^
// CHECK:STDERR:
( } )
// CHECK:STDOUT:   - { index: 2, kind:  "OpenParen", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "(", closing_token: 4, has_leading_space: true }
// CHECK:STDOUT:   - { index: 3, kind:      "Error", line: {{ *}}[[@LINE-2]], column:   3, indent: 1, spelling: "}", has_leading_space: true }
// CHECK:STDOUT:   - { index: 4, kind: "CloseParen", line: {{ *}}[[@LINE-3]], column:   5, indent: 1, spelling: ")", opening_token: 2, has_leading_space: true }

// CHECK:STDERR: fail_mismatched_brackets.carbon:[[@LINE+4]]:1: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
// CHECK:STDERR: [
// CHECK:STDERR: ^
// CHECK:STDERR:
[
// CHECK:STDOUT:   - { index: 5, kind:      "Error", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "[", has_leading_space: true }
