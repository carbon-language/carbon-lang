// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lex/testdata/fail_mismatched_brackets_at_start_of_file.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lex/testdata/fail_mismatched_brackets_at_start_of_file.carbon

// This test is split out of fail_mismatched_brackets.carbon so that it can use
// AUTOUPDATE-SPLIT. We don't want the autoupdate output to be included inline
// here, since that would insert a comment at the start of the file, which
// breaks the situation that we're trying to test for: a correction at the very
// start of the file.

// --- fail_mismatch_at_start_of_file.carbon
(}var(

// --- AUTOUPDATE-SPLIT

// CHECK:STDERR: fail_mismatch_at_start_of_file.carbon:1:1: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_at_start_of_file.carbon:1:2: error: closing symbol without a corresponding opening symbol [UnmatchedClosing]
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_at_start_of_file.carbon:1:6: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
// CHECK:STDERR: fail_mismatch_at_start_of_file.carbon:1:7: note: possibly missing `)` here [PossiblyMissingBracketHere]
// CHECK:STDERR:
// CHECK:STDOUT: - filename: fail_mismatch_at_start_of_file.carbon
// CHECK:STDOUT:   tokens:
// CHECK:STDOUT:   - { index: 1, kind:      "Error", line: {{ *}}1, column: 1, indent: 1, spelling: "(", has_leading_space: true }
// CHECK:STDOUT:   - { index: 2, kind:      "Error", line: {{ *}}1, column: 2, indent: 1, spelling: "}" }
// CHECK:STDOUT:   - { index: 3, kind:        "Var", line: {{ *}}1, column: 3, indent: 1, spelling: "var" }
// CHECK:STDOUT:   - { index: 4, kind:  "OpenParen", line: {{ *}}1, column: 6, indent: 1, spelling: "(", closing_token: 5 }
// CHECK:STDOUT:   - { index: 5, kind: "CloseParen", line: {{ *}}1, column: 7, indent: 1, spelling: ")", opening_token: 4, has_leading_space: true, recovery: true }
