// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lex/testdata/fail_mismatched_brackets_close_brace.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lex/testdata/fail_mismatched_brackets_close_brace.carbon

// The cases from `fail_mismatched_brackets.carbon` where recovery suggests a
// `}`. A `}` goes on a line of its own, so its note points at the line after the
// scope's last token, and the `CHECK`s have to live in their own split:
// autoupdate would otherwise write them at that very position, and the note
// would quote its own `CHECK` line.

// --- fail_dedent_inside_scope.carbon

fn F() {
  if (x) {
    foo();
  bar();
}

// --- fail_unclosed_at_eof.carbon

fn F() {
  var x: i32 = 1;


// --- AUTOUPDATE-SPLIT

// CHECK:STDERR: fail_dedent_inside_scope.carbon:3:10: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
// CHECK:STDERR:   if (x) {
// CHECK:STDERR:          ^
// CHECK:STDERR: fail_dedent_inside_scope.carbon:5:3: note: possibly missing `}` here [PossiblyMissingBracketHere]
// CHECK:STDERR:   bar();
// CHECK:STDERR:   ^
// CHECK:STDERR:
// CHECK:STDERR: fail_unclosed_at_eof.carbon:2:8: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
// CHECK:STDERR: fn F() {
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_unclosed_at_eof.carbon:4:1: note: possibly missing `}` here [PossiblyMissingBracketHere]
// CHECK:STDERR:
// CHECK:STDERR: ^
// CHECK:STDERR:
// CHECK:STDOUT: - filename: fail_dedent_inside_scope.carbon
// CHECK:STDOUT:   tokens:
// CHECK:STDOUT:   - { index:  1, kind:              "Fn", line: {{ *}}2, column:  1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:      "Identifier", line: {{ *}}2, column:  4, indent: 1, spelling: "F", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:       "OpenParen", line: {{ *}}2, column:  5, indent: 1, spelling: "(", closing_token: 4 }
// CHECK:STDOUT:   - { index:  4, kind:      "CloseParen", line: {{ *}}2, column:  6, indent: 1, spelling: ")", opening_token: 3 }
// CHECK:STDOUT:   - { index:  5, kind:  "OpenCurlyBrace", line: {{ *}}2, column:  8, indent: 1, spelling: "{", closing_token: 20, has_leading_space: true }
// CHECK:STDOUT:   - { index:  6, kind:              "If", line: {{ *}}3, column:  3, indent: 3, spelling: "if", has_leading_space: true }
// CHECK:STDOUT:   - { index:  7, kind:       "OpenParen", line: {{ *}}3, column:  6, indent: 3, spelling: "(", closing_token: 9, has_leading_space: true }
// CHECK:STDOUT:   - { index:  8, kind:      "Identifier", line: {{ *}}3, column:  7, indent: 3, spelling: "x", identifier: 1 }
// CHECK:STDOUT:   - { index:  9, kind:      "CloseParen", line: {{ *}}3, column:  8, indent: 3, spelling: ")", opening_token: 7 }
// CHECK:STDOUT:   - { index: 10, kind:  "OpenCurlyBrace", line: {{ *}}3, column: 10, indent: 3, spelling: "{", closing_token: 15, has_leading_space: true }
// CHECK:STDOUT:   - { index: 11, kind:      "Identifier", line: {{ *}}4, column:  5, indent: 5, spelling: "foo", identifier: 2, has_leading_space: true }
// CHECK:STDOUT:   - { index: 12, kind:       "OpenParen", line: {{ *}}4, column:  8, indent: 5, spelling: "(", closing_token: 13 }
// CHECK:STDOUT:   - { index: 13, kind:      "CloseParen", line: {{ *}}4, column:  9, indent: 5, spelling: ")", opening_token: 12 }
// CHECK:STDOUT:   - { index: 14, kind:            "Semi", line: {{ *}}4, column: 10, indent: 5, spelling: ";" }
// CHECK:STDOUT:   - { index: 15, kind: "CloseCurlyBrace", line: {{ *}}4, column: 11, indent: 5, spelling: "}", opening_token: 10, has_leading_space: true, recovery: true }
// CHECK:STDOUT:   - { index: 16, kind:      "Identifier", line: {{ *}}5, column:  3, indent: 3, spelling: "bar", identifier: 3 }
// CHECK:STDOUT:   - { index: 17, kind:       "OpenParen", line: {{ *}}5, column:  6, indent: 3, spelling: "(", closing_token: 18 }
// CHECK:STDOUT:   - { index: 18, kind:      "CloseParen", line: {{ *}}5, column:  7, indent: 3, spelling: ")", opening_token: 17 }
// CHECK:STDOUT:   - { index: 19, kind:            "Semi", line: {{ *}}5, column:  8, indent: 3, spelling: ";" }
// CHECK:STDOUT:   - { index: 20, kind: "CloseCurlyBrace", line: {{ *}}6, column:  1, indent: 1, spelling: "}", opening_token: 5, has_leading_space: true }
// CHECK:STDOUT: - filename: fail_unclosed_at_eof.carbon
// CHECK:STDOUT:   tokens:
// CHECK:STDOUT:   - { index:  1, kind:              "Fn", line: {{ *}}2, column:  1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:      "Identifier", line: {{ *}}2, column:  4, indent: 1, spelling: "F", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:       "OpenParen", line: {{ *}}2, column:  5, indent: 1, spelling: "(", closing_token: 4 }
// CHECK:STDOUT:   - { index:  4, kind:      "CloseParen", line: {{ *}}2, column:  6, indent: 1, spelling: ")", opening_token: 3 }
// CHECK:STDOUT:   - { index:  5, kind:  "OpenCurlyBrace", line: {{ *}}2, column:  8, indent: 1, spelling: "{", closing_token: 13, has_leading_space: true }
// CHECK:STDOUT:   - { index:  6, kind:             "Var", line: {{ *}}3, column:  3, indent: 3, spelling: "var", has_leading_space: true }
// CHECK:STDOUT:   - { index:  7, kind:      "Identifier", line: {{ *}}3, column:  7, indent: 3, spelling: "x", identifier: 1, has_leading_space: true }
// CHECK:STDOUT:   - { index:  8, kind:           "Colon", line: {{ *}}3, column:  8, indent: 3, spelling: ":" }
// CHECK:STDOUT:   - { index:  9, kind:  "IntTypeLiteral", line: {{ *}}3, column: 10, indent: 3, spelling: "i32", has_leading_space: true }
// CHECK:STDOUT:   - { index: 10, kind:           "Equal", line: {{ *}}3, column: 14, indent: 3, spelling: "=", has_leading_space: true }
// CHECK:STDOUT:   - { index: 11, kind:      "IntLiteral", line: {{ *}}3, column: 16, indent: 3, spelling: "1", value: "1", has_leading_space: true }
// CHECK:STDOUT:   - { index: 12, kind:            "Semi", line: {{ *}}3, column: 17, indent: 3, spelling: ";" }
// CHECK:STDOUT:   - { index: 13, kind: "CloseCurlyBrace", line: {{ *}}3, column: 18, indent: 3, spelling: "}", opening_token: 5, has_leading_space: true, recovery: true }
