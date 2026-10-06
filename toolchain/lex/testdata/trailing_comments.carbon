// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// A trailing comment follows other content on a line. It produces no token and
// no error; the next token simply has leading whitespace. A trailing comment is
// never coalesced with the leading comments around it.
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lex/testdata/trailing_comments.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lex/testdata/trailing_comments.carbon
// CHECK:STDOUT: - filename: trailing_comments.carbon
// CHECK:STDOUT:   tokens:

// A leading comment introduces the code below it.
var a: i32 = 1; // A trailing comment annotates the code on its line.
// CHECK:STDOUT:   - { index:  1, kind:             "Var", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "var", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:   5, indent: 1, spelling: "a", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:           "Colon", line: {{ *}}[[@LINE-3]], column:   6, indent: 1, spelling: ":" }
// CHECK:STDOUT:   - { index:  4, kind:  "IntTypeLiteral", line: {{ *}}[[@LINE-4]], column:   8, indent: 1, spelling: "i32", has_leading_space: true }
// CHECK:STDOUT:   - { index:  5, kind:           "Equal", line: {{ *}}[[@LINE-5]], column:  12, indent: 1, spelling: "=", has_leading_space: true }
// CHECK:STDOUT:   - { index:  6, kind:      "IntLiteral", line: {{ *}}[[@LINE-6]], column:  14, indent: 1, spelling: "1", value: "1", has_leading_space: true }
// CHECK:STDOUT:   - { index:  7, kind:            "Semi", line: {{ *}}[[@LINE-7]], column:  15, indent: 1, spelling: ";" }
var b: i32 = 2; // Another trailing comment.
// CHECK:STDOUT:   - { index:  8, kind:             "Var", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "var", has_leading_space: true }
// CHECK:STDOUT:   - { index:  9, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:   5, indent: 1, spelling: "b", identifier: 1, has_leading_space: true }
// CHECK:STDOUT:   - { index: 10, kind:           "Colon", line: {{ *}}[[@LINE-3]], column:   6, indent: 1, spelling: ":" }
// CHECK:STDOUT:   - { index: 11, kind:  "IntTypeLiteral", line: {{ *}}[[@LINE-4]], column:   8, indent: 1, spelling: "i32", has_leading_space: true }
// CHECK:STDOUT:   - { index: 12, kind:           "Equal", line: {{ *}}[[@LINE-5]], column:  12, indent: 1, spelling: "=", has_leading_space: true }
// CHECK:STDOUT:   - { index: 13, kind:      "IntLiteral", line: {{ *}}[[@LINE-6]], column:  14, indent: 1, spelling: "2", value: "2", has_leading_space: true }
// CHECK:STDOUT:   - { index: 14, kind:            "Semi", line: {{ *}}[[@LINE-7]], column:  15, indent: 1, spelling: ";" }

// A trailing comment may directly follow a token with no space before `//`. The
// next line's first token (`var`) is still marked as having leading whitespace.
var c: i32 = 3;// no space before this comment
// CHECK:STDOUT:   - { index: 15, kind:             "Var", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "var", has_leading_space: true }
// CHECK:STDOUT:   - { index: 16, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:   5, indent: 1, spelling: "c", identifier: 2, has_leading_space: true }
// CHECK:STDOUT:   - { index: 17, kind:           "Colon", line: {{ *}}[[@LINE-3]], column:   6, indent: 1, spelling: ":" }
// CHECK:STDOUT:   - { index: 18, kind:  "IntTypeLiteral", line: {{ *}}[[@LINE-4]], column:   8, indent: 1, spelling: "i32", has_leading_space: true }
// CHECK:STDOUT:   - { index: 19, kind:           "Equal", line: {{ *}}[[@LINE-5]], column:  12, indent: 1, spelling: "=", has_leading_space: true }
// CHECK:STDOUT:   - { index: 20, kind:      "IntLiteral", line: {{ *}}[[@LINE-6]], column:  14, indent: 1, spelling: "3", value: "3", has_leading_space: true }
// CHECK:STDOUT:   - { index: 21, kind:            "Semi", line: {{ *}}[[@LINE-7]], column:  15, indent: 1, spelling: ";" }
var d: i32 = 4;
// CHECK:STDOUT:   - { index: 22, kind:             "Var", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "var", has_leading_space: true }
// CHECK:STDOUT:   - { index: 23, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:   5, indent: 1, spelling: "d", identifier: 3, has_leading_space: true }
// CHECK:STDOUT:   - { index: 24, kind:           "Colon", line: {{ *}}[[@LINE-3]], column:   6, indent: 1, spelling: ":" }
// CHECK:STDOUT:   - { index: 25, kind:  "IntTypeLiteral", line: {{ *}}[[@LINE-4]], column:   8, indent: 1, spelling: "i32", has_leading_space: true }
// CHECK:STDOUT:   - { index: 26, kind:           "Equal", line: {{ *}}[[@LINE-5]], column:  12, indent: 1, spelling: "=", has_leading_space: true }
// CHECK:STDOUT:   - { index: 27, kind:      "IntLiteral", line: {{ *}}[[@LINE-6]], column:  14, indent: 1, spelling: "4", value: "4", has_leading_space: true }
// CHECK:STDOUT:   - { index: 28, kind:            "Semi", line: {{ *}}[[@LINE-7]], column:  15, indent: 1, spelling: ";" }

fn F() {
// CHECK:STDOUT:   - { index: 29, kind:              "Fn", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index: 30, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:   4, indent: 1, spelling: "F", identifier: 4, has_leading_space: true }
// CHECK:STDOUT:   - { index: 31, kind:       "OpenParen", line: {{ *}}[[@LINE-3]], column:   5, indent: 1, spelling: "(", closing_token: 32 }
// CHECK:STDOUT:   - { index: 32, kind:      "CloseParen", line: {{ *}}[[@LINE-4]], column:   6, indent: 1, spelling: ")", opening_token: 31 }
// CHECK:STDOUT:   - { index: 33, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-5]], column:   8, indent: 1, spelling: "{", closing_token: 38, has_leading_space: true }
  G(); // Trailing comments may follow any content, including a `;`.
  // CHECK:STDOUT:   - { index: 34, kind:      "Identifier", line: {{ *}}[[@LINE-1]], column:   3, indent: 3, spelling: "G", identifier: 5, has_leading_space: true }
  // CHECK:STDOUT:   - { index: 35, kind:       "OpenParen", line: {{ *}}[[@LINE-2]], column:   4, indent: 3, spelling: "(", closing_token: 36 }
  // CHECK:STDOUT:   - { index: 36, kind:      "CloseParen", line: {{ *}}[[@LINE-3]], column:   5, indent: 3, spelling: ")", opening_token: 35 }
  // CHECK:STDOUT:   - { index: 37, kind:            "Semi", line: {{ *}}[[@LINE-4]], column:   6, indent: 3, spelling: ";" }
}
// CHECK:STDOUT:   - { index: 38, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "}", opening_token: 33, has_leading_space: true }
