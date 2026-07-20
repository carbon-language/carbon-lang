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
// CHECK:STDOUT:   - { index: 1, kind:              "Error", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "}", has_leading_space: true }

// CHECK:STDERR: top_level_brackets.carbon:[[@LINE+4]]:3: error: closing symbol without a corresponding opening symbol [UnmatchedClosing]
// CHECK:STDERR: ( } )
// CHECK:STDERR:   ^
// CHECK:STDERR:
( } )
// CHECK:STDOUT:   - { index: 2, kind:          "OpenParen", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "(", closing_token: 4, has_leading_space: true }
// CHECK:STDOUT:   - { index: 3, kind:              "Error", line: {{ *}}[[@LINE-2]], column: 3, indent: 1, spelling: "}", has_leading_space: true }
// CHECK:STDOUT:   - { index: 4, kind:         "CloseParen", line: {{ *}}[[@LINE-3]], column: 5, indent: 1, spelling: ")", opening_token: 2, has_leading_space: true }

// CHECK:STDERR: top_level_brackets.carbon:[[@LINE+3]]:1: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
// CHECK:STDERR: [
// CHECK:STDERR: ^
[
// CHECK:STDOUT:   - { index: 5, kind:  "OpenSquareBracket", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "[", closing_token: 6, has_leading_space: true }
// CHECK:STDOUT:   - { index: 6, kind: "CloseSquareBracket", line: {{ *}}[[@LINE-2]], column: 2, indent: 1, spelling: "]", opening_token: 5, has_leading_space: true, recovery: true }
// CHECK:STDERR: top_level_brackets.carbon:[[@LINE+4]]:1: note: possibly missing `]` here [PossiblyMissingBracketHere]
// CHECK:STDERR:
// CHECK:STDERR: ^
// CHECK:STDERR:

// --- missing_brace_after_if.carbon
// CHECK:STDOUT: - filename: missing_brace_after_if.carbon
// CHECK:STDOUT:   tokens:

fn F() {
// CHECK:STDOUT:   - { index:  1, kind:              "Fn", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:  4, indent: 1, spelling: "F", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:       "OpenParen", line: {{ *}}[[@LINE-3]], column:  5, indent: 1, spelling: "(", closing_token: 4 }
// CHECK:STDOUT:   - { index:  4, kind:      "CloseParen", line: {{ *}}[[@LINE-4]], column:  6, indent: 1, spelling: ")", opening_token: 3 }
// CHECK:STDOUT:   - { index:  5, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-5]], column:  8, indent: 1, spelling: "{", closing_token: 14, has_leading_space: true }
  if (thing1)
  // CHECK:STDOUT:   - { index:  6, kind:              "If", line: {{ *}}[[@LINE-1]], column:  3, indent: 3, spelling: "if", has_leading_space: true }
  // CHECK:STDOUT:   - { index:  7, kind:       "OpenParen", line: {{ *}}[[@LINE-2]], column:  6, indent: 3, spelling: "(", closing_token: 9, has_leading_space: true }
  // CHECK:STDOUT:   - { index:  8, kind:      "Identifier", line: {{ *}}[[@LINE-3]], column:  7, indent: 3, spelling: "thing1", identifier: 1 }
  // CHECK:STDOUT:   - { index:  9, kind:      "CloseParen", line: {{ *}}[[@LINE-4]], column: 13, indent: 3, spelling: ")", opening_token: 7 }
  // CHECK:STDOUT:   - { index: 10, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-5]], column: 14, indent: 3, spelling: "{", closing_token: 13, has_leading_space: true, recovery: true }
    thing2;
    // CHECK:STDOUT:   - { index: 11, kind:      "Identifier", line: {{ *}}[[@LINE-1]], column:  5, indent: 5, spelling: "thing2", identifier: 2, has_leading_space: true }
    // CHECK:STDOUT:   - { index: 12, kind:            "Semi", line: {{ *}}[[@LINE-2]], column: 11, indent: 5, spelling: ";" }
  // CHECK:STDERR: missing_brace_after_if.carbon:[[@LINE+7]]:3: error: closing symbol without a corresponding opening symbol [UnmatchedClosing]
  // CHECK:STDERR:   }
  // CHECK:STDERR:   ^
  // CHECK:STDERR: missing_brace_after_if.carbon:[[@LINE-6]]:5: note: possibly missing `{` here [PossiblyMissingBracketHere]
  // CHECK:STDERR:     thing2;
  // CHECK:STDERR:     ^~~~~~
  // CHECK:STDERR:
  }
  // CHECK:STDOUT:   - { index: 13, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column:  3, indent: 3, spelling: "}", opening_token: 10, has_leading_space: true }
}
// CHECK:STDOUT:   - { index: 14, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "}", opening_token: 5, has_leading_space: true }

// --- missing_brace_multi_line_header.carbon
// CHECK:STDOUT: - filename: missing_brace_multi_line_header.carbon
// CHECK:STDOUT:   tokens:

fn F[T: type]
// CHECK:STDOUT:   - { index:  1, kind:                 "Fn", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:         "Identifier", line: {{ *}}[[@LINE-2]], column:  4, indent: 1, spelling: "F", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:  "OpenSquareBracket", line: {{ *}}[[@LINE-3]], column:  5, indent: 1, spelling: "[", closing_token: 7 }
// CHECK:STDOUT:   - { index:  4, kind:         "Identifier", line: {{ *}}[[@LINE-4]], column:  6, indent: 1, spelling: "T", identifier: 1 }
// CHECK:STDOUT:   - { index:  5, kind:              "Colon", line: {{ *}}[[@LINE-5]], column:  7, indent: 1, spelling: ":" }
// CHECK:STDOUT:   - { index:  6, kind:               "Type", line: {{ *}}[[@LINE-6]], column:  9, indent: 1, spelling: "type", has_leading_space: true }
// CHECK:STDOUT:   - { index:  7, kind: "CloseSquareBracket", line: {{ *}}[[@LINE-7]], column: 13, indent: 1, spelling: "]", opening_token: 3 }
    (x: T)
    // CHECK:STDOUT:   - { index:  8, kind:          "OpenParen", line: {{ *}}[[@LINE-1]], column:  5, indent: 5, spelling: "(", closing_token: 12, has_leading_space: true }
    // CHECK:STDOUT:   - { index:  9, kind:         "Identifier", line: {{ *}}[[@LINE-2]], column:  6, indent: 5, spelling: "x", identifier: 2 }
    // CHECK:STDOUT:   - { index: 10, kind:              "Colon", line: {{ *}}[[@LINE-3]], column:  7, indent: 5, spelling: ":" }
    // CHECK:STDOUT:   - { index: 11, kind:         "Identifier", line: {{ *}}[[@LINE-4]], column:  9, indent: 5, spelling: "T", identifier: 1, has_leading_space: true }
    // CHECK:STDOUT:   - { index: 12, kind:         "CloseParen", line: {{ *}}[[@LINE-5]], column: 10, indent: 5, spelling: ")", opening_token: 8 }
    // CHECK:STDOUT:   - { index: 13, kind:     "OpenCurlyBrace", line: {{ *}}[[@LINE-6]], column: 11, indent: 5, spelling: "{", closing_token: 22, has_leading_space: true, recovery: true }
  foo();
  // CHECK:STDOUT:   - { index: 14, kind:         "Identifier", line: {{ *}}[[@LINE-1]], column:  3, indent: 3, spelling: "foo", identifier: 3, has_leading_space: true }
  // CHECK:STDOUT:   - { index: 15, kind:          "OpenParen", line: {{ *}}[[@LINE-2]], column:  6, indent: 3, spelling: "(", closing_token: 16 }
  // CHECK:STDOUT:   - { index: 16, kind:         "CloseParen", line: {{ *}}[[@LINE-3]], column:  7, indent: 3, spelling: ")", opening_token: 15 }
  // CHECK:STDOUT:   - { index: 17, kind:               "Semi", line: {{ *}}[[@LINE-4]], column:  8, indent: 3, spelling: ";" }
  bar();
  // CHECK:STDOUT:   - { index: 18, kind:         "Identifier", line: {{ *}}[[@LINE-1]], column:  3, indent: 3, spelling: "bar", identifier: 4, has_leading_space: true }
  // CHECK:STDOUT:   - { index: 19, kind:          "OpenParen", line: {{ *}}[[@LINE-2]], column:  6, indent: 3, spelling: "(", closing_token: 20 }
  // CHECK:STDOUT:   - { index: 20, kind:         "CloseParen", line: {{ *}}[[@LINE-3]], column:  7, indent: 3, spelling: ")", opening_token: 19 }
  // CHECK:STDOUT:   - { index: 21, kind:               "Semi", line: {{ *}}[[@LINE-4]], column:  8, indent: 3, spelling: ";" }
// CHECK:STDERR: missing_brace_multi_line_header.carbon:[[@LINE+7]]:1: error: closing symbol without a corresponding opening symbol [UnmatchedClosing]
// CHECK:STDERR: }
// CHECK:STDERR: ^
// CHECK:STDERR: missing_brace_multi_line_header.carbon:[[@LINE-13]]:3: note: possibly missing `{` here [PossiblyMissingBracketHere]
// CHECK:STDERR:   foo();
// CHECK:STDERR:   ^~~
// CHECK:STDERR:
}
// CHECK:STDOUT:   - { index: 22, kind:    "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "}", opening_token: 13, has_leading_space: true }

// --- dedent_inside_scope.carbon
// CHECK:STDOUT: - filename: dedent_inside_scope.carbon
// CHECK:STDOUT:   tokens:

fn F() {
// CHECK:STDOUT:   - { index:  1, kind:              "Fn", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:  4, indent: 1, spelling: "F", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:       "OpenParen", line: {{ *}}[[@LINE-3]], column:  5, indent: 1, spelling: "(", closing_token: 4 }
// CHECK:STDOUT:   - { index:  4, kind:      "CloseParen", line: {{ *}}[[@LINE-4]], column:  6, indent: 1, spelling: ")", opening_token: 3 }
// CHECK:STDOUT:   - { index:  5, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-5]], column:  8, indent: 1, spelling: "{", closing_token: 20, has_leading_space: true }
  // CHECK:STDERR: dedent_inside_scope.carbon:[[@LINE+3]]:10: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
  // CHECK:STDERR:   if (x) {
  // CHECK:STDERR:          ^
  if (x) {
  // CHECK:STDOUT:   - { index:  6, kind:              "If", line: {{ *}}[[@LINE-1]], column:  3, indent: 3, spelling: "if", has_leading_space: true }
  // CHECK:STDOUT:   - { index:  7, kind:       "OpenParen", line: {{ *}}[[@LINE-2]], column:  6, indent: 3, spelling: "(", closing_token: 9, has_leading_space: true }
  // CHECK:STDOUT:   - { index:  8, kind:      "Identifier", line: {{ *}}[[@LINE-3]], column:  7, indent: 3, spelling: "x", identifier: 1 }
  // CHECK:STDOUT:   - { index:  9, kind:      "CloseParen", line: {{ *}}[[@LINE-4]], column:  8, indent: 3, spelling: ")", opening_token: 7 }
  // CHECK:STDOUT:   - { index: 10, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-5]], column: 10, indent: 3, spelling: "{", closing_token: 15, has_leading_space: true }
    foo();
    // CHECK:STDOUT:   - { index: 11, kind:      "Identifier", line: {{ *}}[[@LINE-1]], column:  5, indent: 5, spelling: "foo", identifier: 2, has_leading_space: true }
    // CHECK:STDOUT:   - { index: 12, kind:       "OpenParen", line: {{ *}}[[@LINE-2]], column:  8, indent: 5, spelling: "(", closing_token: 13 }
    // CHECK:STDOUT:   - { index: 13, kind:      "CloseParen", line: {{ *}}[[@LINE-3]], column:  9, indent: 5, spelling: ")", opening_token: 12 }
    // CHECK:STDOUT:   - { index: 14, kind:            "Semi", line: {{ *}}[[@LINE-4]], column: 10, indent: 5, spelling: ";" }
    // CHECK:STDOUT:   - { index: 15, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-5]], column: 11, indent: 5, spelling: "}", opening_token: 10, has_leading_space: true, recovery: true }
  // CHECK:STDERR: dedent_inside_scope.carbon:[[@LINE+4]]:3: note: possibly missing `}` here [PossiblyMissingBracketHere]
  // CHECK:STDERR:   bar();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  bar();
  // CHECK:STDOUT:   - { index: 16, kind:      "Identifier", line: {{ *}}[[@LINE-1]], column:  3, indent: 3, spelling: "bar", identifier: 3, has_leading_space: true }
  // CHECK:STDOUT:   - { index: 17, kind:       "OpenParen", line: {{ *}}[[@LINE-2]], column:  6, indent: 3, spelling: "(", closing_token: 18 }
  // CHECK:STDOUT:   - { index: 18, kind:      "CloseParen", line: {{ *}}[[@LINE-3]], column:  7, indent: 3, spelling: ")", opening_token: 17 }
  // CHECK:STDOUT:   - { index: 19, kind:            "Semi", line: {{ *}}[[@LINE-4]], column:  8, indent: 3, spelling: ";" }
}
// CHECK:STDOUT:   - { index: 20, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "}", opening_token: 5, has_leading_space: true }

// --- semi_in_paren.carbon
// CHECK:STDOUT: - filename: semi_in_paren.carbon
// CHECK:STDOUT:   tokens:

fn F() {
// CHECK:STDOUT:   - { index:  1, kind:              "Fn", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:  4, indent: 1, spelling: "F", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:       "OpenParen", line: {{ *}}[[@LINE-3]], column:  5, indent: 1, spelling: "(", closing_token: 4 }
// CHECK:STDOUT:   - { index:  4, kind:      "CloseParen", line: {{ *}}[[@LINE-4]], column:  6, indent: 1, spelling: ")", opening_token: 3 }
// CHECK:STDOUT:   - { index:  5, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-5]], column:  8, indent: 1, spelling: "{", closing_token: 17, has_leading_space: true }
  // CHECK:STDERR: semi_in_paren.carbon:[[@LINE+7]]:16: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
  // CHECK:STDERR:   var x: i32 = (1 + 2;
  // CHECK:STDERR:                ^
  // CHECK:STDERR: semi_in_paren.carbon:[[@LINE+4]]:22: note: possibly missing `)` here [PossiblyMissingBracketHere]
  // CHECK:STDERR:   var x: i32 = (1 + 2;
  // CHECK:STDERR:                      ^
  // CHECK:STDERR:
  var x: i32 = (1 + 2;
  // CHECK:STDOUT:   - { index:  6, kind:             "Var", line: {{ *}}[[@LINE-1]], column:  3, indent: 3, spelling: "var", has_leading_space: true }
  // CHECK:STDOUT:   - { index:  7, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:  7, indent: 3, spelling: "x", identifier: 1, has_leading_space: true }
  // CHECK:STDOUT:   - { index:  8, kind:           "Colon", line: {{ *}}[[@LINE-3]], column:  8, indent: 3, spelling: ":" }
  // CHECK:STDOUT:   - { index:  9, kind:  "IntTypeLiteral", line: {{ *}}[[@LINE-4]], column: 10, indent: 3, spelling: "i32", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 10, kind:           "Equal", line: {{ *}}[[@LINE-5]], column: 14, indent: 3, spelling: "=", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 11, kind:       "OpenParen", line: {{ *}}[[@LINE-6]], column: 16, indent: 3, spelling: "(", closing_token: 15, has_leading_space: true }
  // CHECK:STDOUT:   - { index: 12, kind:      "IntLiteral", line: {{ *}}[[@LINE-7]], column: 17, indent: 3, spelling: "1", value: "1" }
  // CHECK:STDOUT:   - { index: 13, kind:            "Plus", line: {{ *}}[[@LINE-8]], column: 19, indent: 3, spelling: "+", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 14, kind:      "IntLiteral", line: {{ *}}[[@LINE-9]], column: 21, indent: 3, spelling: "2", value: "2", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 15, kind:      "CloseParen", line: {{ *}}[[@LINE-10]], column: 22, indent: 3, spelling: ")", opening_token: 11, recovery: true }
  // CHECK:STDOUT:   - { index: 16, kind:            "Semi", line: {{ *}}[[@LINE-11]], column: 22, indent: 3, spelling: ";" }
}
// CHECK:STDOUT:   - { index: 17, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "}", opening_token: 5, has_leading_space: true }

// --- scope_in_paren.carbon
// CHECK:STDOUT: - filename: scope_in_paren.carbon
// CHECK:STDOUT:   tokens:

// CHECK:STDERR: scope_in_paren.carbon:[[@LINE+7]]:14: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
// CHECK:STDERR: var x: i32 = (1 + 2;
// CHECK:STDERR:              ^
// CHECK:STDERR: scope_in_paren.carbon:[[@LINE+4]]:20: note: possibly missing `)` here [PossiblyMissingBracketHere]
// CHECK:STDERR: var x: i32 = (1 + 2;
// CHECK:STDERR:                    ^
// CHECK:STDERR:
var x: i32 = (1 + 2;
// CHECK:STDOUT:   - { index:  1, kind:             "Var", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "var", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:  5, indent: 1, spelling: "x", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:           "Colon", line: {{ *}}[[@LINE-3]], column:  6, indent: 1, spelling: ":" }
// CHECK:STDOUT:   - { index:  4, kind:  "IntTypeLiteral", line: {{ *}}[[@LINE-4]], column:  8, indent: 1, spelling: "i32", has_leading_space: true }
// CHECK:STDOUT:   - { index:  5, kind:           "Equal", line: {{ *}}[[@LINE-5]], column: 12, indent: 1, spelling: "=", has_leading_space: true }
// CHECK:STDOUT:   - { index:  6, kind:       "OpenParen", line: {{ *}}[[@LINE-6]], column: 14, indent: 1, spelling: "(", closing_token: 10, has_leading_space: true }
// CHECK:STDOUT:   - { index:  7, kind:      "IntLiteral", line: {{ *}}[[@LINE-7]], column: 15, indent: 1, spelling: "1", value: "1" }
// CHECK:STDOUT:   - { index:  8, kind:            "Plus", line: {{ *}}[[@LINE-8]], column: 17, indent: 1, spelling: "+", has_leading_space: true }
// CHECK:STDOUT:   - { index:  9, kind:      "IntLiteral", line: {{ *}}[[@LINE-9]], column: 19, indent: 1, spelling: "2", value: "2", has_leading_space: true }
// CHECK:STDOUT:   - { index: 10, kind:      "CloseParen", line: {{ *}}[[@LINE-10]], column: 20, indent: 1, spelling: ")", opening_token: 6, recovery: true }
// CHECK:STDOUT:   - { index: 11, kind:            "Semi", line: {{ *}}[[@LINE-11]], column: 20, indent: 1, spelling: ";" }

fn F() {
// CHECK:STDOUT:   - { index: 12, kind:              "Fn", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index: 13, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:  4, indent: 1, spelling: "F", identifier: 1, has_leading_space: true }
// CHECK:STDOUT:   - { index: 14, kind:       "OpenParen", line: {{ *}}[[@LINE-3]], column:  5, indent: 1, spelling: "(", closing_token: 15 }
// CHECK:STDOUT:   - { index: 15, kind:      "CloseParen", line: {{ *}}[[@LINE-4]], column:  6, indent: 1, spelling: ")", opening_token: 14 }
// CHECK:STDOUT:   - { index: 16, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-5]], column:  8, indent: 1, spelling: "{", closing_token: 17, has_leading_space: true }
}
// CHECK:STDOUT:   - { index: 17, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "}", opening_token: 16, has_leading_space: true }

// --- mismatched_kinds.carbon
// CHECK:STDOUT: - filename: mismatched_kinds.carbon
// CHECK:STDOUT:   tokens:

fn F() {
// CHECK:STDOUT:   - { index:  1, kind:              "Fn", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:  4, indent: 1, spelling: "F", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:       "OpenParen", line: {{ *}}[[@LINE-3]], column:  5, indent: 1, spelling: "(", closing_token: 4 }
// CHECK:STDOUT:   - { index:  4, kind:      "CloseParen", line: {{ *}}[[@LINE-4]], column:  6, indent: 1, spelling: ")", opening_token: 3 }
// CHECK:STDOUT:   - { index:  5, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-5]], column:  8, indent: 1, spelling: "{", closing_token: 18, has_leading_space: true }
  // CHECK:STDERR: mismatched_kinds.carbon:[[@LINE+11]]:16: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
  // CHECK:STDERR:   var x: i32 = (1 + 2};
  // CHECK:STDERR:                ^
  // CHECK:STDERR: mismatched_kinds.carbon:[[@LINE+8]]:23: note: possibly missing `)` here [PossiblyMissingBracketHere]
  // CHECK:STDERR:   var x: i32 = (1 + 2};
  // CHECK:STDERR:                       ^
  // CHECK:STDERR:
  // CHECK:STDERR: mismatched_kinds.carbon:[[@LINE+4]]:22: error: closing symbol without a corresponding opening symbol [UnmatchedClosing]
  // CHECK:STDERR:   var x: i32 = (1 + 2};
  // CHECK:STDERR:                      ^
  // CHECK:STDERR:
  var x: i32 = (1 + 2};
  // CHECK:STDOUT:   - { index:  6, kind:             "Var", line: {{ *}}[[@LINE-1]], column:  3, indent: 3, spelling: "var", has_leading_space: true }
  // CHECK:STDOUT:   - { index:  7, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:  7, indent: 3, spelling: "x", identifier: 1, has_leading_space: true }
  // CHECK:STDOUT:   - { index:  8, kind:           "Colon", line: {{ *}}[[@LINE-3]], column:  8, indent: 3, spelling: ":" }
  // CHECK:STDOUT:   - { index:  9, kind:  "IntTypeLiteral", line: {{ *}}[[@LINE-4]], column: 10, indent: 3, spelling: "i32", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 10, kind:           "Equal", line: {{ *}}[[@LINE-5]], column: 14, indent: 3, spelling: "=", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 11, kind:       "OpenParen", line: {{ *}}[[@LINE-6]], column: 16, indent: 3, spelling: "(", closing_token: 16, has_leading_space: true }
  // CHECK:STDOUT:   - { index: 12, kind:      "IntLiteral", line: {{ *}}[[@LINE-7]], column: 17, indent: 3, spelling: "1", value: "1" }
  // CHECK:STDOUT:   - { index: 13, kind:            "Plus", line: {{ *}}[[@LINE-8]], column: 19, indent: 3, spelling: "+", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 14, kind:      "IntLiteral", line: {{ *}}[[@LINE-9]], column: 21, indent: 3, spelling: "2", value: "2", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 15, kind:           "Error", line: {{ *}}[[@LINE-10]], column: 22, indent: 3, spelling: "}" }
  // CHECK:STDOUT:   - { index: 16, kind:      "CloseParen", line: {{ *}}[[@LINE-11]], column: 23, indent: 3, spelling: ")", opening_token: 11, recovery: true }
  // CHECK:STDOUT:   - { index: 17, kind:            "Semi", line: {{ *}}[[@LINE-12]], column: 23, indent: 3, spelling: ";" }
}
// CHECK:STDOUT:   - { index: 18, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "}", opening_token: 5, has_leading_space: true }

// --- unclosed_at_eof.carbon
// CHECK:STDOUT: - filename: unclosed_at_eof.carbon
// CHECK:STDOUT:   tokens:

// CHECK:STDERR: unclosed_at_eof.carbon:[[@LINE+3]]:8: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
// CHECK:STDERR: fn F() {
// CHECK:STDERR:        ^
fn F() {
// CHECK:STDOUT:   - { index:  1, kind:              "Fn", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:  4, indent: 1, spelling: "F", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:       "OpenParen", line: {{ *}}[[@LINE-3]], column:  5, indent: 1, spelling: "(", closing_token: 4 }
// CHECK:STDOUT:   - { index:  4, kind:      "CloseParen", line: {{ *}}[[@LINE-4]], column:  6, indent: 1, spelling: ")", opening_token: 3 }
// CHECK:STDOUT:   - { index:  5, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-5]], column:  8, indent: 1, spelling: "{", closing_token: 13, has_leading_space: true }
  var x: i32 = 1;
  // CHECK:STDOUT:   - { index:  6, kind:             "Var", line: {{ *}}[[@LINE-1]], column:  3, indent: 3, spelling: "var", has_leading_space: true }
  // CHECK:STDOUT:   - { index:  7, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:  7, indent: 3, spelling: "x", identifier: 1, has_leading_space: true }
  // CHECK:STDOUT:   - { index:  8, kind:           "Colon", line: {{ *}}[[@LINE-3]], column:  8, indent: 3, spelling: ":" }
  // CHECK:STDOUT:   - { index:  9, kind:  "IntTypeLiteral", line: {{ *}}[[@LINE-4]], column: 10, indent: 3, spelling: "i32", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 10, kind:           "Equal", line: {{ *}}[[@LINE-5]], column: 14, indent: 3, spelling: "=", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 11, kind:      "IntLiteral", line: {{ *}}[[@LINE-6]], column: 16, indent: 3, spelling: "1", value: "1", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 12, kind:            "Semi", line: {{ *}}[[@LINE-7]], column: 17, indent: 3, spelling: ";" }
  // CHECK:STDOUT:   - { index: 13, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-8]], column: 18, indent: 3, spelling: "}", opening_token: 5, has_leading_space: true, recovery: true }

// CHECK:STDERR: unclosed_at_eof.carbon:[[@LINE+3]]:17: note: possibly missing `}` here [PossiblyMissingBracketHere]
// CHECK:STDERR: // CHECK:STDERR:
// CHECK:STDERR:                 ^
// CHECK:STDERR:
