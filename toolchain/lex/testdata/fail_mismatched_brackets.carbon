// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lex/testdata/fail_mismatched_brackets.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lex/testdata/fail_mismatched_brackets.carbon

// Recovery picks the cheapest way to make a damaged region well-bracketed, so
// each case below is really about which cue makes one repair cheaper than the
// others. Cases where the repair is a `}` live in
// `fail_mismatched_brackets_close_brace.carbon` instead, because those notes
// point at the following line and so can't have `CHECK`s written there.

// --- fail_top_level_brackets.carbon
// CHECK:STDOUT: - filename: fail_top_level_brackets.carbon
// CHECK:STDOUT:   tokens:

// The `}`s here have no plausible opener, so they're replaced with error tokens
// rather than guessed at. The trailing `[` is closed immediately, since there's
// nothing left for it to contain.

// CHECK:STDERR: fail_top_level_brackets.carbon:[[@LINE+2]]:1: error: closing symbol without a corresponding opening symbol [UnmatchedClosing]
// CHECK:STDERR:
}
// CHECK:STDOUT:   - { index: 1, kind:              "Error", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "}", has_leading_space: true }

// CHECK:STDERR: fail_top_level_brackets.carbon:[[@LINE+2]]:3: error: closing symbol without a corresponding opening symbol [UnmatchedClosing]
// CHECK:STDERR:
( } )
// CHECK:STDOUT:   - { index: 2, kind:          "OpenParen", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "(", closing_token: 4, has_leading_space: true }
// CHECK:STDOUT:   - { index: 3, kind:              "Error", line: {{ *}}[[@LINE-2]], column: 3, indent: 1, spelling: "}", has_leading_space: true }
// CHECK:STDOUT:   - { index: 4, kind:         "CloseParen", line: {{ *}}[[@LINE-3]], column: 5, indent: 1, spelling: ")", opening_token: 2, has_leading_space: true }

// CHECK:STDERR: fail_top_level_brackets.carbon:[[@LINE+3]]:1: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
// CHECK:STDERR: fail_top_level_brackets.carbon:[[@LINE+2]]:2: note: possibly missing `]` here [PossiblyMissingBracketHere]
// CHECK:STDERR:
[
// CHECK:STDOUT:   - { index: 5, kind:  "OpenSquareBracket", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "[", closing_token: 6, has_leading_space: true }
// CHECK:STDOUT:   - { index: 6, kind: "CloseSquareBracket", line: {{ *}}[[@LINE-2]], column: 2, indent: 1, spelling: "]", opening_token: 5, has_leading_space: true, recovery: true }

// --- fail_semi_in_paren.carbon
// CHECK:STDOUT: - filename: fail_semi_in_paren.carbon
// CHECK:STDOUT:   tokens:

// A `;` ends a statement, so it can't appear inside parens: the group has to
// have ended before it.

fn F() {
// CHECK:STDOUT:   - { index:  1, kind:              "Fn", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:  4, indent: 1, spelling: "F", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:       "OpenParen", line: {{ *}}[[@LINE-3]], column:  5, indent: 1, spelling: "(", closing_token: 4 }
// CHECK:STDOUT:   - { index:  4, kind:      "CloseParen", line: {{ *}}[[@LINE-4]], column:  6, indent: 1, spelling: ")", opening_token: 3 }
// CHECK:STDOUT:   - { index:  5, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-5]], column:  8, indent: 1, spelling: "{", closing_token: 17, has_leading_space: true }
  // CHECK:STDERR: fail_semi_in_paren.carbon:[[@LINE+3]]:16: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
  // CHECK:STDERR: fail_semi_in_paren.carbon:[[@LINE+2]]:22: note: possibly missing `)` here [PossiblyMissingBracketHere]
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

// --- fail_scope_in_paren.carbon
// CHECK:STDOUT: - filename: fail_scope_in_paren.carbon
// CHECK:STDOUT:   tokens:

// As above, at the top level rather than inside a function.

// CHECK:STDERR: fail_scope_in_paren.carbon:[[@LINE+3]]:14: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
// CHECK:STDERR: fail_scope_in_paren.carbon:[[@LINE+2]]:20: note: possibly missing `)` here [PossiblyMissingBracketHere]
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

// --- fail_close_paren_before_brace.carbon
// CHECK:STDOUT: - filename: fail_close_paren_before_brace.carbon
// CHECK:STDOUT:   tokens:

// A `(` can't still be open when a block `{` starts, so an `if` condition's
// paren closes before the brace.

fn F() {
// CHECK:STDOUT:   - { index:  1, kind:              "Fn", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column: 4, indent: 1, spelling: "F", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:       "OpenParen", line: {{ *}}[[@LINE-3]], column: 5, indent: 1, spelling: "(", closing_token: 4 }
// CHECK:STDOUT:   - { index:  4, kind:      "CloseParen", line: {{ *}}[[@LINE-4]], column: 6, indent: 1, spelling: ")", opening_token: 3 }
// CHECK:STDOUT:   - { index:  5, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-5]], column: 8, indent: 1, spelling: "{", closing_token: 16, has_leading_space: true }
  // CHECK:STDERR: fail_close_paren_before_brace.carbon:[[@LINE+3]]:6: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
  // CHECK:STDERR: fail_close_paren_before_brace.carbon:[[@LINE+2]]:8: note: possibly missing `)` here [PossiblyMissingBracketHere]
  // CHECK:STDERR:
  if (C {
  // CHECK:STDOUT:   - { index:  6, kind:              "If", line: {{ *}}[[@LINE-1]], column: 3, indent: 3, spelling: "if", has_leading_space: true }
  // CHECK:STDOUT:   - { index:  7, kind:       "OpenParen", line: {{ *}}[[@LINE-2]], column: 6, indent: 3, spelling: "(", closing_token: 9, has_leading_space: true }
  // CHECK:STDOUT:   - { index:  8, kind:      "Identifier", line: {{ *}}[[@LINE-3]], column: 7, indent: 3, spelling: "C", identifier: 1 }
  // CHECK:STDOUT:   - { index:  9, kind:      "CloseParen", line: {{ *}}[[@LINE-4]], column: 8, indent: 3, spelling: ")", opening_token: 7, has_leading_space: true, recovery: true }
  // CHECK:STDOUT:   - { index: 10, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-5]], column: 9, indent: 3, spelling: "{", closing_token: 15 }
    G();
    // CHECK:STDOUT:   - { index: 11, kind:      "Identifier", line: {{ *}}[[@LINE-1]], column: 5, indent: 5, spelling: "G", identifier: 2, has_leading_space: true }
    // CHECK:STDOUT:   - { index: 12, kind:       "OpenParen", line: {{ *}}[[@LINE-2]], column: 6, indent: 5, spelling: "(", closing_token: 13 }
    // CHECK:STDOUT:   - { index: 13, kind:      "CloseParen", line: {{ *}}[[@LINE-3]], column: 7, indent: 5, spelling: ")", opening_token: 12 }
    // CHECK:STDOUT:   - { index: 14, kind:            "Semi", line: {{ *}}[[@LINE-4]], column: 8, indent: 5, spelling: ";" }
  }
  // CHECK:STDOUT:   - { index: 15, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column: 3, indent: 3, spelling: "}", opening_token: 10, has_leading_space: true }
}
// CHECK:STDOUT:   - { index: 16, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "}", opening_token: 5, has_leading_space: true }

// --- fail_close_paren_before_arrow.carbon
// CHECK:STDOUT: - filename: fail_close_paren_before_arrow.carbon
// CHECK:STDOUT:   tokens:

// `->` structures a declaration rather than an expression, so a parameter
// list's paren must have closed before it.

// CHECK:STDERR: fail_close_paren_before_arrow.carbon:[[@LINE+3]]:5: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
// CHECK:STDERR: fail_close_paren_before_arrow.carbon:[[@LINE+2]]:12: note: possibly missing `)` here [PossiblyMissingBracketHere]
// CHECK:STDERR:
fn F(x: i32 -> i32 {
// CHECK:STDOUT:   - { index:  1, kind:              "Fn", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:  4, indent: 1, spelling: "F", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:       "OpenParen", line: {{ *}}[[@LINE-3]], column:  5, indent: 1, spelling: "(", closing_token: 7 }
// CHECK:STDOUT:   - { index:  4, kind:      "Identifier", line: {{ *}}[[@LINE-4]], column:  6, indent: 1, spelling: "x", identifier: 1 }
// CHECK:STDOUT:   - { index:  5, kind:           "Colon", line: {{ *}}[[@LINE-5]], column:  7, indent: 1, spelling: ":" }
// CHECK:STDOUT:   - { index:  6, kind:  "IntTypeLiteral", line: {{ *}}[[@LINE-6]], column:  9, indent: 1, spelling: "i32", has_leading_space: true }
// CHECK:STDOUT:   - { index:  7, kind:      "CloseParen", line: {{ *}}[[@LINE-7]], column: 12, indent: 1, spelling: ")", opening_token: 3, has_leading_space: true, recovery: true }
// CHECK:STDOUT:   - { index:  8, kind:    "MinusGreater", line: {{ *}}[[@LINE-8]], column: 13, indent: 1, spelling: "->" }
// CHECK:STDOUT:   - { index:  9, kind:  "IntTypeLiteral", line: {{ *}}[[@LINE-9]], column: 16, indent: 1, spelling: "i32", has_leading_space: true }
// CHECK:STDOUT:   - { index: 10, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-10]], column: 20, indent: 1, spelling: "{", closing_token: 14, has_leading_space: true }
  return x;
  // CHECK:STDOUT:   - { index: 11, kind:          "Return", line: {{ *}}[[@LINE-1]], column:  3, indent: 3, spelling: "return", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 12, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column: 10, indent: 3, spelling: "x", identifier: 1, has_leading_space: true }
  // CHECK:STDOUT:   - { index: 13, kind:            "Semi", line: {{ *}}[[@LINE-3]], column: 11, indent: 3, spelling: ";" }
}
// CHECK:STDOUT:   - { index: 14, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "}", opening_token: 10, has_leading_space: true }

// --- fail_close_paren_at_wide_gap.carbon
// CHECK:STDOUT: - filename: fail_close_paren_at_wide_gap.carbon
// CHECK:STDOUT:   tokens:

// Formatted code has no space before a `)`, so a wide mid-line gap is evidence
// that the closer used to be there. Note that the repair goes in the gap, not
// at the `;`, which would also have been well-bracketed.

fn F() {
// CHECK:STDOUT:   - { index:  1, kind:              "Fn", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:  4, indent: 1, spelling: "F", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:       "OpenParen", line: {{ *}}[[@LINE-3]], column:  5, indent: 1, spelling: "(", closing_token: 4 }
// CHECK:STDOUT:   - { index:  4, kind:      "CloseParen", line: {{ *}}[[@LINE-4]], column:  6, indent: 1, spelling: ")", opening_token: 3 }
// CHECK:STDOUT:   - { index:  5, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-5]], column:  8, indent: 1, spelling: "{", closing_token: 13, has_leading_space: true }
  // CHECK:STDERR: fail_close_paren_at_wide_gap.carbon:[[@LINE+3]]:4: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
  // CHECK:STDERR: fail_close_paren_at_wide_gap.carbon:[[@LINE+2]]:6: note: possibly missing `)` here [PossiblyMissingBracketHere]
  // CHECK:STDERR:
  G(1  + 2;
  // CHECK:STDOUT:   - { index:  6, kind:      "Identifier", line: {{ *}}[[@LINE-1]], column:  3, indent: 3, spelling: "G", identifier: 1, has_leading_space: true }
  // CHECK:STDOUT:   - { index:  7, kind:       "OpenParen", line: {{ *}}[[@LINE-2]], column:  4, indent: 3, spelling: "(", closing_token: 9 }
  // CHECK:STDOUT:   - { index:  8, kind:      "IntLiteral", line: {{ *}}[[@LINE-3]], column:  5, indent: 3, spelling: "1", value: "1" }
  // CHECK:STDOUT:   - { index:  9, kind:      "CloseParen", line: {{ *}}[[@LINE-4]], column:  6, indent: 3, spelling: ")", opening_token: 7, has_leading_space: true, recovery: true }
  // CHECK:STDOUT:   - { index: 10, kind:            "Plus", line: {{ *}}[[@LINE-5]], column:  8, indent: 3, spelling: "+" }
  // CHECK:STDOUT:   - { index: 11, kind:      "IntLiteral", line: {{ *}}[[@LINE-6]], column: 10, indent: 3, spelling: "2", value: "2", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 12, kind:            "Semi", line: {{ *}}[[@LINE-7]], column: 11, indent: 3, spelling: ";" }
}
// CHECK:STDOUT:   - { index: 13, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "}", opening_token: 5, has_leading_space: true }

// --- fail_paren_cascade.carbon
// CHECK:STDOUT: - filename: fail_paren_cascade.carbon
// CHECK:STDOUT:   tokens:

// Several groups can end at the same point, so a single `;` closes all three.

fn F() {
// CHECK:STDOUT:   - { index:  1, kind:              "Fn", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:  4, indent: 1, spelling: "F", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:       "OpenParen", line: {{ *}}[[@LINE-3]], column:  5, indent: 1, spelling: "(", closing_token: 4 }
// CHECK:STDOUT:   - { index:  4, kind:      "CloseParen", line: {{ *}}[[@LINE-4]], column:  6, indent: 1, spelling: ")", opening_token: 3 }
// CHECK:STDOUT:   - { index:  5, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-5]], column:  8, indent: 1, spelling: "{", closing_token: 22, has_leading_space: true }
  // CHECK:STDERR: fail_paren_cascade.carbon:[[@LINE+9]]:17: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
  // CHECK:STDERR: fail_paren_cascade.carbon:[[@LINE+8]]:23: note: possibly missing `)` here [PossiblyMissingBracketHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_paren_cascade.carbon:[[@LINE+6]]:19: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
  // CHECK:STDERR: fail_paren_cascade.carbon:[[@LINE+5]]:23: note: possibly missing `)` here [PossiblyMissingBracketHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_paren_cascade.carbon:[[@LINE+3]]:21: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
  // CHECK:STDERR: fail_paren_cascade.carbon:[[@LINE+2]]:23: note: possibly missing `)` here [PossiblyMissingBracketHere]
  // CHECK:STDERR:
  var x: i32 = G(H(I(1;
  // CHECK:STDOUT:   - { index:  6, kind:             "Var", line: {{ *}}[[@LINE-1]], column:  3, indent: 3, spelling: "var", has_leading_space: true }
  // CHECK:STDOUT:   - { index:  7, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:  7, indent: 3, spelling: "x", identifier: 1, has_leading_space: true }
  // CHECK:STDOUT:   - { index:  8, kind:           "Colon", line: {{ *}}[[@LINE-3]], column:  8, indent: 3, spelling: ":" }
  // CHECK:STDOUT:   - { index:  9, kind:  "IntTypeLiteral", line: {{ *}}[[@LINE-4]], column: 10, indent: 3, spelling: "i32", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 10, kind:           "Equal", line: {{ *}}[[@LINE-5]], column: 14, indent: 3, spelling: "=", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 11, kind:      "Identifier", line: {{ *}}[[@LINE-6]], column: 16, indent: 3, spelling: "G", identifier: 2, has_leading_space: true }
  // CHECK:STDOUT:   - { index: 12, kind:       "OpenParen", line: {{ *}}[[@LINE-7]], column: 17, indent: 3, spelling: "(", closing_token: 20 }
  // CHECK:STDOUT:   - { index: 13, kind:      "Identifier", line: {{ *}}[[@LINE-8]], column: 18, indent: 3, spelling: "H", identifier: 3 }
  // CHECK:STDOUT:   - { index: 14, kind:       "OpenParen", line: {{ *}}[[@LINE-9]], column: 19, indent: 3, spelling: "(", closing_token: 19 }
  // CHECK:STDOUT:   - { index: 15, kind:      "Identifier", line: {{ *}}[[@LINE-10]], column: 20, indent: 3, spelling: "I", identifier: 4 }
  // CHECK:STDOUT:   - { index: 16, kind:       "OpenParen", line: {{ *}}[[@LINE-11]], column: 21, indent: 3, spelling: "(", closing_token: 18 }
  // CHECK:STDOUT:   - { index: 17, kind:      "IntLiteral", line: {{ *}}[[@LINE-12]], column: 22, indent: 3, spelling: "1", value: "1" }
  // CHECK:STDOUT:   - { index: 18, kind:      "CloseParen", line: {{ *}}[[@LINE-13]], column: 23, indent: 3, spelling: ")", opening_token: 16, recovery: true }
  // CHECK:STDOUT:   - { index: 19, kind:      "CloseParen", line: {{ *}}[[@LINE-14]], column: 23, indent: 3, spelling: ")", opening_token: 14, recovery: true }
  // CHECK:STDOUT:   - { index: 20, kind:      "CloseParen", line: {{ *}}[[@LINE-15]], column: 23, indent: 3, spelling: ")", opening_token: 12, recovery: true }
  // CHECK:STDOUT:   - { index: 21, kind:            "Semi", line: {{ *}}[[@LINE-16]], column: 23, indent: 3, spelling: ";" }
}
// CHECK:STDOUT:   - { index: 22, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "}", opening_token: 5, has_leading_space: true }

// --- fail_close_struct_before_semi.carbon
// CHECK:STDOUT: - filename: fail_close_struct_before_semi.carbon
// CHECK:STDOUT:   tokens:

// A `{` introducing a struct literal is tracked separately from a scope brace,
// and closes at the end of the statement.

fn F() {
// CHECK:STDOUT:   - { index:  1, kind:              "Fn", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:  4, indent: 1, spelling: "F", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:       "OpenParen", line: {{ *}}[[@LINE-3]], column:  5, indent: 1, spelling: "(", closing_token: 4 }
// CHECK:STDOUT:   - { index:  4, kind:      "CloseParen", line: {{ *}}[[@LINE-4]], column:  6, indent: 1, spelling: ")", opening_token: 3 }
// CHECK:STDOUT:   - { index:  5, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-5]], column:  8, indent: 1, spelling: "{", closing_token: 23, has_leading_space: true }
  // CHECK:STDERR: fail_close_struct_before_semi.carbon:[[@LINE+3]]:17: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
  // CHECK:STDERR: fail_close_struct_before_semi.carbon:[[@LINE+2]]:32: note: possibly missing `}` here [PossiblyMissingBracketHere]
  // CHECK:STDERR:
  var x: auto = {.a = 1, .b = 2;
  // CHECK:STDOUT:   - { index:  6, kind:             "Var", line: {{ *}}[[@LINE-1]], column:  3, indent: 3, spelling: "var", has_leading_space: true }
  // CHECK:STDOUT:   - { index:  7, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:  7, indent: 3, spelling: "x", identifier: 1, has_leading_space: true }
  // CHECK:STDOUT:   - { index:  8, kind:           "Colon", line: {{ *}}[[@LINE-3]], column:  8, indent: 3, spelling: ":" }
  // CHECK:STDOUT:   - { index:  9, kind:            "Auto", line: {{ *}}[[@LINE-4]], column: 10, indent: 3, spelling: "auto", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 10, kind:           "Equal", line: {{ *}}[[@LINE-5]], column: 15, indent: 3, spelling: "=", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 11, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-6]], column: 17, indent: 3, spelling: "{", closing_token: 21, has_leading_space: true }
  // CHECK:STDOUT:   - { index: 12, kind:          "Period", line: {{ *}}[[@LINE-7]], column: 18, indent: 3, spelling: "." }
  // CHECK:STDOUT:   - { index: 13, kind:      "Identifier", line: {{ *}}[[@LINE-8]], column: 19, indent: 3, spelling: "a", identifier: 2 }
  // CHECK:STDOUT:   - { index: 14, kind:           "Equal", line: {{ *}}[[@LINE-9]], column: 21, indent: 3, spelling: "=", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 15, kind:      "IntLiteral", line: {{ *}}[[@LINE-10]], column: 23, indent: 3, spelling: "1", value: "1", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 16, kind:           "Comma", line: {{ *}}[[@LINE-11]], column: 24, indent: 3, spelling: "," }
  // CHECK:STDOUT:   - { index: 17, kind:          "Period", line: {{ *}}[[@LINE-12]], column: 26, indent: 3, spelling: ".", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 18, kind:      "Identifier", line: {{ *}}[[@LINE-13]], column: 27, indent: 3, spelling: "b", identifier: 3 }
  // CHECK:STDOUT:   - { index: 19, kind:           "Equal", line: {{ *}}[[@LINE-14]], column: 29, indent: 3, spelling: "=", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 20, kind:      "IntLiteral", line: {{ *}}[[@LINE-15]], column: 31, indent: 3, spelling: "2", value: "2", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 21, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-16]], column: 32, indent: 3, spelling: "}", opening_token: 11, recovery: true }
  // CHECK:STDOUT:   - { index: 22, kind:            "Semi", line: {{ *}}[[@LINE-17]], column: 32, indent: 3, spelling: ";" }
}
// CHECK:STDOUT:   - { index: 23, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "}", opening_token: 5, has_leading_space: true }

// --- fail_mismatched_kinds.carbon
// CHECK:STDOUT: - filename: fail_mismatched_kinds.carbon
// CHECK:STDOUT:   tokens:

// The `}` can't match the `(`, so the two are repaired separately: a `)` closes
// the paren, and a `{` before the `}` explains it as an empty struct literal.
// Both insertions land at the same point, and each is reported against the
// bracket it repairs.

fn F() {
// CHECK:STDOUT:   - { index:  1, kind:              "Fn", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:  4, indent: 1, spelling: "F", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:       "OpenParen", line: {{ *}}[[@LINE-3]], column:  5, indent: 1, spelling: "(", closing_token: 4 }
// CHECK:STDOUT:   - { index:  4, kind:      "CloseParen", line: {{ *}}[[@LINE-4]], column:  6, indent: 1, spelling: ")", opening_token: 3 }
// CHECK:STDOUT:   - { index:  5, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-5]], column:  8, indent: 1, spelling: "{", closing_token: 19, has_leading_space: true }
  // CHECK:STDERR: fail_mismatched_kinds.carbon:[[@LINE+6]]:16: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
  // CHECK:STDERR: fail_mismatched_kinds.carbon:[[@LINE+5]]:22: note: possibly missing `)` here [PossiblyMissingBracketHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_mismatched_kinds.carbon:[[@LINE+3]]:22: error: closing symbol without a corresponding opening symbol [UnmatchedClosing]
  // CHECK:STDERR: fail_mismatched_kinds.carbon:[[@LINE+2]]:22: note: possibly missing `{` here [PossiblyMissingBracketHere]
  // CHECK:STDERR:
  var x: i32 = (1 + 2};
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
  // CHECK:STDOUT:   - { index: 16, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-11]], column: 22, indent: 3, spelling: "{", closing_token: 17, recovery: true }
  // CHECK:STDOUT:   - { index: 17, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-12]], column: 22, indent: 3, spelling: "}", opening_token: 16 }
  // CHECK:STDOUT:   - { index: 18, kind:            "Semi", line: {{ *}}[[@LINE-13]], column: 23, indent: 3, spelling: ";" }
}
// CHECK:STDOUT:   - { index: 19, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "}", opening_token: 5, has_leading_space: true }

// --- fail_missing_open_paren_after_if.carbon
// CHECK:STDOUT: - filename: fail_missing_open_paren_after_if.carbon
// CHECK:STDOUT:   tokens:

// `if` must be followed by `(`, which makes inserting one much cheaper than
// discarding the `)`.

fn F() {
// CHECK:STDOUT:   - { index:  1, kind:              "Fn", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:  4, indent: 1, spelling: "F", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:       "OpenParen", line: {{ *}}[[@LINE-3]], column:  5, indent: 1, spelling: "(", closing_token: 4 }
// CHECK:STDOUT:   - { index:  4, kind:      "CloseParen", line: {{ *}}[[@LINE-4]], column:  6, indent: 1, spelling: ")", opening_token: 3 }
// CHECK:STDOUT:   - { index:  5, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-5]], column:  8, indent: 1, spelling: "{", closing_token: 18, has_leading_space: true }
  // CHECK:STDERR: fail_missing_open_paren_after_if.carbon:[[@LINE+3]]:11: error: closing symbol without a corresponding opening symbol [UnmatchedClosing]
  // CHECK:STDERR: fail_missing_open_paren_after_if.carbon:[[@LINE+2]]:6: note: possibly missing `(` here [PossiblyMissingBracketHere]
  // CHECK:STDERR:
  if C > 0) {
  // CHECK:STDOUT:   - { index:  6, kind:              "If", line: {{ *}}[[@LINE-1]], column:  3, indent: 3, spelling: "if", has_leading_space: true }
  // CHECK:STDOUT:   - { index:  7, kind:       "OpenParen", line: {{ *}}[[@LINE-2]], column:  5, indent: 3, spelling: "(", closing_token: 11, has_leading_space: true, recovery: true }
  // CHECK:STDOUT:   - { index:  8, kind:      "Identifier", line: {{ *}}[[@LINE-3]], column:  6, indent: 3, spelling: "C", identifier: 1 }
  // CHECK:STDOUT:   - { index:  9, kind:         "Greater", line: {{ *}}[[@LINE-4]], column:  8, indent: 3, spelling: ">", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 10, kind:      "IntLiteral", line: {{ *}}[[@LINE-5]], column: 10, indent: 3, spelling: "0", value: "0", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 11, kind:      "CloseParen", line: {{ *}}[[@LINE-6]], column: 11, indent: 3, spelling: ")", opening_token: 7 }
  // CHECK:STDOUT:   - { index: 12, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-7]], column: 13, indent: 3, spelling: "{", closing_token: 17, has_leading_space: true }
    G();
    // CHECK:STDOUT:   - { index: 13, kind:      "Identifier", line: {{ *}}[[@LINE-1]], column:  5, indent: 5, spelling: "G", identifier: 2, has_leading_space: true }
    // CHECK:STDOUT:   - { index: 14, kind:       "OpenParen", line: {{ *}}[[@LINE-2]], column:  6, indent: 5, spelling: "(", closing_token: 15 }
    // CHECK:STDOUT:   - { index: 15, kind:      "CloseParen", line: {{ *}}[[@LINE-3]], column:  7, indent: 5, spelling: ")", opening_token: 14 }
    // CHECK:STDOUT:   - { index: 16, kind:            "Semi", line: {{ *}}[[@LINE-4]], column:  8, indent: 5, spelling: ";" }
  }
  // CHECK:STDOUT:   - { index: 17, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column:  3, indent: 3, spelling: "}", opening_token: 12, has_leading_space: true }
}
// CHECK:STDOUT:   - { index: 18, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "}", opening_token: 5, has_leading_space: true }

// --- fail_missing_open_square_after_forall.carbon
// CHECK:STDOUT: - filename: fail_missing_open_square_after_forall.carbon
// CHECK:STDOUT:   tokens:

// As above for `forall`, which wants a `[` rather than a `(`.

// CHECK:STDERR: fail_missing_open_square_after_forall.carbon:[[@LINE+3]]:21: error: closing symbol without a corresponding opening symbol [UnmatchedClosing]
// CHECK:STDERR: fail_missing_open_square_after_forall.carbon:[[@LINE+2]]:13: note: possibly missing `[` here [PossiblyMissingBracketHere]
// CHECK:STDERR:
impl forall T:! type] X as Y {
// CHECK:STDOUT:   - { index:  1, kind:               "Impl", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "impl", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:             "Forall", line: {{ *}}[[@LINE-2]], column:  6, indent: 1, spelling: "forall", has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:  "OpenSquareBracket", line: {{ *}}[[@LINE-3]], column: 12, indent: 1, spelling: "[", closing_token: 8, has_leading_space: true, recovery: true }
// CHECK:STDOUT:   - { index:  4, kind:         "Identifier", line: {{ *}}[[@LINE-4]], column: 13, indent: 1, spelling: "T", identifier: 0 }
// CHECK:STDOUT:   - { index:  5, kind:              "Colon", line: {{ *}}[[@LINE-5]], column: 14, indent: 1, spelling: ":" }
// CHECK:STDOUT:   - { index:  6, kind:            "Exclaim", line: {{ *}}[[@LINE-6]], column: 15, indent: 1, spelling: "!" }
// CHECK:STDOUT:   - { index:  7, kind:               "Type", line: {{ *}}[[@LINE-7]], column: 17, indent: 1, spelling: "type", has_leading_space: true }
// CHECK:STDOUT:   - { index:  8, kind: "CloseSquareBracket", line: {{ *}}[[@LINE-8]], column: 21, indent: 1, spelling: "]", opening_token: 3 }
// CHECK:STDOUT:   - { index:  9, kind:         "Identifier", line: {{ *}}[[@LINE-9]], column: 23, indent: 1, spelling: "X", identifier: 1, has_leading_space: true }
// CHECK:STDOUT:   - { index: 10, kind:                 "As", line: {{ *}}[[@LINE-10]], column: 25, indent: 1, spelling: "as", has_leading_space: true }
// CHECK:STDOUT:   - { index: 11, kind:         "Identifier", line: {{ *}}[[@LINE-11]], column: 28, indent: 1, spelling: "Y", identifier: 2, has_leading_space: true }
// CHECK:STDOUT:   - { index: 12, kind:     "OpenCurlyBrace", line: {{ *}}[[@LINE-12]], column: 30, indent: 1, spelling: "{", closing_token: 13, has_leading_space: true }
}
// CHECK:STDOUT:   - { index: 13, kind:    "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "}", opening_token: 12, has_leading_space: true }

// --- fail_missing_brace_after_if.carbon
// CHECK:STDOUT: - filename: fail_missing_brace_after_if.carbon
// CHECK:STDOUT:   tokens:

// A statement header can be followed by a block, so the body of this `if` is
// explained by a missing `{` rather than by discarding the `}` that closes it.

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
    // CHECK:STDOUT:   - { index: 11, kind:      "Identifier", line: {{ *}}[[@LINE-1]], column:  5, indent: 5, spelling: "thing2", identifier: 2 }
    // CHECK:STDOUT:   - { index: 12, kind:            "Semi", line: {{ *}}[[@LINE-2]], column: 11, indent: 5, spelling: ";" }
  // CHECK:STDERR: fail_missing_brace_after_if.carbon:[[@LINE+3]]:3: error: closing symbol without a corresponding opening symbol [UnmatchedClosing]
  // CHECK:STDERR: fail_missing_brace_after_if.carbon:[[@LINE-10]]:14: note: possibly missing `{` here [PossiblyMissingBracketHere]
  // CHECK:STDERR:
  }
  // CHECK:STDOUT:   - { index: 13, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column:  3, indent: 3, spelling: "}", opening_token: 10, has_leading_space: true }
}
// CHECK:STDOUT:   - { index: 14, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "}", opening_token: 5, has_leading_space: true }

// --- fail_missing_brace_multi_line_header.carbon
// CHECK:STDOUT: - filename: fail_missing_brace_multi_line_header.carbon
// CHECK:STDOUT:   tokens:

// As above, where the header spans several lines: the `{` belongs at the end of
// the header, not at the start of the body.

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
  // CHECK:STDOUT:   - { index: 14, kind:         "Identifier", line: {{ *}}[[@LINE-1]], column:  3, indent: 3, spelling: "foo", identifier: 3 }
  // CHECK:STDOUT:   - { index: 15, kind:          "OpenParen", line: {{ *}}[[@LINE-2]], column:  6, indent: 3, spelling: "(", closing_token: 16 }
  // CHECK:STDOUT:   - { index: 16, kind:         "CloseParen", line: {{ *}}[[@LINE-3]], column:  7, indent: 3, spelling: ")", opening_token: 15 }
  // CHECK:STDOUT:   - { index: 17, kind:               "Semi", line: {{ *}}[[@LINE-4]], column:  8, indent: 3, spelling: ";" }
  bar();
  // CHECK:STDOUT:   - { index: 18, kind:         "Identifier", line: {{ *}}[[@LINE-1]], column:  3, indent: 3, spelling: "bar", identifier: 4, has_leading_space: true }
  // CHECK:STDOUT:   - { index: 19, kind:          "OpenParen", line: {{ *}}[[@LINE-2]], column:  6, indent: 3, spelling: "(", closing_token: 20 }
  // CHECK:STDOUT:   - { index: 20, kind:         "CloseParen", line: {{ *}}[[@LINE-3]], column:  7, indent: 3, spelling: ")", opening_token: 19 }
  // CHECK:STDOUT:   - { index: 21, kind:               "Semi", line: {{ *}}[[@LINE-4]], column:  8, indent: 3, spelling: ";" }
// CHECK:STDERR: fail_missing_brace_multi_line_header.carbon:[[@LINE+3]]:1: error: closing symbol without a corresponding opening symbol [UnmatchedClosing]
// CHECK:STDERR: fail_missing_brace_multi_line_header.carbon:[[@LINE-18]]:11: note: possibly missing `{` here [PossiblyMissingBracketHere]
// CHECK:STDERR:
}
// CHECK:STDOUT:   - { index: 22, kind:    "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "}", opening_token: 13, has_leading_space: true }

// --- fail_missing_open_paren_at_leaf.carbon
// CHECK:STDOUT: - filename: fail_missing_open_paren_at_leaf.carbon
// CHECK:STDOUT:   tokens:

// Two values can't be adjacent, so a `(` is missing between `G` and `1` even
// though nothing else on the line is out of place.

fn F() {
// CHECK:STDOUT:   - { index:  1, kind:              "Fn", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:  4, indent: 1, spelling: "F", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:       "OpenParen", line: {{ *}}[[@LINE-3]], column:  5, indent: 1, spelling: "(", closing_token: 4 }
// CHECK:STDOUT:   - { index:  4, kind:      "CloseParen", line: {{ *}}[[@LINE-4]], column:  6, indent: 1, spelling: ")", opening_token: 3 }
// CHECK:STDOUT:   - { index:  5, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-5]], column:  8, indent: 1, spelling: "{", closing_token: 16, has_leading_space: true }
  // CHECK:STDERR: fail_missing_open_paren_at_leaf.carbon:[[@LINE+3]]:19: error: closing symbol without a corresponding opening symbol [UnmatchedClosing]
  // CHECK:STDERR: fail_missing_open_paren_at_leaf.carbon:[[@LINE+2]]:18: note: possibly missing `(` here [PossiblyMissingBracketHere]
  // CHECK:STDERR:
  var x: i32 = G 1);
  // CHECK:STDOUT:   - { index:  6, kind:             "Var", line: {{ *}}[[@LINE-1]], column:  3, indent: 3, spelling: "var", has_leading_space: true }
  // CHECK:STDOUT:   - { index:  7, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:  7, indent: 3, spelling: "x", identifier: 1, has_leading_space: true }
  // CHECK:STDOUT:   - { index:  8, kind:           "Colon", line: {{ *}}[[@LINE-3]], column:  8, indent: 3, spelling: ":" }
  // CHECK:STDOUT:   - { index:  9, kind:  "IntTypeLiteral", line: {{ *}}[[@LINE-4]], column: 10, indent: 3, spelling: "i32", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 10, kind:           "Equal", line: {{ *}}[[@LINE-5]], column: 14, indent: 3, spelling: "=", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 11, kind:      "Identifier", line: {{ *}}[[@LINE-6]], column: 16, indent: 3, spelling: "G", identifier: 2, has_leading_space: true }
  // CHECK:STDOUT:   - { index: 12, kind:       "OpenParen", line: {{ *}}[[@LINE-7]], column: 17, indent: 3, spelling: "(", closing_token: 14, has_leading_space: true, recovery: true }
  // CHECK:STDOUT:   - { index: 13, kind:      "IntLiteral", line: {{ *}}[[@LINE-8]], column: 18, indent: 3, spelling: "1", value: "1" }
  // CHECK:STDOUT:   - { index: 14, kind:      "CloseParen", line: {{ *}}[[@LINE-9]], column: 19, indent: 3, spelling: ")", opening_token: 12 }
  // CHECK:STDOUT:   - { index: 15, kind:            "Semi", line: {{ *}}[[@LINE-10]], column: 20, indent: 3, spelling: ";" }
}
// CHECK:STDOUT:   - { index: 16, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "}", opening_token: 5, has_leading_space: true }

// --- fail_unmatched_closer.carbon
// CHECK:STDOUT: - filename: fail_unmatched_closer.carbon
// CHECK:STDOUT:   tokens:

// Nothing suggests where a matching `(` would go, so the extra `)` is replaced
// with an error token and no repair is offered.

fn F() {
// CHECK:STDOUT:   - { index:  1, kind:              "Fn", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:  4, indent: 1, spelling: "F", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:       "OpenParen", line: {{ *}}[[@LINE-3]], column:  5, indent: 1, spelling: "(", closing_token: 4 }
// CHECK:STDOUT:   - { index:  4, kind:      "CloseParen", line: {{ *}}[[@LINE-4]], column:  6, indent: 1, spelling: ")", opening_token: 3 }
// CHECK:STDOUT:   - { index:  5, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-5]], column:  8, indent: 1, spelling: "{", closing_token: 17, has_leading_space: true }
  // CHECK:STDERR: fail_unmatched_closer.carbon:[[@LINE+2]]:20: error: closing symbol without a corresponding opening symbol [UnmatchedClosing]
  // CHECK:STDERR:
  var x: i32 = G(x));
  // CHECK:STDOUT:   - { index:  6, kind:             "Var", line: {{ *}}[[@LINE-1]], column:  3, indent: 3, spelling: "var", has_leading_space: true }
  // CHECK:STDOUT:   - { index:  7, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:  7, indent: 3, spelling: "x", identifier: 1, has_leading_space: true }
  // CHECK:STDOUT:   - { index:  8, kind:           "Colon", line: {{ *}}[[@LINE-3]], column:  8, indent: 3, spelling: ":" }
  // CHECK:STDOUT:   - { index:  9, kind:  "IntTypeLiteral", line: {{ *}}[[@LINE-4]], column: 10, indent: 3, spelling: "i32", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 10, kind:           "Equal", line: {{ *}}[[@LINE-5]], column: 14, indent: 3, spelling: "=", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 11, kind:      "Identifier", line: {{ *}}[[@LINE-6]], column: 16, indent: 3, spelling: "G", identifier: 2, has_leading_space: true }
  // CHECK:STDOUT:   - { index: 12, kind:       "OpenParen", line: {{ *}}[[@LINE-7]], column: 17, indent: 3, spelling: "(", closing_token: 14 }
  // CHECK:STDOUT:   - { index: 13, kind:      "Identifier", line: {{ *}}[[@LINE-8]], column: 18, indent: 3, spelling: "x", identifier: 1 }
  // CHECK:STDOUT:   - { index: 14, kind:      "CloseParen", line: {{ *}}[[@LINE-9]], column: 19, indent: 3, spelling: ")", opening_token: 12 }
  // CHECK:STDOUT:   - { index: 15, kind:           "Error", line: {{ *}}[[@LINE-10]], column: 20, indent: 3, spelling: ")" }
  // CHECK:STDOUT:   - { index: 16, kind:            "Semi", line: {{ *}}[[@LINE-11]], column: 21, indent: 3, spelling: ";" }
}
// CHECK:STDOUT:   - { index: 17, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "}", opening_token: 5, has_leading_space: true }

// --- fail_two_damaged_regions.carbon
// CHECK:STDOUT: - filename: fail_two_damaged_regions.carbon
// CHECK:STDOUT:   tokens:

// A region ends at each top-level declaration, so one mistake can't smear into
// the next declaration: these two are diagnosed and repaired independently.

fn F() {
// CHECK:STDOUT:   - { index:  1, kind:              "Fn", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:   4, indent: 1, spelling: "F", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:       "OpenParen", line: {{ *}}[[@LINE-3]], column:   5, indent: 1, spelling: "(", closing_token: 4 }
// CHECK:STDOUT:   - { index:  4, kind:      "CloseParen", line: {{ *}}[[@LINE-4]], column:   6, indent: 1, spelling: ")", opening_token: 3 }
// CHECK:STDOUT:   - { index:  5, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-5]], column:   8, indent: 1, spelling: "{", closing_token: 13, has_leading_space: true }
  // CHECK:STDERR: fail_two_damaged_regions.carbon:[[@LINE+3]]:4: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
  // CHECK:STDERR: fail_two_damaged_regions.carbon:[[@LINE+2]]:6: note: possibly missing `)` here [PossiblyMissingBracketHere]
  // CHECK:STDERR:
  G(1  + 2;
  // CHECK:STDOUT:   - { index:  6, kind:      "Identifier", line: {{ *}}[[@LINE-1]], column:   3, indent: 3, spelling: "G", identifier: 1, has_leading_space: true }
  // CHECK:STDOUT:   - { index:  7, kind:       "OpenParen", line: {{ *}}[[@LINE-2]], column:   4, indent: 3, spelling: "(", closing_token: 9 }
  // CHECK:STDOUT:   - { index:  8, kind:      "IntLiteral", line: {{ *}}[[@LINE-3]], column:   5, indent: 3, spelling: "1", value: "1" }
  // CHECK:STDOUT:   - { index:  9, kind:      "CloseParen", line: {{ *}}[[@LINE-4]], column:   6, indent: 3, spelling: ")", opening_token: 7, has_leading_space: true, recovery: true }
  // CHECK:STDOUT:   - { index: 10, kind:            "Plus", line: {{ *}}[[@LINE-5]], column:   8, indent: 3, spelling: "+" }
  // CHECK:STDOUT:   - { index: 11, kind:      "IntLiteral", line: {{ *}}[[@LINE-6]], column:  10, indent: 3, spelling: "2", value: "2", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 12, kind:            "Semi", line: {{ *}}[[@LINE-7]], column:  11, indent: 3, spelling: ";" }
}
// CHECK:STDOUT:   - { index: 13, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "}", opening_token: 5, has_leading_space: true }

fn H() {
// CHECK:STDOUT:   - { index: 14, kind:              "Fn", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index: 15, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:   4, indent: 1, spelling: "H", identifier: 2, has_leading_space: true }
// CHECK:STDOUT:   - { index: 16, kind:       "OpenParen", line: {{ *}}[[@LINE-3]], column:   5, indent: 1, spelling: "(", closing_token: 17 }
// CHECK:STDOUT:   - { index: 17, kind:      "CloseParen", line: {{ *}}[[@LINE-4]], column:   6, indent: 1, spelling: ")", opening_token: 16 }
// CHECK:STDOUT:   - { index: 18, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-5]], column:   8, indent: 1, spelling: "{", closing_token: 31, has_leading_space: true }
  // CHECK:STDERR: fail_two_damaged_regions.carbon:[[@LINE+3]]:17: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
  // CHECK:STDERR: fail_two_damaged_regions.carbon:[[@LINE+2]]:24: note: possibly missing `}` here [PossiblyMissingBracketHere]
  // CHECK:STDERR:
  var y: auto = {.a = 3;
  // CHECK:STDOUT:   - { index: 19, kind:             "Var", line: {{ *}}[[@LINE-1]], column:   3, indent: 3, spelling: "var", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 20, kind:      "Identifier", line: {{ *}}[[@LINE-2]], column:   7, indent: 3, spelling: "y", identifier: 3, has_leading_space: true }
  // CHECK:STDOUT:   - { index: 21, kind:           "Colon", line: {{ *}}[[@LINE-3]], column:   8, indent: 3, spelling: ":" }
  // CHECK:STDOUT:   - { index: 22, kind:            "Auto", line: {{ *}}[[@LINE-4]], column:  10, indent: 3, spelling: "auto", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 23, kind:           "Equal", line: {{ *}}[[@LINE-5]], column:  15, indent: 3, spelling: "=", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 24, kind:  "OpenCurlyBrace", line: {{ *}}[[@LINE-6]], column:  17, indent: 3, spelling: "{", closing_token: 29, has_leading_space: true }
  // CHECK:STDOUT:   - { index: 25, kind:          "Period", line: {{ *}}[[@LINE-7]], column:  18, indent: 3, spelling: "." }
  // CHECK:STDOUT:   - { index: 26, kind:      "Identifier", line: {{ *}}[[@LINE-8]], column:  19, indent: 3, spelling: "a", identifier: 4 }
  // CHECK:STDOUT:   - { index: 27, kind:           "Equal", line: {{ *}}[[@LINE-9]], column:  21, indent: 3, spelling: "=", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 28, kind:      "IntLiteral", line: {{ *}}[[@LINE-10]], column:  23, indent: 3, spelling: "3", value: "3", has_leading_space: true }
  // CHECK:STDOUT:   - { index: 29, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-11]], column:  24, indent: 3, spelling: "}", opening_token: 24, recovery: true }
  // CHECK:STDOUT:   - { index: 30, kind:            "Semi", line: {{ *}}[[@LINE-12]], column:  24, indent: 3, spelling: ";" }
}
// CHECK:STDOUT:   - { index: 31, kind: "CloseCurlyBrace", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: "}", opening_token: 18, has_leading_space: true }
