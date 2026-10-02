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

// A line dedented past the header of the scope it's in can't belong to that
// scope, so the `}` goes at the dedent. Indentation is the whole cue here: the
// brackets alone balance if the last `}` closes the `if` instead.

fn F() {
  if (x) {
    foo();
  bar();
}

// --- fail_missing_brace_before_else.carbon

// A first-on-line `else` should have had a `}` before it. Only one is inserted:
// the `else` block's own `}` still lines up, because the block's header
// indentation is taken from the `if` it continues rather than from the `else`'s
// own column, which is wherever the author left it once the `}` went missing.

fn F() {
  if (C) {
    G();
  else {
    H();
  }
}

// --- fail_missing_brace_in_class.carbon

// As above for a class body, whose `}` is missing entirely rather than
// misattributed.

class C {
  var a: i32;
fn F() {
  G();
}

// --- fail_unclosed_at_eof.carbon

// Whatever is still open at the end of the file is closed there.

fn F() {
  var x: i32 = 1;


// --- AUTOUPDATE-SPLIT

// CHECK:STDERR: fail_dedent_inside_scope.carbon:7:10: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
// CHECK:STDERR: fail_dedent_inside_scope.carbon:9:3: note: possibly missing `}` here [PossiblyMissingBracketHere]
// CHECK:STDERR:
// CHECK:STDERR: fail_missing_brace_before_else.carbon:8:10: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
// CHECK:STDERR: fail_missing_brace_before_else.carbon:10:3: note: possibly missing `}` here [PossiblyMissingBracketHere]
// CHECK:STDERR:
// CHECK:STDERR: fail_missing_brace_in_class.carbon:5:9: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
// CHECK:STDERR: fail_missing_brace_in_class.carbon:7:1: note: possibly missing `}` here [PossiblyMissingBracketHere]
// CHECK:STDERR:
// CHECK:STDERR: fail_unclosed_at_eof.carbon:4:8: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
// CHECK:STDERR: fail_unclosed_at_eof.carbon:6:1: note: possibly missing `}` here [PossiblyMissingBracketHere]
// CHECK:STDERR:
// CHECK:STDOUT: - filename: fail_dedent_inside_scope.carbon
// CHECK:STDOUT:   tokens:
// CHECK:STDOUT:   - { index:  1, kind:              "Fn", line: {{ *}}6, column:  1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:      "Identifier", line: {{ *}}6, column:  4, indent: 1, spelling: "F", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:       "OpenParen", line: {{ *}}6, column:  5, indent: 1, spelling: "(", closing_token: 4 }
// CHECK:STDOUT:   - { index:  4, kind:      "CloseParen", line: {{ *}}6, column:  6, indent: 1, spelling: ")", opening_token: 3 }
// CHECK:STDOUT:   - { index:  5, kind:  "OpenCurlyBrace", line: {{ *}}6, column:  8, indent: 1, spelling: "{", closing_token: 20, has_leading_space: true }
// CHECK:STDOUT:   - { index:  6, kind:              "If", line: {{ *}}7, column:  3, indent: 3, spelling: "if", has_leading_space: true }
// CHECK:STDOUT:   - { index:  7, kind:       "OpenParen", line: {{ *}}7, column:  6, indent: 3, spelling: "(", closing_token: 9, has_leading_space: true }
// CHECK:STDOUT:   - { index:  8, kind:      "Identifier", line: {{ *}}7, column:  7, indent: 3, spelling: "x", identifier: 1 }
// CHECK:STDOUT:   - { index:  9, kind:      "CloseParen", line: {{ *}}7, column:  8, indent: 3, spelling: ")", opening_token: 7 }
// CHECK:STDOUT:   - { index: 10, kind:  "OpenCurlyBrace", line: {{ *}}7, column: 10, indent: 3, spelling: "{", closing_token: 15, has_leading_space: true }
// CHECK:STDOUT:   - { index: 11, kind:      "Identifier", line: {{ *}}8, column:  5, indent: 5, spelling: "foo", identifier: 2, has_leading_space: true }
// CHECK:STDOUT:   - { index: 12, kind:       "OpenParen", line: {{ *}}8, column:  8, indent: 5, spelling: "(", closing_token: 13 }
// CHECK:STDOUT:   - { index: 13, kind:      "CloseParen", line: {{ *}}8, column:  9, indent: 5, spelling: ")", opening_token: 12 }
// CHECK:STDOUT:   - { index: 14, kind:            "Semi", line: {{ *}}8, column: 10, indent: 5, spelling: ";" }
// CHECK:STDOUT:   - { index: 15, kind: "CloseCurlyBrace", line: {{ *}}8, column: 11, indent: 5, spelling: "}", opening_token: 10, has_leading_space: true, recovery: true }
// CHECK:STDOUT:   - { index: 16, kind:      "Identifier", line: {{ *}}9, column:  3, indent: 3, spelling: "bar", identifier: 3 }
// CHECK:STDOUT:   - { index: 17, kind:       "OpenParen", line: {{ *}}9, column:  6, indent: 3, spelling: "(", closing_token: 18 }
// CHECK:STDOUT:   - { index: 18, kind:      "CloseParen", line: {{ *}}9, column:  7, indent: 3, spelling: ")", opening_token: 17 }
// CHECK:STDOUT:   - { index: 19, kind:            "Semi", line: {{ *}}9, column:  8, indent: 3, spelling: ";" }
// CHECK:STDOUT:   - { index: 20, kind: "CloseCurlyBrace", line: {{ *}}10, column:  1, indent: 1, spelling: "}", opening_token: 5, has_leading_space: true }
// CHECK:STDOUT: - filename: fail_missing_brace_before_else.carbon
// CHECK:STDOUT:   tokens:
// CHECK:STDOUT:   - { index:  1, kind:              "Fn", line: {{ *}}7, column:  1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:      "Identifier", line: {{ *}}7, column:  4, indent: 1, spelling: "F", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:       "OpenParen", line: {{ *}}7, column:  5, indent: 1, spelling: "(", closing_token: 4 }
// CHECK:STDOUT:   - { index:  4, kind:      "CloseParen", line: {{ *}}7, column:  6, indent: 1, spelling: ")", opening_token: 3 }
// CHECK:STDOUT:   - { index:  5, kind:  "OpenCurlyBrace", line: {{ *}}7, column:  8, indent: 1, spelling: "{", closing_token: 23, has_leading_space: true }
// CHECK:STDOUT:   - { index:  6, kind:              "If", line: {{ *}}8, column:  3, indent: 3, spelling: "if", has_leading_space: true }
// CHECK:STDOUT:   - { index:  7, kind:       "OpenParen", line: {{ *}}8, column:  6, indent: 3, spelling: "(", closing_token: 9, has_leading_space: true }
// CHECK:STDOUT:   - { index:  8, kind:      "Identifier", line: {{ *}}8, column:  7, indent: 3, spelling: "C", identifier: 1 }
// CHECK:STDOUT:   - { index:  9, kind:      "CloseParen", line: {{ *}}8, column:  8, indent: 3, spelling: ")", opening_token: 7 }
// CHECK:STDOUT:   - { index: 10, kind:  "OpenCurlyBrace", line: {{ *}}8, column: 10, indent: 3, spelling: "{", closing_token: 15, has_leading_space: true }
// CHECK:STDOUT:   - { index: 11, kind:      "Identifier", line: {{ *}}9, column:  5, indent: 5, spelling: "G", identifier: 2, has_leading_space: true }
// CHECK:STDOUT:   - { index: 12, kind:       "OpenParen", line: {{ *}}9, column:  6, indent: 5, spelling: "(", closing_token: 13 }
// CHECK:STDOUT:   - { index: 13, kind:      "CloseParen", line: {{ *}}9, column:  7, indent: 5, spelling: ")", opening_token: 12 }
// CHECK:STDOUT:   - { index: 14, kind:            "Semi", line: {{ *}}9, column:  8, indent: 5, spelling: ";" }
// CHECK:STDOUT:   - { index: 15, kind: "CloseCurlyBrace", line: {{ *}}9, column:  9, indent: 5, spelling: "}", opening_token: 10, has_leading_space: true, recovery: true }
// CHECK:STDOUT:   - { index: 16, kind:            "Else", line: {{ *}}10, column:  3, indent: 3, spelling: "else" }
// CHECK:STDOUT:   - { index: 17, kind:  "OpenCurlyBrace", line: {{ *}}10, column:  8, indent: 3, spelling: "{", closing_token: 22, has_leading_space: true }
// CHECK:STDOUT:   - { index: 18, kind:      "Identifier", line: {{ *}}11, column:  5, indent: 5, spelling: "H", identifier: 3, has_leading_space: true }
// CHECK:STDOUT:   - { index: 19, kind:       "OpenParen", line: {{ *}}11, column:  6, indent: 5, spelling: "(", closing_token: 20 }
// CHECK:STDOUT:   - { index: 20, kind:      "CloseParen", line: {{ *}}11, column:  7, indent: 5, spelling: ")", opening_token: 19 }
// CHECK:STDOUT:   - { index: 21, kind:            "Semi", line: {{ *}}11, column:  8, indent: 5, spelling: ";" }
// CHECK:STDOUT:   - { index: 22, kind: "CloseCurlyBrace", line: {{ *}}12, column:  3, indent: 3, spelling: "}", opening_token: 17, has_leading_space: true }
// CHECK:STDOUT:   - { index: 23, kind: "CloseCurlyBrace", line: {{ *}}13, column:  1, indent: 1, spelling: "}", opening_token: 5, has_leading_space: true }
// CHECK:STDOUT: - filename: fail_missing_brace_in_class.carbon
// CHECK:STDOUT:   tokens:
// CHECK:STDOUT:   - { index:  1, kind:           "Class", line: {{ *}}5, column:  1, indent: 1, spelling: "class", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:      "Identifier", line: {{ *}}5, column:  7, indent: 1, spelling: "C", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:  "OpenCurlyBrace", line: {{ *}}5, column:  9, indent: 1, spelling: "{", closing_token: 9, has_leading_space: true }
// CHECK:STDOUT:   - { index:  4, kind:             "Var", line: {{ *}}6, column:  3, indent: 3, spelling: "var", has_leading_space: true }
// CHECK:STDOUT:   - { index:  5, kind:      "Identifier", line: {{ *}}6, column:  7, indent: 3, spelling: "a", identifier: 1, has_leading_space: true }
// CHECK:STDOUT:   - { index:  6, kind:           "Colon", line: {{ *}}6, column:  8, indent: 3, spelling: ":" }
// CHECK:STDOUT:   - { index:  7, kind:  "IntTypeLiteral", line: {{ *}}6, column: 10, indent: 3, spelling: "i32", has_leading_space: true }
// CHECK:STDOUT:   - { index:  8, kind:            "Semi", line: {{ *}}6, column: 13, indent: 3, spelling: ";" }
// CHECK:STDOUT:   - { index:  9, kind: "CloseCurlyBrace", line: {{ *}}6, column: 14, indent: 3, spelling: "}", opening_token: 3, has_leading_space: true, recovery: true }
// CHECK:STDOUT:   - { index: 10, kind:              "Fn", line: {{ *}}7, column:  1, indent: 1, spelling: "fn" }
// CHECK:STDOUT:   - { index: 11, kind:      "Identifier", line: {{ *}}7, column:  4, indent: 1, spelling: "F", identifier: 2, has_leading_space: true }
// CHECK:STDOUT:   - { index: 12, kind:       "OpenParen", line: {{ *}}7, column:  5, indent: 1, spelling: "(", closing_token: 13 }
// CHECK:STDOUT:   - { index: 13, kind:      "CloseParen", line: {{ *}}7, column:  6, indent: 1, spelling: ")", opening_token: 12 }
// CHECK:STDOUT:   - { index: 14, kind:  "OpenCurlyBrace", line: {{ *}}7, column:  8, indent: 1, spelling: "{", closing_token: 19, has_leading_space: true }
// CHECK:STDOUT:   - { index: 15, kind:      "Identifier", line: {{ *}}8, column:  3, indent: 3, spelling: "G", identifier: 3, has_leading_space: true }
// CHECK:STDOUT:   - { index: 16, kind:       "OpenParen", line: {{ *}}8, column:  4, indent: 3, spelling: "(", closing_token: 17 }
// CHECK:STDOUT:   - { index: 17, kind:      "CloseParen", line: {{ *}}8, column:  5, indent: 3, spelling: ")", opening_token: 16 }
// CHECK:STDOUT:   - { index: 18, kind:            "Semi", line: {{ *}}8, column:  6, indent: 3, spelling: ";" }
// CHECK:STDOUT:   - { index: 19, kind: "CloseCurlyBrace", line: {{ *}}9, column:  1, indent: 1, spelling: "}", opening_token: 14, has_leading_space: true }
// CHECK:STDOUT: - filename: fail_unclosed_at_eof.carbon
// CHECK:STDOUT:   tokens:
// CHECK:STDOUT:   - { index:  1, kind:              "Fn", line: {{ *}}4, column:  1, indent: 1, spelling: "fn", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:      "Identifier", line: {{ *}}4, column:  4, indent: 1, spelling: "F", identifier: 0, has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:       "OpenParen", line: {{ *}}4, column:  5, indent: 1, spelling: "(", closing_token: 4 }
// CHECK:STDOUT:   - { index:  4, kind:      "CloseParen", line: {{ *}}4, column:  6, indent: 1, spelling: ")", opening_token: 3 }
// CHECK:STDOUT:   - { index:  5, kind:  "OpenCurlyBrace", line: {{ *}}4, column:  8, indent: 1, spelling: "{", closing_token: 13, has_leading_space: true }
// CHECK:STDOUT:   - { index:  6, kind:             "Var", line: {{ *}}5, column:  3, indent: 3, spelling: "var", has_leading_space: true }
// CHECK:STDOUT:   - { index:  7, kind:      "Identifier", line: {{ *}}5, column:  7, indent: 3, spelling: "x", identifier: 1, has_leading_space: true }
// CHECK:STDOUT:   - { index:  8, kind:           "Colon", line: {{ *}}5, column:  8, indent: 3, spelling: ":" }
// CHECK:STDOUT:   - { index:  9, kind:  "IntTypeLiteral", line: {{ *}}5, column: 10, indent: 3, spelling: "i32", has_leading_space: true }
// CHECK:STDOUT:   - { index: 10, kind:           "Equal", line: {{ *}}5, column: 14, indent: 3, spelling: "=", has_leading_space: true }
// CHECK:STDOUT:   - { index: 11, kind:      "IntLiteral", line: {{ *}}5, column: 16, indent: 3, spelling: "1", value: "1", has_leading_space: true }
// CHECK:STDOUT:   - { index: 12, kind:            "Semi", line: {{ *}}5, column: 17, indent: 3, spelling: ";" }
// CHECK:STDOUT:   - { index: 13, kind: "CloseCurlyBrace", line: {{ *}}5, column: 18, indent: 3, spelling: "}", opening_token: 5, has_leading_space: true, recovery: true }
