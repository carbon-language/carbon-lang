// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// FIXME: We need to figure out how to integrate this style of integration
// testing with Bazel.
//
// RUN: %{carbon} dump-tokens %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false \
// RUN:   --check-prefix=TOKENS %s

fn run(String program) {
  return True;
}

// TOKENS: token: { index:  0, kind:              'Fn', line: 12, column:   1, indent: 1, spelling: 'fn', has_trailing_space: true }
// TOKENS: token: { index:  1, kind:      'Identifier', line: 12, column:   4, indent: 1, spelling: 'run', identifier: 0 }
// TOKENS: token: { index:  2, kind:       'OpenParen', line: 12, column:   7, indent: 1, spelling: '(', closing_token: 5 }
// TOKENS: token: { index:  3, kind:      'Identifier', line: 12, column:   8, indent: 1, spelling: 'String', identifier: 1, has_trailing_space: true }
// TOKENS: token: { index:  4, kind:      'Identifier', line: 12, column:  15, indent: 1, spelling: 'program', identifier: 2 }
// TOKENS: token: { index:  5, kind:      'CloseParen', line: 12, column:  22, indent: 1, spelling: ')', opening_token: 2, has_trailing_space: true }
// TOKENS: token: { index:  6, kind:  'OpenCurlyBrace', line: 12, column:  24, indent: 1, spelling: '{', closing_token: 10, has_trailing_space: true }
// TOKENS: token: { index:  7, kind:          'Return', line: 13, column:   3, indent: 3, spelling: 'return', has_trailing_space: true }
// TOKENS: token: { index:  8, kind:      'Identifier', line: 13, column:  10, indent: 3, spelling: 'True', identifier: 3 }
// TOKENS: token: { index:  9, kind:            'Semi', line: 13, column:  14, indent: 3, spelling: ';', has_trailing_space: true }
// TOKENS: token: { index: 10, kind: 'CloseCurlyBrace', line: 14, column:   1, indent: 1, spelling: '}', opening_token: 6, has_trailing_space: true }
// TOKENS: token: { index: 11, kind:       'EndOfFile', line: 27, column: 105, indent: 1, spelling: '' }
