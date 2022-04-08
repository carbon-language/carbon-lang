// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{carbon} dump-tokens %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s

fn run(String program) {
  return True;
}

// CHECK: token: { index:  0, kind:              'Fn', line: 8, column:   1, indent: 1, spelling: 'fn', has_trailing_space: true }
// CHECK: token: { index:  1, kind:      'Identifier', line: 8, column:   4, indent: 1, spelling: 'run', identifier: 0 }
// CHECK: token: { index:  2, kind:       'OpenParen', line: 8, column:   7, indent: 1, spelling: '(', closing_token: 5 }
// CHECK: token: { index:  3, kind:      'Identifier', line: 8, column:   8, indent: 1, spelling: 'String', identifier: 1, has_trailing_space: true }
// CHECK: token: { index:  4, kind:      'Identifier', line: 8, column:  15, indent: 1, spelling: 'program', identifier: 2 }
// CHECK: token: { index:  5, kind:      'CloseParen', line: 8, column:  22, indent: 1, spelling: ')', opening_token: 2, has_trailing_space: true }
// CHECK: token: { index:  6, kind:  'OpenCurlyBrace', line: 8, column:  24, indent: 1, spelling: '{', closing_token: 10, has_trailing_space: true }
// CHECK: token: { index:  7, kind:          'Return', line: 9, column:   3, indent: 3, spelling: 'return', has_trailing_space: true }
// CHECK: token: { index:  8, kind:      'Identifier', line: 9, column:  10, indent: 3, spelling: 'True', identifier: 3 }
// CHECK: token: { index:  9, kind:            'Semi', line: 9, column:  14, indent: 3, spelling: ';', has_trailing_space: true }
// CHECK: token: { index: 10, kind: 'CloseCurlyBrace', line: 10, column:   1, indent: 1, spelling: '}', opening_token: 6, has_trailing_space: true }
// CHECK: token: { index: 11, kind:       'EndOfFile', line: 23, column: 104, indent: 1, spelling: '' }
