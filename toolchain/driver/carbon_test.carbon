// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// FIXME: We need to figure out how to integrate this style of integration
// testing with Bazel.
//
// RUN: driver/carbon dump-tokens %s | FileCheck %s --check-prefix=TOKENS

fn run(String program) {
  return True;
}

// TOKENS: token: { index:  0, kind:       'FnKeyword', line: 4, column:  1, indent: 1, spelling: 'fn' }
// TOKENS: token: { index:  1, kind:      'Identifier', line: 4, column:  4, indent: 1, spelling: 'run', identifier: 0 }
// TOKENS: token: { index:  2, kind:       'OpenParen', line: 4, column:  7, indent: 1, spelling: '(', closing_token: 5 }
// TOKENS: token: { index:  3, kind:      'Identifier', line: 4, column:  8, indent: 1, spelling: 'String', identifier: 1 }
// TOKENS: token: { index:  4, kind:      'Identifier', line: 4, column: 15, indent: 1, spelling: 'program', identifier: 2 }
// TOKENS: token: { index:  5, kind:      'CloseParen', line: 4, column: 22, indent: 1, spelling: ')', opening_token: 2 }
// TOKENS: token: { index:  6, kind:  'OpenCurlyBrace', line: 4, column: 24, indent: 1, spelling: '{', closing_token: 10 }
// TOKENS: token: { index:  7, kind:   'ReturnKeyword', line: 5, column:  3, indent: 3, spelling: 'return' }
// TOKENS: token: { index:  8, kind:      'Identifier', line: 5, column: 10, indent: 3, spelling: 'True', identifier: 3 }
// TOKENS: token: { index:  9, kind:            'Semi', line: 5, column: 14, indent: 3, spelling: ';' }
// TOKENS: token: { index: 10, kind: 'CloseCurlyBrace', line: 6, column:  1, indent: 1, spelling: '}', opening_token: 6 }
