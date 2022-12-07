// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-tokens}
// CHECK:STDOUT: [

// CHECK:STDOUT: { index:  0, kind:              'Fn', line: {{ *}}[[@LINE+7]], column:  1, indent: 1, spelling: 'fn', has_trailing_space: true },
// CHECK:STDOUT: { index:  1, kind:      'Identifier', line: {{ *}}[[@LINE+6]], column:  4, indent: 1, spelling: 'run', identifier: 0 },
// CHECK:STDOUT: { index:  2, kind:       'OpenParen', line: {{ *}}[[@LINE+5]], column:  7, indent: 1, spelling: '(', closing_token: 5 },
// CHECK:STDOUT: { index:  3, kind:      'Identifier', line: {{ *}}[[@LINE+4]], column:  8, indent: 1, spelling: 'String', identifier: 1, has_trailing_space: true },
// CHECK:STDOUT: { index:  4, kind:      'Identifier', line: {{ *}}[[@LINE+3]], column: 15, indent: 1, spelling: 'program', identifier: 2 },
// CHECK:STDOUT: { index:  5, kind:      'CloseParen', line: {{ *}}[[@LINE+2]], column: 22, indent: 1, spelling: ')', opening_token: 2, has_trailing_space: true },
// CHECK:STDOUT: { index:  6, kind:  'OpenCurlyBrace', line: {{ *}}[[@LINE+1]], column: 24, indent: 1, spelling: '{', closing_token: 10, has_trailing_space: true },
fn run(String program) {
  // CHECK:STDOUT: { index:  7, kind:          'Return', line: {{ *}}[[@LINE+3]], column:  3, indent: 3, spelling: 'return', has_trailing_space: true },
  // CHECK:STDOUT: { index:  8, kind:      'Identifier', line: {{ *}}[[@LINE+2]], column: 10, indent: 3, spelling: 'True', identifier: 3 },
  // CHECK:STDOUT: { index:  9, kind:            'Semi', line: {{ *}}[[@LINE+1]], column: 14, indent: 3, spelling: ';', has_trailing_space: true },
  return True;
// CHECK:STDOUT: { index: 10, kind: 'CloseCurlyBrace', line: {{ *}}[[@LINE+3]], column:  1, indent: 1, spelling: '}', opening_token: 6, has_trailing_space: true },
// CHECK:STDOUT: { index: 11, kind:       'EndOfFile', line: {{ *}}[[@LINE+2]], column:  2, indent: 1, spelling: '' },
// CHECK:STDOUT: ]
}
