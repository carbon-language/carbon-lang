// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TODO: For this example, we should ideally report a missing `{` on the `if`
// line, for example based on the indentation of the surrounding lines.

fn F() {
  if (thing1)
    thing2;
  }
}

// The check lines are inserted at the end so that they don't disrupt the
// indentation of lines in the text.
// AUTOUPDATE
// CHECK:STDERR: fail_mismatched_brackets_2.carbon:[[@LINE-5]]:1: ERROR: Closing symbol without a corresponding opening symbol.
// CHECK:STDERR: }
// CHECK:STDERR: ^
// CHECK:STDOUT: - filename: fail_mismatched_brackets_2.carbon
// CHECK:STDOUT:   tokens: [
// CHECK:STDOUT:     { index:  0, kind:       'FileStart', line: {{ *\d+}}, column:  1, indent: 1, spelling: '', has_trailing_space: true },
// CHECK:STDOUT:     { index:  1, kind:              'Fn', line: {{ *}}[[@LINE-15]], column:  1, indent: 1, spelling: 'fn', has_trailing_space: true },
// CHECK:STDOUT:     { index:  2, kind:      'Identifier', line: {{ *}}[[@LINE-16]], column:  4, indent: 1, spelling: 'F', identifier: 0 },
// CHECK:STDOUT:     { index:  3, kind:       'OpenParen', line: {{ *}}[[@LINE-17]], column:  5, indent: 1, spelling: '(', closing_token: 4 },
// CHECK:STDOUT:     { index:  4, kind:      'CloseParen', line: {{ *}}[[@LINE-18]], column:  6, indent: 1, spelling: ')', opening_token: 3, has_trailing_space: true },
// CHECK:STDOUT:     { index:  5, kind:  'OpenCurlyBrace', line: {{ *}}[[@LINE-19]], column:  8, indent: 1, spelling: '{', closing_token: 12, has_trailing_space: true },
// CHECK:STDOUT:     { index:  6, kind:              'If', line: {{ *}}[[@LINE-19]], column:  3, indent: 3, spelling: 'if', has_trailing_space: true },
// CHECK:STDOUT:     { index:  7, kind:       'OpenParen', line: {{ *}}[[@LINE-20]], column:  6, indent: 3, spelling: '(', closing_token: 9 },
// CHECK:STDOUT:     { index:  8, kind:      'Identifier', line: {{ *}}[[@LINE-21]], column:  7, indent: 3, spelling: 'thing1', identifier: 1 },
// CHECK:STDOUT:     { index:  9, kind:      'CloseParen', line: {{ *}}[[@LINE-22]], column: 13, indent: 3, spelling: ')', opening_token: 7, has_trailing_space: true },
// CHECK:STDOUT:     { index: 10, kind:      'Identifier', line: {{ *}}[[@LINE-22]], column:  5, indent: 5, spelling: 'thing2', identifier: 2 },
// CHECK:STDOUT:     { index: 11, kind:            'Semi', line: {{ *}}[[@LINE-23]], column: 11, indent: 5, spelling: ';', has_trailing_space: true },
// CHECK:STDOUT:     { index: 12, kind: 'CloseCurlyBrace', line: {{ *}}[[@LINE-23]], column:  3, indent: 3, spelling: '}', opening_token: 5, has_trailing_space: true },
// CHECK:STDOUT:     { index: 13, kind:           'Error', line: {{ *}}[[@LINE-23]], column:  1, indent: 1, spelling: '}', has_trailing_space: true },

// CHECK:STDOUT:     { index: 14, kind:         'FileEnd', line: {{ *}}[[@LINE+1]], column: {{ *\d+}}, indent: 1, spelling: '' },
// CHECK:STDOUT:   ]
