// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AUTOUPDATE
// CHECK:STDOUT: - filename: fail_mismatched_brackets.carbon
// CHECK:STDOUT:   tokens: [
// CHECK:STDOUT:     { index: 0, kind:  'FileStart', line: {{ *\d+}}, column:  1, indent: 1, spelling: '', has_trailing_space: true },

// CHECK:STDERR: fail_mismatched_brackets.carbon:[[@LINE+3]]:1: ERROR: Closing symbol without a corresponding opening symbol.
// CHECK:STDERR: }
// CHECK:STDERR: ^
}
// CHECK:STDOUT:     { index: 1, kind:      'Error', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: '}', has_trailing_space: true },

// CHECK:STDERR: fail_mismatched_brackets.carbon:[[@LINE+3]]:3: ERROR: Closing symbol without a corresponding opening symbol.
// CHECK:STDERR: ( } )
// CHECK:STDERR:   ^
( } )
// CHECK:STDOUT:     { index: 2, kind:  'OpenParen', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: '(', closing_token: 4, has_trailing_space: true },
// CHECK:STDOUT:     { index: 3, kind:      'Error', line: {{ *}}[[@LINE-2]], column:  3, indent: 1, spelling: '}', has_trailing_space: true },
// CHECK:STDOUT:     { index: 4, kind: 'CloseParen', line: {{ *}}[[@LINE-3]], column:  5, indent: 1, spelling: ')', opening_token: 2, has_trailing_space: true },

// CHECK:STDERR: fail_mismatched_brackets.carbon:[[@LINE+3]]:1: ERROR: Opening symbol without a corresponding closing symbol.
// CHECK:STDERR: [
// CHECK:STDERR: ^
[
// CHECK:STDOUT:     { index: 5, kind:      'Error', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: '[', has_trailing_space: true },

// CHECK:STDOUT:     { index: 6, kind:    'FileEnd', line: {{ *}}[[@LINE+1]], column: {{ *\d+}}, indent: 1, spelling: '' },
// CHECK:STDOUT:   ]
