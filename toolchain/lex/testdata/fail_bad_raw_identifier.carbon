// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: - filename: fail_bad_raw_identifier.carbon
// CHECK:STDOUT:   tokens: [
// CHECK:STDOUT:     { index: 0, kind:  'FileStart', line: {{ *\d+}}, column:  1, indent: 1, spelling: '', has_trailing_space: true },

// Missing the character after `#`.
// CHECK:STDERR: fail_bad_raw_identifier.carbon:[[@LINE+3]]:2: ERROR: Encountered unrecognized characters while parsing.
// CHECK:STDERR: r#
// CHECK:STDERR:  ^
r#
// CHECK:STDOUT:     { index: 1, kind: 'Identifier', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'r', identifier: 0 },
// CHECK:STDOUT:     { index: 2, kind:      'Error', line: {{ *}}[[@LINE-2]], column:  2, indent: 1, spelling: '#', has_trailing_space: true },

// Not a valid identifier.
// CHECK:STDERR: fail_bad_raw_identifier.carbon:[[@LINE+3]]:2: ERROR: Encountered unrecognized characters while parsing.
// CHECK:STDERR: r#3
// CHECK:STDERR:  ^
r#3
// CHECK:STDOUT:     { index: 3, kind: 'Identifier', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'r', identifier: 0 },
// CHECK:STDOUT:     { index: 4, kind:      'Error', line: {{ *}}[[@LINE-2]], column:  2, indent: 1, spelling: '#' },
// CHECK:STDOUT:     { index: 5, kind: 'IntLiteral', line: {{ *}}[[@LINE-3]], column:  3, indent: 1, spelling: '3', value: `3`, has_trailing_space: true },

// Non ascii start to identifier.
// CHECK:STDERR: fail_bad_raw_identifier.carbon:[[@LINE+3]]:2: ERROR: Encountered unrecognized characters while parsing.
// CHECK:STDERR: r#á
// CHECK:STDERR:  ^
r#á
// CHECK:STDOUT:     { index: 6, kind: 'Identifier', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'r', identifier: 0 },
// CHECK:STDOUT:     { index: 7, kind:      'Error', line: {{ *}}[[@LINE-2]], column:  2, indent: 1, spelling: '#á', has_trailing_space: true },

// CHECK:STDOUT:     { index: 8, kind:    'FileEnd', line: {{ *}}[[@LINE+1]], column: {{ *\d+}}, indent: 1, spelling: '' },
// CHECK:STDOUT:   ]
