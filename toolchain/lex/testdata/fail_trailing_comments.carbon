// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Trailing comments should produce errors, but we should continue to parse (and
// diagnose) code errors beyond them.

var a: i32 = 1;   // trailing comment
var b: 32 = 13;   // more trailing comment
var c: i32 = 0.4; // still more trailing comment

// We keep the auto-update marker below the above so check lines don't disrupt
// the trailing comment lines with similar prefixes.
// AUTOUPDATE
// CHECK:STDERR: fail_trailing_comments.carbon:[[@LINE-7]]:19: ERROR: Trailing comments are not permitted.
// CHECK:STDERR: var a: i32 = 1;   // trailing comment
// CHECK:STDERR:                   ^
// CHECK:STDERR: fail_trailing_comments.carbon:[[@LINE-9]]:19: ERROR: Trailing comments are not permitted.
// CHECK:STDERR: var b: 32 = 13;   // more trailing comment
// CHECK:STDERR:                   ^
// CHECK:STDERR: fail_trailing_comments.carbon:[[@LINE-11]]:19: ERROR: Trailing comments are not permitted.
// CHECK:STDERR: var c: i32 = 0.4; // still more trailing comment
// CHECK:STDERR:                   ^
// CHECK:STDOUT: - filename: fail_trailing_comments.carbon
// CHECK:STDOUT:   tokens: [
// CHECK:STDOUT:     { index:  0, kind:      'FileStart', line: {{ *\d+}}, column:  1, indent: 1, spelling: '', has_trailing_space: true },
// CHECK:STDOUT:     { index:  1, kind:            'Var', line: {{ *}}[[@LINE-19]], column:  1, indent: 1, spelling: 'var', has_trailing_space: true },
// CHECK:STDOUT:     { index:  2, kind:     'Identifier', line: {{ *}}[[@LINE-20]], column:  5, indent: 1, spelling: 'a', identifier: 0 },
// CHECK:STDOUT:     { index:  3, kind:          'Colon', line: {{ *}}[[@LINE-21]], column:  6, indent: 1, spelling: ':', has_trailing_space: true },
// CHECK:STDOUT:     { index:  4, kind: 'IntTypeLiteral', line: {{ *}}[[@LINE-22]], column:  8, indent: 1, spelling: 'i32', has_trailing_space: true },
// CHECK:STDOUT:     { index:  5, kind:          'Equal', line: {{ *}}[[@LINE-23]], column: 12, indent: 1, spelling: '=', has_trailing_space: true },
// CHECK:STDOUT:     { index:  6, kind:     'IntLiteral', line: {{ *}}[[@LINE-24]], column: 14, indent: 1, spelling: '1', value: `1` },
// CHECK:STDOUT:     { index:  7, kind:           'Semi', line: {{ *}}[[@LINE-25]], column: 15, indent: 1, spelling: ';', has_trailing_space: true },
// CHECK:STDOUT:     { index:  8, kind:            'Var', line: {{ *}}[[@LINE-25]], column:  1, indent: 1, spelling: 'var', has_trailing_space: true },
// CHECK:STDOUT:     { index:  9, kind:     'Identifier', line: {{ *}}[[@LINE-26]], column:  5, indent: 1, spelling: 'b', identifier: 1 },
// CHECK:STDOUT:     { index: 10, kind:          'Colon', line: {{ *}}[[@LINE-27]], column:  6, indent: 1, spelling: ':', has_trailing_space: true },
// CHECK:STDOUT:     { index: 11, kind:     'IntLiteral', line: {{ *}}[[@LINE-28]], column:  8, indent: 1, spelling: '32', value: `32`, has_trailing_space: true },
// CHECK:STDOUT:     { index: 12, kind:          'Equal', line: {{ *}}[[@LINE-29]], column: 11, indent: 1, spelling: '=', has_trailing_space: true },
// CHECK:STDOUT:     { index: 13, kind:     'IntLiteral', line: {{ *}}[[@LINE-30]], column: 13, indent: 1, spelling: '13', value: `13` },
// CHECK:STDOUT:     { index: 14, kind:           'Semi', line: {{ *}}[[@LINE-31]], column: 15, indent: 1, spelling: ';', has_trailing_space: true },
// CHECK:STDOUT:     { index: 15, kind:            'Var', line: {{ *}}[[@LINE-31]], column:  1, indent: 1, spelling: 'var', has_trailing_space: true },
// CHECK:STDOUT:     { index: 16, kind:     'Identifier', line: {{ *}}[[@LINE-32]], column:  5, indent: 1, spelling: 'c', identifier: 2 },
// CHECK:STDOUT:     { index: 17, kind:          'Colon', line: {{ *}}[[@LINE-33]], column:  6, indent: 1, spelling: ':', has_trailing_space: true },
// CHECK:STDOUT:     { index: 18, kind: 'IntTypeLiteral', line: {{ *}}[[@LINE-34]], column:  8, indent: 1, spelling: 'i32', has_trailing_space: true },
// CHECK:STDOUT:     { index: 19, kind:          'Equal', line: {{ *}}[[@LINE-35]], column: 12, indent: 1, spelling: '=', has_trailing_space: true },
// CHECK:STDOUT:     { index: 20, kind:    'RealLiteral', line: {{ *}}[[@LINE-36]], column: 14, indent: 1, spelling: '0.4', value: `4*10^-1` },
// CHECK:STDOUT:     { index: 21, kind:           'Semi', line: {{ *}}[[@LINE-37]], column: 17, indent: 1, spelling: ';', has_trailing_space: true },

// CHECK:STDOUT:     { index: 22, kind:        'FileEnd', line: {{ *}}[[@LINE+1]], column: {{ *\d+}}, indent: 1, spelling: '' },
// CHECK:STDOUT:   ]
