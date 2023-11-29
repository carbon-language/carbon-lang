// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// An autoupdate wants to put the CHECK in the middle of the block string.
// NOAUTOUPDATE
// CHECK:STDOUT: - filename: fail_block_string_second_line.carbon
// CHECK:STDOUT:   tokens: [

// CHECK:STDERR: fail_block_string_second_line.carbon:[[@LINE+4]]:3: ERROR: Only whitespace is permitted before the closing `'''` of a multi-line string.
// CHECK:STDERR:   error here: '''
// CHECK:STDERR:   ^
var s: String = '''
  error here: '''
// CHECK:STDOUT:     { index: 0, kind:         'FileStart', line:  1, column:  1, indent: 1, spelling: '', has_trailing_space: true },
// CHECK:STDOUT:     { index: 1, kind:               'Var', line: {{ *}}[[@LINE-3]], column:  1, indent: 1, spelling: 'var', has_trailing_space: true },
// CHECK:STDOUT:     { index: 2, kind:        'Identifier', line: {{ *}}[[@LINE-4]], column:  5, indent: 1, spelling: 's', identifier: 0 },
// CHECK:STDOUT:     { index: 3, kind:             'Colon', line: {{ *}}[[@LINE-5]], column:  6, indent: 1, spelling: ':', has_trailing_space: true },
// CHECK:STDOUT:     { index: 4, kind: 'StringTypeLiteral', line: {{ *}}[[@LINE-6]], column:  8, indent: 1, spelling: 'String', has_trailing_space: true },
// CHECK:STDOUT:     { index: 5, kind:             'Equal', line: {{ *}}[[@LINE-7]], column: 15, indent: 1, spelling: '=', has_trailing_space: true },
// CHECK:STDOUT:     { index: 6, kind:     'StringLiteral', line: {{ *}}[[@LINE-8]], column: 17, indent: 1, spelling: ''''
// CHECK:STDOUT:   error here: '''', value: `error here: `, has_trailing_space: true },

// CHECK:STDOUT:     { index: 7, kind:           'FileEnd', line: {{ *}}[[@LINE+1]], column: {{ *\d+}}, indent: 1, spelling: '' },
// CHECK:STDOUT:   ]
