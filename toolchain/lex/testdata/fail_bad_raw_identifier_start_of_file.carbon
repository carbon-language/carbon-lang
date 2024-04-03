// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: - filename: fail_bad_raw_identifier_start_of_file.carbon
// CHECK:STDOUT:   tokens: [
// CHECK:STDOUT:     { index: 0, kind:  'FileStart', line: {{ *\d+}}, column:  1, indent: 1, spelling: '', has_trailing_space: true },

// Ensure that we correctly handle a `#` as the first token in the file.
// CHECK:STDERR: fail_bad_raw_identifier_start_of_file.carbon:[[@LINE+3]]:1: ERROR: Encountered unrecognized characters while parsing.
// CHECK:STDERR: #foo
// CHECK:STDERR: ^
#foo
// CHECK:STDOUT:     { index: 1, kind:      'Error', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: '#' },
// CHECK:STDOUT:     { index: 2, kind: 'Identifier', line: {{ *}}[[@LINE-2]], column:  2, indent: 1, spelling: 'foo', identifier: 0, has_trailing_space: true },

// CHECK:STDOUT:     { index: 3, kind:    'FileEnd', line: {{ *}}[[@LINE+1]], column: {{ *\d+}}, indent: 1, spelling: '' },
// CHECK:STDOUT:   ]
