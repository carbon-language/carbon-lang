// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- a.carbon
// CHECK:STDOUT: - filename: a.carbon
// CHECK:STDOUT:   tokens: [
// CHECK:STDOUT:     { index: 0, kind:  'FileStart', line: {{ *\d+}}, column:  1, indent: 1, spelling: '', has_trailing_space: true },
a;
// CHECK:STDOUT:     { index: 1, kind: 'Identifier', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'a', identifier: 0 },
// CHECK:STDOUT:     { index: 2, kind:       'Semi', line: {{ *}}[[@LINE-2]], column:  2, indent: 1, spelling: ';', has_trailing_space: true },

// CHECK:STDOUT:     { index: 3, kind:    'FileEnd', line: {{ *}}[[@LINE+1]], column: {{ *\d+}}, indent: 1, spelling: '' },
// CHECK:STDOUT:   ]
// --- b.carbon
// CHECK:STDOUT: - filename: b.carbon
// CHECK:STDOUT:   tokens: [
// CHECK:STDOUT:     { index: 0, kind:  'FileStart', line: {{ *\d+}}, column:  1, indent: 1, spelling: '', has_trailing_space: true },
b;
// CHECK:STDOUT:     { index: 1, kind: 'Identifier', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'b', identifier: 0 },
// CHECK:STDOUT:     { index: 2, kind:       'Semi', line: {{ *}}[[@LINE-2]], column:  2, indent: 1, spelling: ';', has_trailing_space: true },
a;
// CHECK:STDOUT:     { index: 3, kind: 'Identifier', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'a', identifier: 1 },
// CHECK:STDOUT:     { index: 4, kind:       'Semi', line: {{ *}}[[@LINE-2]], column:  2, indent: 1, spelling: ';', has_trailing_space: true },

// CHECK:STDOUT:     { index: 5, kind:    'FileEnd', line: {{ *}}[[@LINE+1]], column: {{ *\d+}}, indent: 1, spelling: '' },
// CHECK:STDOUT:   ]
