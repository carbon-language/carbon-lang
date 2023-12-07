// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: - filename: raw_identifier.carbon
// CHECK:STDOUT:   tokens: [
// CHECK:STDOUT:     { index: 0, kind:           'FileStart', line: {{ *\d+}}, column:  1, indent: 1, spelling: '', has_trailing_space: true },

// A non-keyword identifier.
r#foo
// CHECK:STDOUT:     { index: 1, kind:          'Identifier', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'foo', identifier: 0, has_trailing_space: true },

// The same non-keyword identifier, for comparison.
foo
// CHECK:STDOUT:     { index: 2, kind:          'Identifier', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'foo', identifier: 0, has_trailing_space: true },

// A keyword as a raw identifier.
r#self
// CHECK:STDOUT:     { index: 3, kind:          'Identifier', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'self', identifier: 1, has_trailing_space: true },

// The same keyword, for comparison.
self
// CHECK:STDOUT:     { index: 4, kind: 'SelfValueIdentifier', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'self', has_trailing_space: true },

// A type literal as a raw identifier.
r#i32
// CHECK:STDOUT:     { index: 5, kind:          'Identifier', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'i32', identifier: 2, has_trailing_space: true },

// The same type literal, for comparison.
i32
// CHECK:STDOUT:     { index: 6, kind:      'IntTypeLiteral', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'i32', has_trailing_space: true },

// CHECK:STDOUT:     { index: 7, kind:             'FileEnd', line: {{ *}}[[@LINE+1]], column: {{ *\d+}}, indent: 1, spelling: '' },
// CHECK:STDOUT:   ]
