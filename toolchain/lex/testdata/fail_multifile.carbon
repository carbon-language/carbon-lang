// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- fail_a.carbon
// CHECK:STDOUT: - filename: fail_a.carbon
// CHECK:STDOUT:   tokens: [
// CHECK:STDOUT:     { index: 0, kind: 'FileStart', line: {{ *\d+}}, column:  1, indent: 1, spelling: '', has_trailing_space: true },
// CHECK:STDERR: fail_a.carbon:[[@LINE+4]]:3: ERROR: Empty digit sequence in numeric literal.
// CHECK:STDERR: 1.a
// CHECK:STDERR:   ^
// CHECK:STDERR:
1.a
// CHECK:STDOUT:     { index: 1, kind:     'Error', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: '1.a', has_trailing_space: true },

// CHECK:STDOUT:     { index: 2, kind:   'FileEnd', line: {{ *}}[[@LINE+1]], column: {{ *\d+}}, indent: 1, spelling: '' },
// CHECK:STDOUT:   ]
// --- fail_b.carbon
// CHECK:STDOUT: - filename: fail_b.carbon
// CHECK:STDOUT:   tokens: [
// CHECK:STDOUT:     { index: 0, kind: 'FileStart', line: {{ *\d+}}, column:  1, indent: 1, spelling: '', has_trailing_space: true },
// CHECK:STDERR: fail_b.carbon:[[@LINE+3]]:3: ERROR: Empty digit sequence in numeric literal.
// CHECK:STDERR: 2.b
// CHECK:STDERR:   ^
2.b
// CHECK:STDOUT:     { index: 1, kind:     'Error', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: '2.b', has_trailing_space: true },

// CHECK:STDOUT:     { index: 2, kind:   'FileEnd', line: {{ *}}[[@LINE+1]], column: {{ *\d+}}, indent: 1, spelling: '' },
// CHECK:STDOUT:   ]
