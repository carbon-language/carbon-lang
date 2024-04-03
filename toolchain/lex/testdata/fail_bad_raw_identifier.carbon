// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- fail_bad_raw_identifier.carbon
// CHECK:STDOUT: - filename: fail_bad_raw_identifier.carbon
// CHECK:STDOUT:   tokens: [
// CHECK:STDOUT:     { index:  0, kind:     'FileStart', line: {{ *\d+}}, column:  1, indent: 1, spelling: '', has_trailing_space: true },


// Missing the character after `#`.
// CHECK:STDERR: fail_bad_raw_identifier.carbon:[[@LINE+4]]:2: ERROR: Encountered unrecognized characters while parsing.
// CHECK:STDERR: r#
// CHECK:STDERR:  ^
// CHECK:STDERR:
r#
// CHECK:STDOUT:     { index:  1, kind:    'Identifier', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'r', identifier: 0 },
// CHECK:STDOUT:     { index:  2, kind:         'Error', line: {{ *}}[[@LINE-2]], column:  2, indent: 1, spelling: '#', has_trailing_space: true },

// Not a valid identifier.
// CHECK:STDERR: fail_bad_raw_identifier.carbon:[[@LINE+4]]:2: ERROR: Encountered unrecognized characters while parsing.
// CHECK:STDERR: r#3
// CHECK:STDERR:  ^
// CHECK:STDERR:
r#3
// CHECK:STDOUT:     { index:  3, kind:    'Identifier', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'r', identifier: 0 },
// CHECK:STDOUT:     { index:  4, kind:         'Error', line: {{ *}}[[@LINE-2]], column:  2, indent: 1, spelling: '#' },
// CHECK:STDOUT:     { index:  5, kind:    'IntLiteral', line: {{ *}}[[@LINE-3]], column:  3, indent: 1, spelling: '3', value: `3`, has_trailing_space: true },

// Non ascii start to identifier.
// CHECK:STDERR: fail_bad_raw_identifier.carbon:[[@LINE+4]]:2: ERROR: Encountered unrecognized characters while parsing.
// CHECK:STDERR: r#á
// CHECK:STDERR:  ^
// CHECK:STDERR:
r#á
// CHECK:STDOUT:     { index:  6, kind:    'Identifier', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'r', identifier: 0 },
// CHECK:STDOUT:     { index:  7, kind:         'Error', line: {{ *}}[[@LINE-2]], column:  2, indent: 1, spelling: '#á', has_trailing_space: true },

// Raw `r` identifier doesn't start a second raw identifier.
// CHECK:STDERR: fail_bad_raw_identifier.carbon:[[@LINE+4]]:4: ERROR: Encountered unrecognized characters while parsing.
// CHECK:STDERR: r#r#foo
// CHECK:STDERR:    ^
// CHECK:STDERR:
r#r#foo
// CHECK:STDOUT:     { index:  8, kind:    'Identifier', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'r', identifier: 0 },
// CHECK:STDOUT:     { index:  9, kind:         'Error', line: {{ *}}[[@LINE-2]], column:  4, indent: 1, spelling: '#' },
// CHECK:STDOUT:     { index: 10, kind:    'Identifier', line: {{ *}}[[@LINE-3]], column:  5, indent: 1, spelling: 'foo', identifier: 1, has_trailing_space: true },

// Other identifier characters don't start a raw identifier.
// CHECK:STDERR: fail_bad_raw_identifier.carbon:[[@LINE+4]]:2: ERROR: Encountered unrecognized characters while parsing.
// CHECK:STDERR: s#foo
// CHECK:STDERR:  ^
// CHECK:STDERR:
s#foo
// CHECK:STDOUT:     { index: 11, kind:    'Identifier', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 's', identifier: 2 },
// CHECK:STDOUT:     { index: 12, kind:         'Error', line: {{ *}}[[@LINE-2]], column:  2, indent: 1, spelling: '#' },
// CHECK:STDOUT:     { index: 13, kind:    'Identifier', line: {{ *}}[[@LINE-3]], column:  3, indent: 1, spelling: 'foo', identifier: 1, has_trailing_space: true },

// Identifier ending in `r` doesn't start a raw identifier.
// CHECK:STDERR: fail_bad_raw_identifier.carbon:[[@LINE+4]]:4: ERROR: Encountered unrecognized characters while parsing.
// CHECK:STDERR: arr#foo
// CHECK:STDERR:    ^
// CHECK:STDERR:
arr#foo
// CHECK:STDOUT:     { index: 14, kind:    'Identifier', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'arr', identifier: 3 },
// CHECK:STDOUT:     { index: 15, kind:         'Error', line: {{ *}}[[@LINE-2]], column:  4, indent: 1, spelling: '#' },
// CHECK:STDOUT:     { index: 16, kind:    'Identifier', line: {{ *}}[[@LINE-3]], column:  5, indent: 1, spelling: 'foo', identifier: 1, has_trailing_space: true },

// Whitespace between `r` and `#` isn't allowed.
// CHECK:STDERR: fail_bad_raw_identifier.carbon:[[@LINE+4]]:3: ERROR: Encountered unrecognized characters while parsing.
// CHECK:STDERR: r #foo
// CHECK:STDERR:   ^
// CHECK:STDERR:
r #foo
// CHECK:STDOUT:     { index: 17, kind:    'Identifier', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'r', identifier: 0, has_trailing_space: true },
// CHECK:STDOUT:     { index: 18, kind:         'Error', line: {{ *}}[[@LINE-2]], column:  3, indent: 1, spelling: '#' },
// CHECK:STDOUT:     { index: 19, kind:    'Identifier', line: {{ *}}[[@LINE-3]], column:  4, indent: 1, spelling: 'foo', identifier: 1, has_trailing_space: true },

// This is an `r` identifier followed by a string literal.
r#"hello"#
// CHECK:STDOUT:     { index: 20, kind:    'Identifier', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'r', identifier: 0 },
// CHECK:STDOUT:     { index: 21, kind: 'StringLiteral', line: {{ *}}[[@LINE-2]], column:  2, indent: 1, spelling: '#"hello"#', value: `hello`, has_trailing_space: true },

// CHECK:STDOUT:     { index: 22, kind:       'FileEnd', line: {{ *}}[[@LINE+1]], column: {{ *\d+}}, indent: 1, spelling: '' },
// CHECK:STDOUT:   ]
// --- fail_hash_at_start_of_file.carbon
// CHECK:STDOUT: - filename: fail_hash_at_start_of_file.carbon
// CHECK:STDOUT:   tokens: [
// CHECK:STDOUT:     { index: 0, kind:  'FileStart', line: {{ *\d+}}, column:  1, indent: 1, spelling: '', has_trailing_space: true },

// Ensure that we correctly handle a `#` as the first token in the file.
// CHECK:STDERR: fail_hash_at_start_of_file.carbon:[[@LINE+3]]:1: ERROR: Encountered unrecognized characters while parsing.
// CHECK:STDERR: #foo
// CHECK:STDERR: ^
#foo
// CHECK:STDOUT:     { index: 1, kind:      'Error', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: '#' },
// CHECK:STDOUT:     { index: 2, kind: 'Identifier', line: {{ *}}[[@LINE-2]], column:  2, indent: 1, spelling: 'foo', identifier: 0, has_trailing_space: true },

// CHECK:STDOUT:     { index: 3, kind:    'FileEnd', line: {{ *}}[[@LINE+1]], column: {{ *\d+}}, indent: 1, spelling: '' },
// CHECK:STDOUT:   ]
