// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_require_semi.carbon:[[@LINE+3]]:12: ERROR: Expected `;` in array type.
// CHECK:STDERR: var x: [i32];
// CHECK:STDERR:            ^
var x: [i32];

// CHECK:STDOUT: - filename: fail_require_semi.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:       {kind: 'VariableIntroducer', text: 'var'},
// CHECK:STDOUT:         {kind: 'Name', text: 'x'},
// CHECK:STDOUT:             {kind: 'ArrayExpressionStart', text: '['},
// CHECK:STDOUT:             {kind: 'Literal', text: 'i32'},
// CHECK:STDOUT:           {kind: 'ArrayExpressionSemi', text: ']', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:         {kind: 'ArrayExpression', text: ']', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:       {kind: 'PatternBinding', text: ':', subtree_size: 6},
// CHECK:STDOUT:     {kind: 'VariableDeclaration', text: ';', subtree_size: 8},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
